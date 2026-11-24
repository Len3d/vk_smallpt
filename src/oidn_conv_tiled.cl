// oidn_conv_tiled.cl -- 3x3 convolution with padding 1, stride 1, optional fused ReLU, using
// shared memory tiling and register blocking, written for clspv (C++ for OpenCL); the counterpart
// of the default GLSL pass src/oidn_conv_tiled.comp.
//
// Decomposition (parameterised by OIDN_PX / OIDN_PY / OIDN_GROUP):
//   - one work group of 8x8 threads covers a (8*PX)x(8*PY) block of output pixels for OIDN_GROUP
//     output channels
//   - every thread computes a PXxPY pixel block for those channels (register blocking), so one
//     set of 9 kernel taps is reused OIDN_PX*OIDN_PY times
//   - one input channel chunk (4 channels) is staged in __local memory ((8*PX+2)x(8*PY+2) per
//     channel), so the input is fetched once per chunk instead of per thread
//
// The load per multiply-add ratio is what the register blocking buys: a (o, i) weight row is 12
// floats (9 taps padded to 12) and covers PX*PY*9 multiply-adds, so 2x2 blocking pays 12 loads per
// 36 MACs and 4x4 blocking pays 12 per 144.
//
// The FP32 tensors are declared as float4 so that a 4-channel chunk is exactly one 16-byte load;
// the FP16 tensors stay scalar because a 4-half vector is only 8 bytes and the conversion is
// elementwise anyway.
//
// Buffer order decides the bindings: 0 = source tensor, 1 = destination tensor, 2 = weights,
// 3 = parameters.
//
// Parameter words (same order as the GLSL `Pc`): 0 cin, 1 cout, 2 width, 3 height, 4 relu,
// 5 weightOfs.

#if defined(OIDN_HALF_TENSOR)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if defined(OIDN_HALF_TENSOR)
#define TENSOR_PTR half
#else
#define TENSOR_PTR float4
#endif

#define PC_INT(p, i) ((int)(p)[i])

// Defaults are the configuration that measured fastest on this GPU: three pixels across, two
// down, four output channels per work group (block 24x16). The other blockings are reachable with
// -DOIDN_PX/-DOIDN_PY/-DOIDN_GROUP and are built as separate modules (OIDN_TILE selects them at
// run time).
#ifndef OIDN_PX
#define OIDN_PX 3
#endif
#ifndef OIDN_PY
#define OIDN_PY 2
#endif
#ifndef OIDN_GROUP
#define OIDN_GROUP 4
#endif

#define GROUP OIDN_GROUP
#define BLOCK_X (8 * OIDN_PX)         // output pixels per block, per dimension
#define BLOCK_Y (8 * OIDN_PY)
#define PATCH_X (BLOCK_X + 2)         // block plus one pixel of padding on each side
#define PATCH_Y (BLOCK_Y + 2)
#define PATCH_PIX (PATCH_X * PATCH_Y)
#define STAGE_CHANNELS 4              // input channels staged at once (one blocked chunk)

inline int oidn_pad4(int c) { return (c + 3) & -4; }

// The staged patch is 4 planes of PATCH_PIX floats. OpenCL has no module-scope local variables, so
// it is declared inside the kernel and handed to the helpers as a pointer.
inline void oidn_stage4(__local float *tileBuf, __global const TENSOR_PTR *src, int base, int p) {
#if defined(OIDN_HALF_TENSOR)
  tileBuf[0 * PATCH_PIX + p] = (float)src[base + 0];
  tileBuf[1 * PATCH_PIX + p] = (float)src[base + 1];
  tileBuf[2 * PATCH_PIX + p] = (float)src[base + 2];
  tileBuf[3 * PATCH_PIX + p] = (float)src[base + 3];
#else
  float4 v = src[base >> 2];
  tileBuf[0 * PATCH_PIX + p] = v.x;
  tileBuf[1 * PATCH_PIX + p] = v.y;
  tileBuf[2 * PATCH_PIX + p] = v.z;
  tileBuf[3 * PATCH_PIX + p] = v.w;
#endif
}

inline void oidn_stage4_zero(__local float *tileBuf, int p) {
  tileBuf[0 * PATCH_PIX + p] = 0.0f;
  tileBuf[1 * PATCH_PIX + p] = 0.0f;
  tileBuf[2 * PATCH_PIX + p] = 0.0f;
  tileBuf[3 * PATCH_PIX + p] = 0.0f;
}

// Store 4 output channels at float offset `base` (a multiple of 4).
inline void oidn_store4(__global TENSOR_PTR *dst, int base, float4 v) {
#if defined(OIDN_HALF_TENSOR)
  dst[base + 0] = (half)v.x;
  dst[base + 1] = (half)v.y;
  dst[base + 2] = (half)v.z;
  dst[base + 3] = (half)v.w;
#else
  dst[base >> 2] = v;
#endif
}

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void oidn_conv_tiled(__global const TENSOR_PTR *src, __global TENSOR_PTR *dst,
                              __global const float *wgt, __global const uint *params) {
  __local float tileBuf[STAGE_CHANNELS * PATCH_PIX];
  const int lx = (int)get_local_id(0);  // 0..7
  const int ly = (int)get_local_id(1);  // 0..7
  const int tid = ly * 8 + lx;
  // The block origin comes from the work group index: get_global_id(0) already includes the local
  // size, so the pixel block lower left corner is derived from the group index instead.
  const int x0 = (int)get_group_id(0) * BLOCK_X;
  const int y0 = (int)get_group_id(1) * BLOCK_Y;
  const int og = (int)get_global_id(2) * GROUP;

  const int cin = PC_INT(params, 0);
  const int cout = PC_INT(params, 1);
  const int w = PC_INT(params, 2);
  const int h = PC_INT(params, 3);
  const int relu = PC_INT(params, 4);
  const int weightOfs = PC_INT(params, 5);
  if (x0 >= w || y0 >= h || og >= cout) return;

  const int srcStride = oidn_pad4(cin);   // channel stride of the source tensor
  const int dstStride = oidn_pad4(cout);  // channel stride of the destination tensor
  const int biasOfs = weightOfs + cout * cin * 12;

  // Accumulators: PX*PY pixels x GROUP output channels. Channels past cout are padding in the
  // destination tensor and must be written deterministically, so they start at zero.
  float acc[OIDN_PY][OIDN_PX][GROUP];
#pragma unroll
  for (int g = 0; g < GROUP; ++g) {
    const int o = og + g;
    const float b = (o < cout) ? wgt[biasOfs + o] : 0.0f;
#pragma unroll
    for (int py = 0; py < OIDN_PY; ++py) {
#pragma unroll
      for (int px = 0; px < OIDN_PX; ++px) acc[py][px][g] = b;
    }
  }

  for (int c0 = 0; c0 < cin; c0 += STAGE_CHANNELS) {
    const int nch = (cin - c0 < STAGE_CHANNELS) ? (cin - c0) : STAGE_CHANNELS;

    // Stage the patch of this 4-channel chunk: one blocked chunk per patch pixel, one thread per
    // patch pixel, 4 channels at a time.
    for (int p = tid; p < PATCH_PIX; p += 64) {
      const int px = p % PATCH_X;  // patch column, -1..BLOCK_X relative to the block
      const int py = p / PATCH_X;
      const int sx = x0 + px - 1;
      const int sy = y0 + py - 1;
      if (sx >= 0 && sx < w && sy >= 0 && sy < h)
        oidn_stage4(tileBuf, src, (sy * w + sx) * srcStride + c0, p);
      else
        oidn_stage4_zero(tileBuf, p);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int c = 0; c < nch; ++c) {
      const int i = c0 + c;
      // (PX+2)x(PY+2) voxel window covering the pixel block, read once and reused by all outputs
      float v[OIDN_PY + 2][OIDN_PX + 2];
#pragma unroll
      for (int dy = 0; dy < OIDN_PY + 2; ++dy) {
#pragma unroll
        for (int dx = 0; dx < OIDN_PX + 2; ++dx) {
          v[dy][dx] = tileBuf[c * PATCH_PIX + (ly * OIDN_PY + dy) * PATCH_X + (lx * OIDN_PX + dx)];
        }
      }

      // The k-channel weights are hoisted out of the pixel loops, and unrolling the pixel loops
      // turns the accumulators of every output channel into registers: without the hint clspv keeps
      // the acc[] array in function memory and pays an access chain per tap, which made the first
      // clc build 1.5x slower than the GLSL pass.
#pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        const int o = og + g;
        if (o >= cout) break;
        const int base = weightOfs + (o * cin + i) * 12;
        const float k0 = wgt[base + 0], k1 = wgt[base + 1], k2 = wgt[base + 2];
        const float k3 = wgt[base + 3], k4 = wgt[base + 4], k5 = wgt[base + 5];
        const float k6 = wgt[base + 6], k7 = wgt[base + 7], k8 = wgt[base + 8];
#pragma unroll
        for (int py = 0; py < OIDN_PY; ++py) {
#pragma unroll
          for (int px = 0; px < OIDN_PX; ++px) {
            const float s =
                k0 * v[py + 0][px + 0] + k1 * v[py + 0][px + 1] + k2 * v[py + 0][px + 2] +
                k3 * v[py + 1][px + 0] + k4 * v[py + 1][px + 1] + k5 * v[py + 1][px + 2] +
                k6 * v[py + 2][px + 0] + k7 * v[py + 2][px + 1] + k8 * v[py + 2][px + 2];
            acc[py][px][g] += s;
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);  // the next chunk overwrites the staged patch
  }

  // Store: one blocked 4-channel chunk per store, so a pixel costs GROUP/4 stores instead of GROUP.
  // og is a multiple of GROUP and dstStride a multiple of 4, so every store offset stays 16-byte
  // aligned.
  for (int g0 = 0; g0 < GROUP; g0 += 4) {
    if (og + g0 + 4 > dstStride) break;
#pragma unroll
    for (int py = 0; py < OIDN_PY; ++py) {
      const int y = y0 + ly * OIDN_PY + py;
      if (y >= h) break;
#pragma unroll
      for (int px = 0; px < OIDN_PX; ++px) {
        const int x = x0 + lx * OIDN_PX + px;
        if (x >= w) break;
        float4 r = (float4)(acc[py][px][g0 + 0], acc[py][px][g0 + 1], acc[py][px][g0 + 2],
                            acc[py][px][g0 + 3]);
        if (relu != 0) r = fmax(r, (float4)(0.0f));
        oidn_store4(dst, (y * w + x) * dstStride + og + g0, r);
      }
    }
  }
}
