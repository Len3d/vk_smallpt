// oidn_pool.cl -- 2x2 max pooling, written for clspv (C++ for OpenCL), the counterpart of
// src/oidn_pool.comp.
//
// Interface differences from the GLSL pass, forced by clspv:
//   * the per-pass parameters arrive in a uint32 buffer (binding 2) instead of the push constant
//     block: clspv always emits a module-scope push constant for the OpenCL 2.0 global offset in
//     CLC++/CL2.0 mode, and --pod-pushconstant refuses to coexist with it (see the README);
//   * the kernel is named after the pass (clspv cannot emit an entry point called "main"), and the
//     workgroup size arrives as spec constants that the build freezes to 8/8/1 with spirv-opt.
//
// The tensors use the channel-blocked layout, so one thread owns one 4-channel chunk of one
// destination pixel and the four source pixels are four aligned 16-byte (FP32) / 4x2-byte (FP16)
// loads, exactly like the GLSL pass.
//
// Parameter words (all 4-byte, same order as the GLSL `Pc` struct): 0 channels, 1 dstWidth,
// 2 dstHeight.

#if defined(OIDN_HALF_TENSOR)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if defined(OIDN_HALF_TENSOR)
#define TENSOR_T half
#define TENSOR_PTR half
#else
#define TENSOR_T float4
#define TENSOR_PTR float4
#endif

#define PC_INT(p, i) ((int)(p)[i])

inline int oidn_pad4(int c) { return (c + 3) & -4; }

#if defined(OIDN_HALF_TENSOR)
inline float4 oidn_ld(__global const TENSOR_PTR *src, int base) {
  return (float4)(src[base + 0], src[base + 1], src[base + 2], src[base + 3]);
}
inline void oidn_st(__global TENSOR_PTR *dst, int base, float4 v) {
  dst[base + 0] = v.x;
  dst[base + 1] = v.y;
  dst[base + 2] = v.z;
  dst[base + 3] = v.w;
}
#else
inline float4 oidn_ld(__global const TENSOR_PTR *src, int base) { return src[base >> 2]; }
inline void oidn_st(__global TENSOR_PTR *dst, int base, float4 v) { dst[base >> 2] = v; }
#endif

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void oidn_pool(__global const TENSOR_PTR *src, __global TENSOR_PTR *dst,
                        __global const uint *params) {
  const int x = (int)get_global_id(0);
  const int y = (int)get_global_id(1);
  const int c0 = (int)get_global_id(2) * 4;
  const int channels = PC_INT(params, 0);
  const int dstWidth = PC_INT(params, 1);
  const int dstHeight = PC_INT(params, 2);
  const int stride = oidn_pad4(channels);
  if (x >= dstWidth || y >= dstHeight || c0 >= stride) return;

  const int srcW = dstWidth * 2;
  const int sx = x * 2;
  const int sy = y * 2;

  float4 m = oidn_ld(src, ((sy + 0) * srcW + sx + 0) * stride + c0);
  m = fmax(m, oidn_ld(src, ((sy + 0) * srcW + sx + 1) * stride + c0));
  m = fmax(m, oidn_ld(src, ((sy + 1) * srcW + sx + 0) * stride + c0));
  m = fmax(m, oidn_ld(src, ((sy + 1) * srcW + sx + 1) * stride + c0));
  oidn_st(dst, (y * dstWidth + x) * stride + c0, m);
}
