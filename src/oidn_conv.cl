// oidn_conv.cl -- 3x3 convolution, padding 1, stride 1, optional fused ReLU, one output pixel per
// thread, written for clspv (C++ for OpenCL), the counterpart of src/oidn_conv.comp.
//
// This is the "simple" kernel (OIDN_CONV=simple keeps it selectable); src/oidn_conv_tiled.cl is
// the shared-memory / register-blocked one and the default.
//
// Buffer order decides the bindings: 0 = source tensor, 1 = destination tensor, 2 = weights,
// 3 = parameters. Weights are laid out oihw with every 3x3 kernel row padded to 12 floats
// (tools/oidn/export_oidn_weights.py).
//
// Parameter words (same order as the GLSL `Pc`): 0 cin, 1 cout, 2 width, 3 height, 4 relu,
// 5 weightOfs.

#if defined(OIDN_HALF_TENSOR)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if defined(OIDN_HALF_TENSOR)
#define TENSOR_T half
#define TENSOR_PTR half
#else
#define TENSOR_T float
#define TENSOR_PTR float
#endif

#define PC_INT(p, i) ((int)(p)[i])

#define GROUP 8

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void oidn_conv(__global const TENSOR_PTR *src, __global TENSOR_PTR *dst,
                        __global const float *wgt, __global const uint *params) {
  const int x = (int)get_global_id(0);
  const int y = (int)get_global_id(1);
  const int og = (int)get_global_id(2) * GROUP;
  const int cin = PC_INT(params, 0);
  const int cout = PC_INT(params, 1);
  const int w = PC_INT(params, 2);
  const int h = PC_INT(params, 3);
  const int relu = PC_INT(params, 4);
  const int weightOfs = PC_INT(params, 5);
  if (x >= w || y >= h || og >= cout) return;

  const int srcStride = (cin + 3) & -4;   // channel-blocked: stride padded to a multiple of 4
  const int dstStride = (cout + 3) & -4;
  const int biasOfs = weightOfs + cout * cin * 12;

  float acc[GROUP];
  for (int g = 0; g < GROUP; ++g) {
    const int o = og + g;
    acc[g] = (o < cout) ? wgt[biasOfs + o] : 0.0f;
  }

  for (int i = 0; i < cin; ++i) {
    // Load the 3x3 neighbourhood of input channel i once, then reuse it for all GROUP outputs.
    float v[9];
    for (int t = 0; t < 9; ++t) {
      const int sx = x + (t % 3) - 1;
      const int sy = y + (t / 3) - 1;
      v[t] = (sx >= 0 && sx < w && sy >= 0 && sy < h)
                 ? (float)src[(sy * w + sx) * srcStride + i]
                 : 0.0f;
    }
    const int rowBase = weightOfs + i * 12;
    for (int g = 0; g < GROUP; ++g) {
      const int o = og + g;
      if (o >= cout) break;
      const int base = rowBase + o * cin * 12;
      float s = wgt[base + 0] * v[0] + wgt[base + 1] * v[1] + wgt[base + 2] * v[2] +
                wgt[base + 3] * v[3] + wgt[base + 4] * v[4] + wgt[base + 5] * v[5] +
                wgt[base + 6] * v[6] + wgt[base + 7] * v[7] + wgt[base + 8] * v[8];
      acc[g] += s;
    }
  }

  for (int g = 0; g < GROUP; ++g) {
    const int o = og + g;
    if (o >= cout) break;
    float r = acc[g];
    if (relu != 0) r = fmax(r, 0.0f);
    dst[(y * w + x) * dstStride + o] = (TENSOR_T)r;
  }
}
