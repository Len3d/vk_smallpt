// oidn_upsample.cl -- 2x nearest-neighbour upsampling into a channel slice of the concat buffer,
// written for clspv (C++ for OpenCL), the counterpart of src/oidn_upsample.comp.
//
// Parameter words (same order as the GLSL `Pc`): 0 channels, 1 srcWidth, 2 srcHeight, 3 dstWidth,
// 4 dstHeight, 5 dstChannels, 6 dstOfsC.

#if defined(OIDN_HALF_TENSOR)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if defined(OIDN_HALF_TENSOR)
#define TENSOR_PTR half
#else
#define TENSOR_PTR float4
#endif

#define PC_INT(p, i) ((int)(p)[i])

inline int oidn_pad4(int c) { return (c + 3) & -4; }

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void oidn_upsample(__global const TENSOR_PTR *src, __global TENSOR_PTR *dst,
                            __global const uint *params) {
  const int dx = (int)get_global_id(0);
  const int dy = (int)get_global_id(1);
  const int c0 = (int)get_global_id(2) * 4;
  const int channels = PC_INT(params, 0);
  const int srcWidth = PC_INT(params, 1);
  const int dstWidth = PC_INT(params, 3);
  const int dstHeight = PC_INT(params, 4);
  const int dstChannels = PC_INT(params, 5);
  const int dstOfsC = PC_INT(params, 6);
  const int srcStride = oidn_pad4(channels);
  const int dstStride = oidn_pad4(dstChannels);
  if (dx >= dstWidth || dy >= dstHeight || c0 >= srcStride) return;

  const int sx = dx / 2;
  const int sy = dy / 2;
  const int srcBase = (sy * srcWidth + sx) * srcStride + c0;
  const int dstBase = (dy * dstWidth + dx) * dstStride + dstOfsC + c0;

#if defined(OIDN_HALF_TENSOR)
  dst[dstBase + 0] = src[srcBase + 0];
  dst[dstBase + 1] = src[srcBase + 1];
  dst[dstBase + 2] = src[srcBase + 2];
  dst[dstBase + 3] = src[srcBase + 3];
#else
  dst[dstBase >> 2] = src[srcBase >> 2];
#endif
}
