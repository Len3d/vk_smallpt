// oidn_concat_copy.cl -- copy a skip connection into a channel slice of a concat buffer, written
// for clspv (C++ for OpenCL), the counterpart of src/oidn_concat_copy.comp.
//
// Parameter words (same order as the GLSL `Pc`): 0 channels, 1 width, 2 height, 3 srcChannels,
// 4 dstChannels, 5 dstOfsC.

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
__kernel void oidn_concat_copy(__global const TENSOR_PTR *src, __global TENSOR_PTR *dst,
                               __global const uint *params) {
  const int x = (int)get_global_id(0);
  const int y = (int)get_global_id(1);
  const int c0 = (int)get_global_id(2) * 4;
  const int channels = PC_INT(params, 0);
  const int width = PC_INT(params, 1);
  const int height = PC_INT(params, 2);
  const int srcChannels = PC_INT(params, 3);
  const int dstChannels = PC_INT(params, 4);
  const int dstOfsC = PC_INT(params, 5);
  const int srcStride = oidn_pad4(srcChannels);
  const int dstStride = oidn_pad4(dstChannels);
  if (x >= width || y >= height || c0 >= oidn_pad4(channels)) return;

  const int srcBase = (y * width + x) * srcStride + c0;
  const int dstBase = (y * width + x) * dstStride + dstOfsC + c0;

#if defined(OIDN_HALF_TENSOR)
  dst[dstBase + 0] = src[srcBase + 0];
  dst[dstBase + 1] = src[srcBase + 1];
  dst[dstBase + 2] = src[srcBase + 2];
  dst[dstBase + 3] = src[srcBase + 3];
#else
  dst[dstBase >> 2] = src[srcBase >> 2];
#endif
}
