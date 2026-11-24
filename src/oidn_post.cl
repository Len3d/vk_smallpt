// oidn_post.cl -- OIDN output reorder: inverse PU, undo the autoexposure, write the RGBA image,
// written for clspv (C++ for OpenCL), the counterpart of src/oidn_post.comp.
//
// Parameter words (same order as the GLSL `Pc`): 0 rcpNormScale, 1 outputScale, 2 tileW, 3 tileH,
// 4 imgW, 5 outX, 6 outY, 7 outW, 8 outH, 9 ofsX, 10 ofsY.

#if defined(OIDN_HALF_TENSOR)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if defined(OIDN_HALF_TENSOR)
#define TENSOR_PTR half
#else
#define TENSOR_PTR float4
#endif

#define PC_INT(p, i) ((int)(p)[i])
#define PC_FLT(p, i) (as_float((p)[i]))

constant float PU_A = 1.41283765e+03f;
constant float PU_B = 1.64593172e+00f;
constant float PU_C = 4.31384981e-01f;
constant float PU_D = -2.94139609e-03f;
constant float PU_E = 1.92653254e-01f;
constant float PU_F = 6.26026094e-03f;
constant float PU_G = 9.98620152e-01f;
constant float PU_X0 = 2.23151711e-03f;
constant float PU_X1 = 3.70974749e-01f;
constant float FLT_MAX_V = 3.4028235e+38f;

inline float oidn_pu_inverse(float x) {
  if (x <= PU_X0) {
    return x / PU_A;
  } else if (x <= PU_X1) {
    return pow((x - PU_D) / PU_B, 1.0f / PU_C);
  } else {
    return exp((x - PU_G) / PU_E) - PU_F;
  }
}

inline float oidn_sanitize_out(float v) {
  if (isnan(v) || v < 0.0f) return 0.0f;
  return fmin(v, FLT_MAX_V);
}

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void oidn_post(__global const TENSOR_PTR *src, __global float *dst,
                        __global const uint *params) {
  const int x = (int)get_global_id(0);
  const int y = (int)get_global_id(1);
  const float rcpNormScale = PC_FLT(params, 0);
  const float outputScale = PC_FLT(params, 1);
  const int tileW = PC_INT(params, 2);
  const int tileH = PC_INT(params, 3);
  if (x >= tileW || y >= tileH) return;

  const int imgW = PC_INT(params, 4);
  const int outX = PC_INT(params, 5);
  const int outY = PC_INT(params, 6);
  const int outW = PC_INT(params, 7);
  const int outH = PC_INT(params, 8);
  const int ofsX = PC_INT(params, 9);
  const int ofsY = PC_INT(params, 10);

  const int vx = x - ofsX;
  const int vy = y - ofsY;
  if (vx < 0 || vy < 0 || vx >= outW || vy >= outH) return;

  const int srcBase = (y * tileW + x) * 4;  // the output tensor has 3 channels padded to 4
  const int dstIdx = ((outY + vy) * imgW + (outX + vx)) * 4;

#if defined(OIDN_HALF_TENSOR)
  float v0 = (float)src[srcBase + 0];
  float v1 = (float)src[srcBase + 1];
  float v2 = (float)src[srcBase + 2];
#else
  float4 c = src[srcBase >> 2];
  float v0 = c.x;
  float v1 = c.y;
  float v2 = c.z;
#endif

  dst[dstIdx + 0] = oidn_pu_inverse(oidn_sanitize_out(v0) * rcpNormScale) * outputScale;
  dst[dstIdx + 1] = oidn_pu_inverse(oidn_sanitize_out(v1) * rcpNormScale) * outputScale;
  dst[dstIdx + 2] = oidn_pu_inverse(oidn_sanitize_out(v2) * rcpNormScale) * outputScale;
  dst[dstIdx + 3] = 1.0f;
}
