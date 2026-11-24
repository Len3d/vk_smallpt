// oidn_pre.cl -- OIDN input reorder, written for clspv (C++ for OpenCL), the counterpart of
// src/oidn_pre.comp. Same three channel groups and the same tile mapping:
//   color  : * inputScale, clamp [0, 65504], PU curve, * normScale
//   albedo : clamp [0, 1], nothing else
//   normal : clamp [-1, 1], then v * 0.5 + 0.5
//
// Buffer order decides the binding numbers (clspv numbers the pointer arguments it keeps, starting
// at 0): 0 = color image, 1 = destination tensor, 2 = albedo image, 3 = normal image,
// 4 = parameters. Note that this differs from the GLSL layout (which puts albedo / normal at
// bindings 3 and 4), so src/oidn_denoise.cpp writes the clc bindings from a per-pass table.
//
// Parameter words (same order as the GLSL `Pc`): 0 inputScale, 1 normScale, 2 tileW, 3 tileH,
// 4 imgW, 5 srcX, 6 srcY, 7 validW, 8 validH, 9 ofsX, 10 ofsY, 11 cin, 12 nAux.

#if defined(OIDN_HALF_TENSOR)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if defined(OIDN_HALF_TENSOR)
#define TENSOR_PTR half
#else
#define TENSOR_PTR float
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
constant float PU_Y0 = 1.57945760e-06f;
constant float PU_Y1 = 3.22087631e-02f;
constant float HDR_Y_MAX = 65504.0f;

inline float oidn_pu_forward(float y) {
  if (y <= PU_Y0) {
    return PU_A * y;
  } else if (y <= PU_Y1) {
    return PU_B * pow(y, PU_C) + PU_D;
  } else {
    return PU_E * log(y + PU_F) + PU_G;
  }
}

inline float oidn_sanitize(float v) {
  // nan_to_zero(v), as in input_reorder.ispc
  if (isnan(v)) return 0.0f;
  return v;
}

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void oidn_pre(__global const float *src, __global TENSOR_PTR *dst,
                       __global const float *alb, __global const float *nrm,
                       __global const uint *params) {
  const int x = (int)get_global_id(0);
  const int y = (int)get_global_id(1);
  const float inputScale = PC_FLT(params, 0);
  const float normScale = PC_FLT(params, 1);
  const int tileW = PC_INT(params, 2);
  const int tileH = PC_INT(params, 3);
  if (x >= tileW || y >= tileH) return;

  const int imgW = PC_INT(params, 4);
  const int srcX = PC_INT(params, 5);
  const int srcY = PC_INT(params, 6);
  const int validW = PC_INT(params, 7);
  const int validH = PC_INT(params, 8);
  const int ofsX = PC_INT(params, 9);
  const int ofsY = PC_INT(params, 10);
  const int cin = PC_INT(params, 11);
  const int nAux = PC_INT(params, 12);

  // Channel-blocked layout: the tensor's channel stride is cin padded to a multiple of 4.
  const int stride = (cin + 3) & -4;
  const int dstBase = (y * tileW + x) * stride;

  const int vx = x - ofsX;
  const int vy = y - ofsY;
  if (vx < 0 || vy < 0 || vx >= validW || vy >= validH) {
    // Zero padding of the tile (alignment slack and the area outside the image), including the
    // blocked padding channels so the tensor never holds uninitialized values.
    for (int c = 0; c < stride; ++c) dst[dstBase + c] = 0.0f;
    return;
  }

  const int pix = (srcY + vy) * imgW + (srcX + vx);
  const int srcIdx = pix * 4;

  // Color: scale, clamp to [0, 65504], PU curve
  for (int c = 0; c < 3; ++c) {
    float v = oidn_sanitize(src[srcIdx + c]) * inputScale;
    v = clamp(v, 0.0f, HDR_Y_MAX);
    dst[dstBase + c] = oidn_pu_forward(v) * normScale;
  }

  // Albedo: clamp to [0, 1], untouched
  if (nAux >= 1) {
    for (int c = 0; c < 3; ++c) {
      dst[dstBase + 3 + c] = clamp(oidn_sanitize(alb[srcIdx + c]), 0.0f, 1.0f);
    }
  }

  // Blocked padding channels: zero them so the consumer never reads uninitialized memory.
  for (int c = cin; c < stride; ++c) dst[dstBase + c] = 0.0f;

  // Normal: clamp to [-1, 1], remap to [0, 1]
  if (nAux >= 2) {
    for (int c = 0; c < 3; ++c) {
      float v = clamp(oidn_sanitize(nrm[srcIdx + c]), -1.0f, 1.0f);
      dst[dstBase + 6 + c] = v * 0.5f + 0.5f;
    }
  }
}
