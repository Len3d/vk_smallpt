// smallpt_lit.cl -- the "literal subscript" version of smallpt: it needs neither a constant
// address space nor a parameter buffer.
//
// Background (a real problem hit in this repository):
//   1. clspv has no OpenCL __constant address space, so a module-scope __constant array is demoted
//      to a Private (per-thread) array, see lines 538-539 of `_base.spvasm`:
//        %18 = OpVariable %_ptr_Private__arr_float_uint_42 Private %16   ; PLANES
//        %35 = OpVariable %_ptr_Private__arr_float_uint_48 Private %33   ; SPHERES
//   2. What really decides "do we go through Private" is not the address space but whether the
//      subscript is a constant. As soon as the subscript is a compile-time constant, clspv folds the
//      __constant array value into an immediate and the array itself does not even keep an
//      OpVariable.
//   3. The original smallpt.cl used `#pragma clang loop unroll(full)` to get the loop unrolled and
//      that turned out to have no effect (the Private accesses still counted 52, roughly 6x5 + 3x7
//      dynamic subscripts), because SPIRVProducer of clspv does not unroll on the pragma and the
//      loop variable i stays a run-time value.
//      So this file unrolls by hand: macros with literal subscripts write out the 6 walls and the
//      3 spheres one by one. The sources look repeated 6+3 times, and in exchange there are
//      0 Private accesses, 0 OpLoad of the tables and the scene tables do not exist at all
//      (everything becomes immediates).
//
// The semantics are identical to smallpt.cl (same planes/spheres, same test order, same RNG and
// same sampling).

#define PI 3.14159265358979f
#define EPS 1e-4f
#define BIG 1e20f
#define NSPHERE 3
#define NPLANE 6
#define MAX_DEPTH 10
#define CAM_PUSH 140.0f

#define MAT_DIFF 0
#define MAT_SPEC 1
#define MAT_GLASS 2

typedef struct { float x, y, z; } vec3;

static vec3 V3(float x, float y, float z) { vec3 v; v.x = x; v.y = y; v.z = z; return v; }
static vec3 add3(vec3 a, vec3 b) { return V3(a.x + b.x, a.y + b.y, a.z + b.z); }
static vec3 sub3(vec3 a, vec3 b) { return V3(a.x - b.x, a.y - b.y, a.z - b.z); }
static vec3 neg3(vec3 a) { return V3(-a.x, -a.y, -a.z); }
static vec3 scl3(vec3 a, float s) { return V3(a.x * s, a.y * s, a.z * s); }
static vec3 mul3(vec3 a, vec3 b) { return V3(a.x * b.x, a.y * b.y, a.z * b.z); }
static float dot3(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static vec3 cross3(vec3 a, vec3 b) {
  return V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static vec3 norm3(vec3 a) { return scl3(a, 1.0f / sqrt(dot3(a, a))); }
static float clamp1(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

// The scene tables stay __constant for readability, but every access below uses literal subscripts
__constant float PLANES[NPLANE * 7] = {
     1.0f,  0.0f,  0.0f,   1.0f,  0.75f, 0.25f, 0.25f, // left wall  (red)
    -1.0f,  0.0f,  0.0f, -99.0f,  0.25f, 0.25f, 0.75f, // right wall (blue)
     0.0f,  0.0f,  1.0f,   0.0f,  0.75f, 0.75f, 0.75f, // back wall  (grey)
     0.0f,  0.0f, -1.0f, -170.0f, 0.0f,  0.0f,  0.0f,  // front wall (black)
     0.0f,  1.0f,  0.0f,   0.0f,  0.75f, 0.75f, 0.75f, // floor      (grey)
     0.0f, -1.0f,  0.0f, -81.6f,  0.75f, 0.75f, 0.75f, // ceiling    (grey)
};

__constant float SPHERES[NSPHERE * 16] = {
    27.0f, 16.5f, 47.0f, 16.5f, 0.999f, 0.999f, 0.999f, 0.0f, 0.0f, 0.0f, MAT_SPEC, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    73.0f, 16.5f, 78.0f, 16.5f, 0.999f, 0.999f, 0.999f, 0.0f, 0.0f, 0.0f, MAT_GLASS, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    50.0f, 681.6f - 0.27f, 81.6f, 600.0f, 0.0f, 0.0f, 0.0f, 12.0f, 12.0f, 12.0f, MAT_DIFF, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};

typedef struct { uint s; } rng_t;

// The rng state stays a local scalar; the helper advances it through a pointer
static float rnd_step(uint *s) {
  *s = *s * 1664525u + 1013904223u;
  return (float)(*s >> 8) * (1.0f / 16777216.0f);
}

// ---- Planes: I must be a literal so that every PLANES[I*7+k] is a constant subscript ----
#define TRY_PLANE(I)                                                              \
  do {                                                                            \
    vec3 nrm_ = V3(PLANES[(I) * 7 + 0], PLANES[(I) * 7 + 1], PLANES[(I) * 7 + 2]); \
    float dd_ = dot3(nrm_, d);                                                     \
    if (!(dd_ > -1e-8f && dd_ < 1e-8f)) {                                          \
      float t_ = (PLANES[(I) * 7 + 3] - dot3(nrm_, o)) / dd_;                       \
      if (t_ > EPS && t_ < best) {                                                  \
        best = t_; id = (I); hn = nrm_; hm = MAT_DIFF; he = V3(0.0f, 0.0f, 0.0f);   \
        ha = V3(PLANES[(I) * 7 + 4], PLANES[(I) * 7 + 5], PLANES[(I) * 7 + 6]);     \
      }                                                                             \
    }                                                                               \
  } while (0)

// ---- Spheres: I must be a literal as well ----
#define TRY_SPHERE(I)                                                             \
  do {                                                                            \
    vec3 p_ = V3(SPHERES[(I) * 16 + 0], SPHERES[(I) * 16 + 1], SPHERES[(I) * 16 + 2]); \
    float rad_ = SPHERES[(I) * 16 + 3];                                            \
    vec3 op_ = sub3(p_, o);                                                        \
    float b_ = dot3(op_, d);                                                       \
    float c_ = dot3(op_, op_) - rad_ * rad_;                                       \
    float det_ = b_ * b_ - c_;                                                     \
    if (det_ >= 0.0f) {                                                            \
      float sq_ = sqrt(det_);                                                      \
      float t0_ = b_ - sq_;                                                        \
      if (t0_ > EPS && t0_ < best) {                                               \
        best = t0_; id = NPLANE + (I); hn = V3(0.0f, 0.0f, 0.0f); hc = p_;         \
        hm = (int)SPHERES[(I) * 16 + 10];                                          \
        ha = V3(SPHERES[(I) * 16 + 4], SPHERES[(I) * 16 + 5], SPHERES[(I) * 16 + 6]); \
        he = V3(SPHERES[(I) * 16 + 7], SPHERES[(I) * 16 + 8], SPHERES[(I) * 16 + 9]); \
      } else {                                                                     \
        float t1_ = b_ + sq_;                                                      \
        if (t1_ > EPS && t1_ < best) {                                             \
          best = t1_; id = NPLANE + (I); hn = V3(0.0f, 0.0f, 0.0f); hc = p_;       \
          hm = (int)SPHERES[(I) * 16 + 10];                                        \
          ha = V3(SPHERES[(I) * 16 + 4], SPHERES[(I) * 16 + 5], SPHERES[(I) * 16 + 6]); \
          he = V3(SPHERES[(I) * 16 + 7], SPHERES[(I) * 16 + 8], SPHERES[(I) * 16 + 9]); \
        }                                                                          \
      }                                                                            \
    }                                                                              \
  } while (0)

static vec3 trace(vec3 o, vec3 d, uint *rng) {
  vec3 L = V3(0.0f, 0.0f, 0.0f);
  vec3 thr = V3(1.0f, 1.0f, 1.0f);

  for (int depth = 0; depth < MAX_DEPTH; ++depth) {
    float best = BIG;
    int id = -1;
    vec3 hn = V3(0.0f, 0.0f, 0.0f);
    vec3 hc = V3(0.0f, 0.0f, 0.0f);
    vec3 ha = V3(0.0f, 0.0f, 0.0f);
    vec3 he = V3(0.0f, 0.0f, 0.0f);
    int hm = MAT_DIFF;

    TRY_PLANE(0); TRY_PLANE(1); TRY_PLANE(2);
    TRY_PLANE(3); TRY_PLANE(4); TRY_PLANE(5);

    TRY_SPHERE(0); TRY_SPHERE(1); TRY_SPHERE(2);

    if (id < 0) break; // miss = black

    vec3 x = add3(o, scl3(d, best));
    vec3 n = (id >= NPLANE) ? norm3(sub3(x, hc)) : hn;
    vec3 nl = dot3(n, d) < 0.0f ? n : neg3(n);
    vec3 f = ha;

    L = add3(L, mul3(thr, he));

    if (thr.x + thr.y + thr.z <= 0.0f) break;

    float pmax = f.x;
    if (f.y > pmax) pmax = f.y;
    if (f.z > pmax) pmax = f.z;
    if (depth > 5) {
      if (rnd_step(rng) < pmax) {
        f = scl3(f, 1.0f / pmax);
      } else {
        break;
      }
    }

    if (hm == MAT_DIFF) {
      float r1 = 2.0f * PI * rnd_step(rng);
      float r2 = rnd_step(rng);
      float r2s = sqrt(r2);
      vec3 w = nl;
      vec3 u = norm3(cross3((w.x > 0.1f || w.x < -0.1f) ? V3(0.0f, 1.0f, 0.0f) : V3(1.0f, 0.0f, 0.0f), w));
      vec3 v = cross3(w, u);
      vec3 nd = norm3(add3(add3(scl3(u, cos(r1) * r2s), scl3(v, sin(r1) * r2s)), scl3(w, sqrt(1.0f - r2))));
      thr = mul3(thr, f);
      o = x;
      d = nd;
      continue;
    } else if (hm == MAT_SPEC) {
      d = sub3(d, scl3(n, 2.0f * dot3(n, d)));
      thr = mul3(thr, f);
      o = x;
      continue;
    } else {
      int into = dot3(n, nl) > 0.0f;
      float nc = 1.0f, nt = 1.5f;
      float nnt = into ? (nc / nt) : (nt / nc);
      float ddn = dot3(d, nl);
      float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);
      vec3 refl = sub3(d, scl3(n, 2.0f * dot3(n, d)));
      if (cos2t < 0.0f) {
        d = refl;
        thr = mul3(thr, f);
        o = x;
        continue;
      }
      vec3 tdir = norm3(sub3(scl3(d, nnt), scl3(n, (into ? 1.0f : -1.0f) * (ddn * nnt + sqrt(cos2t)))));
      float a = nt - nc, b = nt + nc;
      float R0 = (a * a) / (b * b);
      float cc = 1.0f - (into ? (-ddn) : dot3(tdir, n));
      float c2 = cc * cc, c4 = c2 * c2, c5 = c4 * cc;
      float Re = R0 + (1.0f - R0) * c5;
      float Tr = 1.0f - Re;
      float P = 0.25f + 0.5f * Re;
      float RP = Re / P;
      float TP = Tr / (1.0f - P);
      if (rnd_step(rng) < P) {
        d = refl;
        thr = mul3(thr, scl3(f, RP));
      } else {
        d = tdir;
        thr = mul3(thr, scl3(f, TP));
      }
      o = x;
      continue;
    }
  }
  return L;
}

// First hit of a primary ray, used only to fill the denoiser's albedo / normal images. It reuses the
// same literal-subscript macros as trace(), so no subscript becomes dynamic here either.
static void first_hit(vec3 o, vec3 d, vec3 *albOut, vec3 *nrmOut) {
  float best = BIG;
  int id = -1;
  vec3 hn = V3(0.0f, 0.0f, 0.0f);
  vec3 hc = V3(0.0f, 0.0f, 0.0f);
  vec3 ha = V3(0.0f, 0.0f, 0.0f);
  vec3 he = V3(0.0f, 0.0f, 0.0f);
  int hm = MAT_DIFF;

  TRY_PLANE(0); TRY_PLANE(1); TRY_PLANE(2);
  TRY_PLANE(3); TRY_PLANE(4); TRY_PLANE(5);

  TRY_SPHERE(0); TRY_SPHERE(1); TRY_SPHERE(2);

  if (id >= 0) {
    *albOut = ha;
    *nrmOut = (id >= NPLANE) ? norm3(sub3(add3(o, scl3(d, best)), hc)) : hn;
  }
}

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void smallpt(__global float *img, __global float *albOut, __global float *nrmOut,
                      __global const uint *params) {
  const uint w = params[0];
  const uint h = params[1];
  const uint spp = params[2];
  const uint frame = params[3];

  const uint px = get_global_id(0);
  const uint py = get_global_id(1);
  if (px >= w || py >= h) return;

  const float aspect = (float)w / (float)h;
  const vec3 camO = V3(50.0f, 52.0f, 295.6f);
  const vec3 camD = norm3(V3(0.0f, -0.042612f, -1.0f));
  const vec3 cx = V3(aspect * 0.5135f, 0.0f, 0.0f);
  const vec3 cy = scl3(norm3(cross3(cx, camD)), 0.5135f);

  uint rng = (py * w + px) * 1973u + frame * 9277u + 1u;

  const float pyUp = (float)(h - 1u - py);
  vec3 rad = V3(0.0f, 0.0f, 0.0f);

  // ---- auxiliary images for the denoiser ----------------------------------
  // A miss leaves albedo black and the normal pointing at the camera, exactly like src/smallpt.cl.
  vec3 alb = V3(0.0f, 0.0f, 0.0f);
  vec3 nrm = V3(0.0f, 0.0f, 1.0f);
  {
    vec3 gDir = norm3(add3(add3(camD, scl3(cx, ((float)px + 0.5f) / (float)w - 0.5f)),
                            scl3(cy, ((float)pyUp + 0.5f) / (float)h - 0.5f)));
    first_hit(add3(camO, scl3(gDir, CAM_PUSH)), gDir, &alb, &nrm);
  }

  for (uint sy = 0; sy < 2; ++sy) {
    for (uint sx = 0; sx < 2; ++sx) {
      vec3 sub = V3(0.0f, 0.0f, 0.0f);
      for (uint s = 0; s < spp; ++s) {
        float r1 = 2.0f * rnd_step(&rng);
        float dx = (r1 < 1.0f) ? (sqrt(r1) - 1.0f) : (1.0f - sqrt(2.0f - r1));
        float r2 = 2.0f * rnd_step(&rng);
        float dy = (r2 < 1.0f) ? (sqrt(r2) - 1.0f) : (1.0f - sqrt(2.0f - r2));

        vec3 dRaw = add3(add3(scl3(cx, ((((float)sx + 0.5f + dx) * 0.5f) + (float)px) / (float)w - 0.5f),
                              scl3(cy, ((((float)sy + 0.5f + dy) * 0.5f) + pyUp) / (float)h - 0.5f)),
                         camD);
        vec3 d = norm3(dRaw);
        vec3 start = add3(camO, scl3(d, CAM_PUSH));
        sub = add3(sub, trace(start, d, &rng));
      }
      sub = scl3(sub, 1.0f / (float)spp);
      rad = add3(rad, V3(clamp1(sub.x), clamp1(sub.y), clamp1(sub.z)));
    }
  }
  rad = scl3(rad, 0.25f);

  const uint idx = (py * w + px) * 4;
  img[idx + 0] = rad.x;
  img[idx + 1] = rad.y;
  img[idx + 2] = rad.z;
  img[idx + 3] = 1.0f;

  const uint aidx = (py * w + px) * 3;
  albOut[aidx + 0] = alb.x;
  albOut[aidx + 1] = alb.y;
  albOut[aidx + 2] = alb.z;
  nrmOut[aidx + 0] = nrm.x;
  nrmOut[aidx + 1] = nrm.y;
  nrmOut[aidx + 2] = nrm.z;
}
