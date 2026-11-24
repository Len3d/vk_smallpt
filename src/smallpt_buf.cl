// smallpt_buf.cl -- control experiment: identical logic to src/smallpt.cl, the only difference
// being that **the scene data is not a `__constant` array but is read from the parameter buffer**.
//
// Why: clspv demotes program-scope `__constant` arrays to Private storage (per-thread scratch), so
// the hot loop carries 52 dynamic "OpAccessChain %_ptr_Private" accesses and measured more than 5x
// slower than the GLSL version built by glslc. Reading from a `__global` buffer instead (same data,
// same indices) checks whether the slowdown comes from the way __constant arrays are demoted rather
// than from the code generation quality of clspv.
//
// Parameter buffer layout (one buffer serves both purposes, so no extra descriptor is needed):
//   [0..3]   uint  w, h, spp, frame
//   [4..]    float scene data, see the OFF_* offsets below (filled by src/smallpt_main.cpp)
//
// The host side is the same apart from the third argument: the entry point is still smallpt,
// binding 0 = image, binding 1 = parameters + scene.

#define PI 3.14159265358979f
#define EPS 1e-4f
#define BIG 1e20f
#define NPLANE 6
#define NSPHERE 3
#define MAX_DEPTH 10
#define CAM_PUSH 140.0f

#define MAT_DIFF 0
#define MAT_SPEC 1
#define MAT_GLASS 2

// Offsets of the scene data inside the parameter buffer, in floats counted from the start of
// params (the first 4 entries are the uint header)
#define SC_BASE 4
#define OFF_PLANE_GEO 0                      // 6*4: nx,ny,nz,d
#define OFF_PLANE_COL (6 * 4)                // 6*3: r,g,b
#define OFF_SPH_GEO (6 * 3 + 6 * 4)          // 3*4: px,py,pz,radius
#define OFF_SPH_COL (OFF_SPH_GEO + 3 * 4)     // 3*3
#define OFF_SPH_EMIT (OFF_SPH_COL + 3 * 3)    // 3*3
#define OFF_SPH_MAT (OFF_SPH_EMIT + 3 * 3)    // 3*1
#define SC_FLOATS (OFF_SPH_MAT + 3)

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

// sc points at the start of the scene data
static vec3 pl_normal(const __global float *sc, int i) {
  int o = SC_BASE + OFF_PLANE_GEO + i * 4;
  return V3(sc[o], sc[o + 1], sc[o + 2]);
}
static float pl_d(const __global float *sc, int i) { return sc[SC_BASE + OFF_PLANE_GEO + i * 4 + 3]; }
static vec3 pl_color(const __global float *sc, int i) {
  int o = SC_BASE + OFF_PLANE_COL + i * 3;
  return V3(sc[o], sc[o + 1], sc[o + 2]);
}

static vec3 sph_center(const __global float *sc, int i) {
  int o = SC_BASE + OFF_SPH_GEO + i * 4;
  return V3(sc[o], sc[o + 1], sc[o + 2]);
}
static float sph_rad(const __global float *sc, int i) { return sc[SC_BASE + OFF_SPH_GEO + i * 4 + 3]; }
static vec3 sph_color(const __global float *sc, int i) {
  int o = SC_BASE + OFF_SPH_COL + i * 3;
  return V3(sc[o], sc[o + 1], sc[o + 2]);
}
static vec3 sph_emit(const __global float *sc, int i) {
  int o = SC_BASE + OFF_SPH_EMIT + i * 3;
  return V3(sc[o], sc[o + 1], sc[o + 2]);
}
static int sph_mat(const __global float *sc, int i) { return (int)sc[SC_BASE + OFF_SPH_MAT + i]; }

typedef struct { uint s; } rng_t;

static float rnd(rng_t *r) {
  r->s = r->s * 1664525u + 1013904223u;
  return (float)(r->s >> 8) * (1.0f / 16777216.0f);
}

static int intersect(vec3 o, vec3 d, const __global float *sc, float *tOut) {
  float best = BIG;
  int id = -1;

  for (int i = 0; i < NPLANE; ++i) {
    vec3 nrm = pl_normal(sc, i);
    float denom = dot3(nrm, d);
    if (denom > -1e-8f && denom < 1e-8f) continue;
    float t = (pl_d(sc, i) - dot3(nrm, o)) / denom;
    if (t > EPS && t < best) { best = t; id = i; }
  }

  for (int i = 0; i < NSPHERE; ++i) {
    vec3 p = sph_center(sc, i);
    float rad = sph_rad(sc, i);
    vec3 op = sub3(p, o);
    float b = dot3(op, d);
    float c = dot3(op, op) - rad * rad;
    float det = b * b - c;
    if (det < 0.0f) continue;
    float sq = sqrt(det);
    float t0 = b - sq;
    if (t0 > EPS && t0 < best) { best = t0; id = NPLANE + i; continue; }
    float t1 = b + sq;
    if (t1 > EPS && t1 < best) { best = t1; id = NPLANE + i; }
  }

  *tOut = best;
  return id;
}

static vec3 hit_normal(const __global float *sc, int id, vec3 x) {
  if (id < NPLANE) return pl_normal(sc, id);
  return norm3(sub3(x, sph_center(sc, id - NPLANE)));
}
static vec3 hit_color(const __global float *sc, int id) {
  return (id < NPLANE) ? pl_color(sc, id) : sph_color(sc, id - NPLANE);
}
static vec3 hit_emit(const __global float *sc, int id) {
  return (id < NPLANE) ? V3(0.0f, 0.0f, 0.0f) : sph_emit(sc, id - NPLANE);
}
static int hit_mat(const __global float *sc, int id) {
  return (id < NPLANE) ? MAT_DIFF : sph_mat(sc, id - NPLANE);
}

static vec3 trace(vec3 o, vec3 d, const __global float *sc, rng_t *rng) {
  vec3 L = V3(0.0f, 0.0f, 0.0f);
  vec3 thr = V3(1.0f, 1.0f, 1.0f);

  for (int depth = 0; depth < MAX_DEPTH; ++depth) {
    float t;
    int id = intersect(o, d, sc, &t);
    if (id < 0) break;

    vec3 x = add3(o, scl3(d, t));
    vec3 n = hit_normal(sc, id, x);
    vec3 nl = dot3(n, d) < 0.0f ? n : neg3(n);
    vec3 f = hit_color(sc, id);
    vec3 e = hit_emit(sc, id);

    L = add3(L, mul3(thr, e));
    if (thr.x + thr.y + thr.z <= 0.0f) break;

    float pmax = f.x;
    if (f.y > pmax) pmax = f.y;
    if (f.z > pmax) pmax = f.z;
    if (depth > 5) {
      if (rnd(rng) < pmax) {
        f = scl3(f, 1.0f / pmax);
      } else {
        break;
      }
    }

    int mat = hit_mat(sc, id);
    if (mat == MAT_DIFF) {
      float r1 = 2.0f * PI * rnd(rng);
      float r2 = rnd(rng);
      float r2s = sqrt(r2);
      vec3 w = nl;
      vec3 u = norm3(cross3((w.x > 0.1f || w.x < -0.1f) ? V3(0.0f, 1.0f, 0.0f) : V3(1.0f, 0.0f, 0.0f), w));
      vec3 v = cross3(w, u);
      vec3 nd = norm3(add3(add3(scl3(u, cos(r1) * r2s), scl3(v, sin(r1) * r2s)), scl3(w, sqrt(1.0f - r2))));
      thr = mul3(thr, f);
      o = x;
      d = nd;
      continue;
    } else if (mat == MAT_SPEC) {
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
      if (rnd(rng) < P) {
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

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void smallpt(__global float *img, __global float *albOut, __global float *nrmOut,
                      __global const uint *params) {
  const __global float *sc = (const __global float *)params; // scene data comes from params

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

  rng_t rng;
  rng.s = (uint)(py * w + px) * 1973u + frame * 9277u + 1u;

  const float pyUp = (float)(h - 1u - py);
  vec3 rad = V3(0.0f, 0.0f, 0.0f);

  // ---- auxiliary images for the denoiser ----------------------------------
  // Same first-hit G-buffer as src/smallpt.cl: albedo and shading normal of the surface the primary
  // ray hits. Binding 1 and 2 carry these images, binding 3 the parameters (see smallpt_main.cpp).
  vec3 alb = V3(0.0f, 0.0f, 0.0f);
  vec3 nrm = V3(0.0f, 0.0f, 1.0f);
  {
    vec3 gDir = norm3(add3(add3(camD, scl3(cx, ((float)px + 0.5f) / (float)w - 0.5f)),
                            scl3(cy, ((float)pyUp + 0.5f) / (float)h - 0.5f)));
    vec3 gStart = add3(camO, scl3(gDir, CAM_PUSH));
    float t;
    int id = intersect(gStart, gDir, sc, &t);
    if (id >= 0) {
      vec3 x = add3(gStart, scl3(gDir, t));
      alb = hit_color(sc, id);
      nrm = hit_normal(sc, id, x);
    }
  }

  for (uint sy = 0; sy < 2; ++sy) {
    for (uint sx = 0; sx < 2; ++sx) {
      vec3 sub = V3(0.0f, 0.0f, 0.0f);
      for (uint s = 0; s < spp; ++s) {
        float r1 = 2.0f * rnd(&rng);
        float dx = (r1 < 1.0f) ? (sqrt(r1) - 1.0f) : (1.0f - sqrt(2.0f - r1));
        float r2 = 2.0f * rnd(&rng);
        float dy = (r2 < 1.0f) ? (sqrt(r2) - 1.0f) : (1.0f - sqrt(2.0f - r2));

        vec3 dRaw = add3(add3(scl3(cx, ((((float)sx + 0.5f + dx) * 0.5f) + (float)px) / (float)w - 0.5f),
                              scl3(cy, ((((float)sy + 0.5f + dy) * 0.5f) + pyUp) / (float)h - 0.5f)),
                         camD);
        vec3 d = norm3(dRaw);
        vec3 start = add3(camO, scl3(d, CAM_PUSH));
        sub = add3(sub, trace(start, d, sc, &rng));
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
