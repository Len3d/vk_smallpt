// smallpt_view.cl -- the smallpt Cornell-box scene as an interactive viewer kernel.
//
// Same scene and same trace() as src/smallpt.cl, but driven by a free camera and accumulating one
// sample per pixel per dispatch:
//   * the camera (position, target, vertical FOV) arrives in the parameter buffer, so the host can
//     orbit, pan and zoom without recompiling or rebuilding a module;
//   * every dispatch adds exactly one path per pixel, jittered inside the pixel, and the kernel
//     keeps the running mean in the first binding (RGBA, alpha unused) -- the host only has to pass
//     the sample index as `frame` and restart it at 0 after a camera change;
//   * the ray starts where it enters the room (x in [1,99], y in [0,81.6], z in [0,170]) instead of
//     being pushed a fixed 140 units along its direction like smallpt does: the viewer camera can
//     orbit anywhere and the black front wall must not swallow rays that enter through it;
//   * the albedo / normal of the first hit are written on every dispatch (not accumulated), so the
//     denoiser can be run on the current view.
//
// Parameter buffer (float32): 0 w, 1 h, 2 frame (samples already accumulated), 4..6 camera position,
// 8..10 camera target, 12 tan(fovY/2), 13 exposure (read by the resolve kernel only).
//
// clspv binding rule: the n-th __global argument is descriptor set 0 / binding n.
#define PI 3.14159265358979f
#define EPS 1e-4f
#define BIG 1e20f
#define STRIDE 16       // each sphere occupies 16 floats in the SPHERES array
#define NSPHERE 3       // mirror sphere, glass sphere, light sphere
#define NPLANE 6        // the six walls
#define MAX_DEPTH 10
#define CAM_PUSH 140.0f // distance smallpt pushes the camera ray along its direction (the camera
                        // sits outside the box, so the ray origin must start inside)

// Material ids
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
#define CLAMP_MAX 20.0f
static float clamp1(float x) { return x < 0.0f ? 0.0f : (x > CLAMP_MAX ? CLAMP_MAX : x); }

// Six walls: plane n.p = d with the normal pointing into the box. First 4 floats are geometry,
// the next 3 are the albedo. The numbers follow from the six radius-1e5 spheres of the original
// (inside the box those sphere surfaces are exactly these coordinate planes):
//   Left  x=1     Rght x=99    Back z=0     Frnt z=170   Botm y=0   Top y=81.6
__constant float PLANES[NPLANE * 7] = {
    // nx, ny, nz,  d,      cr,   cg,   cb
     1.0f,  0.0f,  0.0f,   1.0f,  0.75f, 0.25f, 0.25f, // left wall  (red)
    -1.0f,  0.0f,  0.0f, -99.0f,  0.25f, 0.25f, 0.75f, // right wall (blue)
     0.0f,  0.0f,  1.0f,   0.0f,  0.75f, 0.75f, 0.75f, // back wall  (grey)
     0.0f,  0.0f, -1.0f, -170.0f, 0.0f,  0.0f,  0.0f,  // front wall (black, blocks light)
     0.0f,  1.0f,  0.0f,   0.0f,  0.75f, 0.75f, 0.75f, // floor      (grey)
     0.0f, -1.0f,  0.0f, -81.6f,  0.75f, 0.75f, 0.75f, // ceiling    (grey)
};

// Three spheres: [px,py,pz, radius, cr,cg,cb, er,eg,eb, material, ...]
__constant float SPHERES[NSPHERE * STRIDE] = {
    // 0 mirror sphere
    27.0f, 16.5f, 47.0f, 16.5f, 0.999f, 0.999f, 0.999f, 0.0f, 0.0f, 0.0f, MAT_SPEC, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // 1 glass sphere
    73.0f, 16.5f, 78.0f, 16.5f, 0.999f, 0.999f, 0.999f, 0.0f, 0.0f, 0.0f, MAT_GLASS, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // 2 area light (a big radius-600 sphere overhead, emission 12)
    50.0f, 681.6f - 0.27f, 81.6f, 600.0f, 0.0f, 0.0f, 0.0f, 12.0f, 12.0f, 12.0f, MAT_DIFF, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};

// Hit id encoding: 0..NPLANE-1 = walls, NPLANE+i = sphere i
static vec3 sph_center(int i) { int o = i * STRIDE; return V3(SPHERES[o], SPHERES[o + 1], SPHERES[o + 2]); }
static float sph_rad(int i) { return SPHERES[i * STRIDE + 3]; }
static vec3 sph_color(int i) { int o = i * STRIDE + 4; return V3(SPHERES[o], SPHERES[o + 1], SPHERES[o + 2]); }
static vec3 sph_emit(int i) { int o = i * STRIDE + 7; return V3(SPHERES[o], SPHERES[o + 1], SPHERES[o + 2]); }
static int sph_mat(int i) { return (int)SPHERES[i * STRIDE + 10]; }

static vec3 pl_normal(int i) { int o = i * 7; return V3(PLANES[o], PLANES[o + 1], PLANES[o + 2]); }
static float pl_d(int i) { return PLANES[i * 7 + 3]; }
static vec3 pl_color(int i) { int o = i * 7 + 4; return V3(PLANES[o], PLANES[o + 1], PLANES[o + 2]); }

typedef struct { uint s; } rng_t;

// PCG-XSH-RR 32 bit. A plain LCG correlates neighbouring seeds, which shows up
// as diagonal hatching in smooth gradients once hundreds of frames are averaged,
// so the state is advanced and output-mixed separately here.
static uint mix32(uint x) {
  x ^= x >> 16u;
  x *= 0x7FEB352Du;
  x ^= x >> 15u;
  x *= 0x846CA68Bu;
  x ^= x >> 16u;
  return x;
}

static float rnd(rng_t *r) {
  r->s = r->s * 747796405u + 2891336453u;
  uint w = ((r->s >> ((r->s >> 28u) + 4u)) ^ r->s) * 277803737u;
  return (float)((w >> 22u) ^ w) * (1.0f / 4294967296.0f);
}

// Seed from pixel index and sample index, avalanched so that adjacent pixels and
// consecutive frames are uncorrelated.
static rng_t rng_seed(uint px, uint py, uint frame) {
  rng_t r;
  r.s = mix32(px * 0x9E3779B1u ^ py * 0x85EBCA77u ^ frame * 0xC2B2AE3Du) | 1u;
  return r;
}

// Nearest hit, returns the id (<0 = miss); t is returned through the pointer
static int intersect(vec3 o, vec3 d, float *tOut) {
  float best = BIG;
  int id = -1;

  // Planes: solve n.(o+t d) = dist directly.
  // (#pragma unroll asks for a full unroll so the indices become compile-time constants; only then
  //  can clspv fold the __constant array into immediates. Without it the array is demoted to
  //  Private per-thread scratch and indexed dynamically, which is several times slower.)
#pragma clang loop unroll(full)
  for (int i = 0; i < NPLANE; ++i) {
    float denom = dot3(pl_normal(i), d);
    if (denom > -1e-8f && denom < 1e-8f) continue; // ray parallel to the wall
    float t = (pl_d(i) - dot3(pl_normal(i), o)) / denom;
    if (t > EPS && t < best) { best = t; id = i; }
  }

  // Spheres
#pragma clang loop unroll(full)
  for (int i = 0; i < NSPHERE; ++i) {
    vec3 p = sph_center(i);
    float rad = sph_rad(i);
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

static vec3 hit_normal(int id, vec3 x) {
  if (id < NPLANE) return pl_normal(id); // wall normals already point inward, exactly what the
                                        // small offset to leave the surface needs
  return norm3(sub3(x, sph_center(id - NPLANE)));
}
static vec3 hit_color(int id) { return (id < NPLANE) ? pl_color(id) : sph_color(id - NPLANE); }
static vec3 hit_emit(int id) { return (id < NPLANE) ? V3(0.0f, 0.0f, 0.0f) : sph_emit(id - NPLANE); }
static int hit_mat(int id) { return (id < NPLANE) ? MAT_DIFF : sph_mat(id - NPLANE); }

// Iterative radiance, equivalent to the recursive implementation of smallpt
static vec3 trace(vec3 o, vec3 d, rng_t *rng) {
  vec3 L = V3(0.0f, 0.0f, 0.0f);
  vec3 thr = V3(1.0f, 1.0f, 1.0f);

  for (int depth = 0; depth < MAX_DEPTH; ++depth) {
    float t;
    int id = intersect(o, d, &t);
    if (id < 0) break; // miss = black (same as smallpt, there is no sky box)

    vec3 x = add3(o, scl3(d, t));
    vec3 n = hit_normal(id, x);
    vec3 nl = dot3(n, d) < 0.0f ? n : neg3(n); // normal facing the ray
    vec3 f = hit_color(id);
    vec3 e = hit_emit(id);

    // Emissive sphere: in smallpt every material branch returns obj.e plus the recursive result,
    // which is the thr*e accumulation here
    L = add3(L, mul3(thr, e));

    if (thr.x + thr.y + thr.z <= 0.0f) break; // throughput exhausted (e.g. hitting the black wall)

    // Russian roulette: past depth 5 the maximum reflectance decides whether to continue
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

    int mat = hit_mat(id);
    if (mat == MAT_DIFF) {
      // Cosine-weighted hemisphere sampling (the u/v/w basis of smallpt)
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
      // Perfect mirror reflection
      d = sub3(d, scl3(n, 2.0f * dot3(n, d)));
      thr = mul3(thr, f);
      o = x;
      continue;
    } else {
      // Ideal dielectric: Fresnel probability decides reflection vs refraction
      int into = dot3(n, nl) > 0.0f; // entering from outside?
      float nc = 1.0f, nt = 1.5f;
      float nnt = into ? (nc / nt) : (nt / nc);
      float ddn = dot3(d, nl);
      float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);
      vec3 refl = sub3(d, scl3(n, 2.0f * dot3(n, d)));
      if (cos2t < 0.0f) { // total internal reflection
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
      float Re = R0 + (1.0f - R0) * c5; // reflection probability (Schlick)
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
// ---------------------------------------------------------------------------------------------
// Interactive entry point
// ---------------------------------------------------------------------------------------------

// Slab test against the room: x in [1,99], y in [0,81.6], z in [0,170]. Written out per axis on
// purpose -- a loop with PLANES[i] would turn the bounds into a dynamically indexed Private array,
// which is exactly the pattern this repository avoids (see the engineering notes).
static int enter_room(vec3 o, vec3 d, float *tEnter) {
  float t0 = 0.0f, t1 = BIG;

  if (d.x > -1e-8f && d.x < 1e-8f) {
    if (o.x < 1.0f || o.x > 99.0f) return 0;
  } else {
    float ta = (1.0f - o.x) / d.x, tb = (99.0f - o.x) / d.x;
    if (ta > tb) { float tmp = ta; ta = tb; tb = tmp; }
    if (ta > t0) t0 = ta;
    if (tb < t1) t1 = tb;
    if (t0 > t1) return 0;
  }

  if (d.y > -1e-8f && d.y < 1e-8f) {
    if (o.y < 0.0f || o.y > 81.6f) return 0;
  } else {
    float ta = (0.0f - o.y) / d.y, tb = (81.6f - o.y) / d.y;
    if (ta > tb) { float tmp = ta; ta = tb; tb = tmp; }
    if (ta > t0) t0 = ta;
    if (tb < t1) t1 = tb;
    if (t0 > t1) return 0;
  }

  if (d.z > -1e-8f && d.z < 1e-8f) {
    if (o.z < 0.0f || o.z > 170.0f) return 0;
  } else {
    float ta = (0.0f - o.z) / d.z, tb = (170.0f - o.z) / d.z;
    if (ta > tb) { float tmp = ta; ta = tb; tb = tmp; }
    if (ta > t0) t0 = ta;
    if (tb < t1) t1 = tb;
    if (t0 > t1) return 0;
  }

  *tEnter = t0;
  return 1;
}

static float clamp_hi(float x) { return x < 0.0f ? 0.0f : (x > 64.0f ? 64.0f : x); }

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void smallpt_view(__global float4 *acc, __global float *albOut, __global float *nrmOut,
                           __global const float *p) {
  const uint w = (uint)p[0];
  const uint h = (uint)p[1];
  const uint frame = (uint)p[2];

  const uint px = get_global_id(0);
  const uint py = get_global_id(1);
  if (px >= w || py >= h) return; // the last column/row may contain idle threads

  const vec3 camO = V3(p[4], p[5], p[6]);
  const vec3 camT = V3(p[8], p[9], p[10]);
  const float tanHalf = p[12];
  const float aspect = (float)w / (float)h;

  // Camera basis: forward to the target, right = forward x world up, up = right x forward
  const vec3 fwd = norm3(sub3(camT, camO));
  const vec3 rt = norm3(cross3(fwd, V3(0.0f, 1.0f, 0.0f)));
  const vec3 upv = cross3(rt, fwd);

  // One stratified sample: the pixel centre plus a random offset inside the pixel
  rng_t rng = rng_seed(px, py, frame);
  const float jx = rnd(&rng);
  const float jy = rnd(&rng);
  const float ndx = (((float)px + jx) / (float)w) * 2.0f - 1.0f;
  const float ndy = 1.0f - (((float)py + jy) / (float)h) * 2.0f;
  const vec3 d = norm3(add3(add3(fwd, scl3(rt, ndx * aspect * tanHalf)), scl3(upv, ndy * tanHalf)));

  // Start the path where it enters the room; a miss keeps the camera as origin (black background)
  vec3 start = camO;
  float tEnter;
  if (enter_room(camO, d, &tEnter)) start = add3(camO, scl3(d, tEnter + 0.01f));

  vec3 rgb = trace(start, d, &rng);
  rgb = V3(clamp_hi(rgb.x), clamp_hi(rgb.y), clamp_hi(rgb.z));

  // Running mean: sample `frame` is the (frame+1)-th one, so the host restarts at 0 after a change
  const uint idx = py * w + px;
  vec3 mean;
  if (frame == 0u) {
    mean = rgb;
  } else {
    const vec3 old = V3(acc[idx].x, acc[idx].y, acc[idx].z);
    mean = scl3(add3(scl3(old, (float)frame), rgb), 1.0f / (float)(frame + 1u));
  }
  acc[idx] = (float4)(mean.x, mean.y, mean.z, 1.0f);

  // First-hit G-buffer for the denoiser (overwritten every dispatch, never accumulated)
  vec3 alb = V3(0.0f, 0.0f, 0.0f);
  vec3 nrm = V3(0.0f, 0.0f, 1.0f);
  {
    float t;
    int id = intersect(start, d, &t);
    if (id >= 0) {
      vec3 x = add3(start, scl3(d, t));
      alb = hit_color(id);
      nrm = hit_normal(id, x);
    }
  }

  const uint auv = idx * 3;
  albOut[auv + 0] = alb.x;
  albOut[auv + 1] = alb.y;
  albOut[auv + 2] = alb.z;
  nrmOut[auv + 0] = nrm.x;
  nrmOut[auv + 1] = nrm.y;
  nrmOut[auv + 2] = nrm.z;
}
