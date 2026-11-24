// smallpt_cpp.cl -- the kernel smallpt.exe renders with, written in C++ for OpenCL
// (clspv --cl-std=CLC++2021) with the vector math expressed as operator overloading instead of
// the add3/sub3/scl3/dot3/cross3/norm3 helper functions:
//
//   vec3 a = add3(b, scl3(c, 2.0f))   ->   Vec3 a = b + c * 2.0f
//   vec3 n = norm3(cross3(u, v))      ->   Vec3 n = normalize(cross(u, v))
//   float d = dot3(n, r.direction)    ->   float d = dot(n, r.direction)
//
// It carries every optimization developed for this kernel over the earlier rounds; none of them
// depends on the language mode:
//
//   1. Literal subscripts (the `smallpt_lit.cl` variant). clspv has no OpenCL __constant address
//      space, so a module-scope __constant array is demoted to Private (a per-thread copy) and a
//      run-time subscript into it becomes an OpAccessChain into that copy. What decides this is the
//      *subscript*, not the address space: with a compile-time subscript clspv folds the value into
//      an immediate and the array keeps no OpVariable at all. `#pragma clang loop unroll(full)` does
//      not achieve it (the loop survives), so the six walls and three spheres are written out by the
//      SCENE_HIT_TEST macro with literal indices. Effect: the scene tables disappear from the module
//      and there is no Private storage for them.
//   2. Hit information selected at the moment of the hit: the macro block stores the plane normal /
//      sphere centre, albedo, emission and material straight into locals, so trace() never looks
//      anything up by a run-time hit id. Keeping this in plain scalars (rather than in a hit struct
//      passed by reference) is what keeps the values in SSA: a struct forces clspv to materialize
//      every field in Function memory and reload them.
//   3. Native sin/cos and relaxed math (the `smallpt_buf.cl` + clspv-flags variant): the default
//      build uses --use-native-builtins=sin,cos --cl-fast-relaxed-math, so cos/sin become the
//      hardware ops instead of the inlined libclc polynomial (196 instructions per call site) and
//      division loses the denormal/range scaling guards. That trades accuracy for speed.
//   4. The current feature set: the CLAMP_MAX per-subsample tone clamp and the albedo / normal
//      first-hit G-buffer written to bindings 1 and 2.
//
// Arithmetic order is preserved for every formula, so building this file with plain -O3 renders an
// image bit-identical to src/smallpt.cl. The fast-math/native-builtin flags (the default) change
// only the last bits.
//
// clspv specifics for the C++ mode (verified on this machine, see the README):
//   - --cl-std=CLC++2021 requires -inline-entry-points (C++ for OpenCL uses the generic address
//     space by default and the entry point must be inlined before the address spaces are resolved);
//   - reqd_work_group_size is NOT honoured in this mode: the module declares the workgroup size as
//     spec constants 0/1/2 instead, so the build freezes them with
//     `spirv-opt --set-spec-const-default-value "0:8 1:8 2:1" --freeze-spec-const` and the module
//     ends up with a constant WorkgroupSize builtin of 8/8/1, the same shape glslc produces;
//   - CLC++ semantics make get_global_id() read the OpenCL "global offset" from a 12-byte push
//     constant block, which src/smallpt_main.cpp declares and fills with zeros before each
//     dispatch;
//   - a ?: that mixes a plain object with a temporary mixes generic and private address spaces and
//     is rejected, so the normal flip and the u/v/w basis pick are written as if statements.

#define PI 3.14159265358979f
// Per-subsample tone clamp, same value as smallpt.cl: without it a single bright light sample
// blows up a whole pixel and the denoiser sees fireflies it cannot remove.
#define CLAMP_MAX 20.0f
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

// ---------------------------------------------------------------------------------------------
// The vector type. Constructors plus overloaded arithmetic replace the old free functions.
// ---------------------------------------------------------------------------------------------
struct Vec3 {
  float x, y, z;

  Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
  Vec3(float a, float b, float c) : x(a), y(b), z(c) {}

  Vec3 operator+(const Vec3 &o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
  Vec3 operator-(const Vec3 &o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
  Vec3 operator-() const { return Vec3(-x, -y, -z); }

  // Component-wise product (the old mul3) and scalar product (the old scl3)
  Vec3 operator*(const Vec3 &o) const { return Vec3(x * o.x, y * o.y, z * o.z); }
  Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
  friend Vec3 operator*(float s, const Vec3 &v) { return Vec3(v.x * s, v.y * s, v.z * s); }

  Vec3 &operator+=(const Vec3 &o) { x += o.x; y += o.y; z += o.z; return *this; }
};

// Same association as the old dot3: ((ax*bx + ay*by) + az*bz)
inline float dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Multiply by the reciprocal of the length (not a division), exactly like the old norm3
inline Vec3 normalize(const Vec3 &a) { return a * (1.0f / sqrt(dot(a, a))); }

inline float clamp1(float v) { return v < 0.0f ? 0.0f : (v > CLAMP_MAX ? CLAMP_MAX : v); }

// ---------------------------------------------------------------------------------------------
// Scene tables. Same numbers as smallpt.cl: six walls as planes n.p = d (normal pointing into the
// box, first four floats are geometry and the next three the albedo) and three spheres
// ([px,py,pz,radius,cr,cg,cd,er,eg,eb,material,...]).
//
// The tables stay __constant for readability, but nothing below ever subscripts them with a
// run-time index, so clspv dissolves them into immediates instead of copying them into per-thread
// Private storage.
// ---------------------------------------------------------------------------------------------
__constant float PLANES[NPLANE * 7] = {
    // nx, ny, nz,  d,      cr,   cg,   cb
     1.0f,  0.0f,  0.0f,   1.0f,  0.75f, 0.25f, 0.25f, // left wall  (red)
    -1.0f,  0.0f,  0.0f, -99.0f,  0.25f, 0.25f, 0.75f, // right wall (blue)
     0.0f,  0.0f,  1.0f,   0.0f,  0.75f, 0.75f, 0.75f, // back wall  (grey)
     0.0f,  0.0f, -1.0f, -170.0f, 0.0f,  0.0f,  0.0f,  // front wall (black, blocks light)
     0.0f,  1.0f,  0.0f,   0.0f,  0.75f, 0.75f, 0.75f, // floor      (grey)
     0.0f, -1.0f,  0.0f, -81.6f,  0.75f, 0.75f, 0.75f, // ceiling    (grey)
};

__constant float SPHERES[NSPHERE * STRIDE] = {
    // 0 mirror sphere
    27.0f, 16.5f, 47.0f, 16.5f, 0.999f, 0.999f, 0.999f, 0.0f, 0.0f, 0.0f, MAT_SPEC, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // 1 glass sphere
    73.0f, 16.5f, 78.0f, 16.5f, 0.999f, 0.999f, 0.999f, 0.0f, 0.0f, 0.0f, MAT_GLASS, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    // 2 area light (a big radius-600 sphere overhead, emission 12)
    50.0f, 681.6f - 0.27f, 81.6f, 600.0f, 0.0f, 0.0f, 0.0f, 12.0f, 12.0f, 12.0f, MAT_DIFF, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};

// Linear congruential generator, keeping only the top 24 bits: fast and needs no libclc rand
struct Rng {
  uint s;
  float next() {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
  }
};

// ---------------------------------------------------------------------------------------------
// The unrolled nearest-hit test. Every subscript is a literal (the macro argument I is spliced in
// as a constant), so the scene tables fold into immediates. The winning primitive's data lands in
// the locals below, which is why nothing needs a run-time hit id afterwards.
//
// The macro block expects these locals at the expansion site: o, d (the ray), and
//   float best; int kind; Vec3 hNorm, hCentre, hAlb, hEmit; int hMat;
// ---------------------------------------------------------------------------------------------
#define SCENE_TRY_PLANE(I)                                                                        \
  do {                                                                                            \
    const Vec3 n_ = Vec3(PLANES[(I) * 7 + 0], PLANES[(I) * 7 + 1], PLANES[(I) * 7 + 2]);           \
    const float dd_ = dot(n_, d);                                                                  \
    if (!(dd_ > -1e-8f && dd_ < 1e-8f)) {                                                          \
      const float t_ = (PLANES[(I) * 7 + 3] - dot(n_, o)) / dd_;                                   \
      if (t_ > EPS && t_ < best) {                                                                 \
        best = t_;                                                                                 \
        kind = 0;                                                                                  \
        hNorm = n_;                                                                                \
        hMat = MAT_DIFF;                                                                           \
        hEmit = Vec3(0.0f, 0.0f, 0.0f);                                                            \
        hAlb = Vec3(PLANES[(I) * 7 + 4], PLANES[(I) * 7 + 5], PLANES[(I) * 7 + 6]);                \
      }                                                                                            \
    }                                                                                              \
  } while (0)

#define SCENE_TRY_SPHERE(I)                                                                        \
  do {                                                                                             \
    const Vec3 p_ = Vec3(SPHERES[(I) * STRIDE + 0], SPHERES[(I) * STRIDE + 1],                      \
                         SPHERES[(I) * STRIDE + 2]);                                                \
    const float rad_ = SPHERES[(I) * STRIDE + 3];                                                   \
    const Vec3 op_ = p_ - o;                                                                        \
    const float b_ = dot(op_, d);                                                                   \
    const float c_ = dot(op_, op_) - rad_ * rad_;                                                    \
    const float det_ = b_ * b_ - c_;                                                                 \
    if (det_ >= 0.0f) {                                                                              \
      const float sq_ = sqrt(det_);                                                                   \
      const float t0_ = b_ - sq_;                                                                     \
      float t_ = -1.0f;                                                                               \
      if (t0_ > EPS && t0_ < best) {                                                                   \
        t_ = t0_;                                                                                      \
      } else {                                                                                         \
        const float t1_ = b_ + sq_;                                                                    \
        if (t1_ > EPS && t1_ < best) t_ = t1_;                                                          \
      }                                                                                                 \
      if (t_ > 0.0f) {                                                                                  \
        best = t_;                                                                                       \
        kind = 1;                                                                                         \
        hCentre = p_;                                                                                     \
        hMat = static_cast<int>(SPHERES[(I) * STRIDE + 10]);                                               \
        hAlb = Vec3(SPHERES[(I) * STRIDE + 4], SPHERES[(I) * STRIDE + 5], SPHERES[(I) * STRIDE + 6]);        \
        hEmit = Vec3(SPHERES[(I) * STRIDE + 7], SPHERES[(I) * STRIDE + 8], SPHERES[(I) * STRIDE + 9]);       \
      }                                                                                                       \
    }                                                                                                         \
  } while (0)

// Six walls, then the three spheres (same test order as smallpt.cl)
#define SCENE_HIT_TEST      \
  SCENE_TRY_PLANE(0);       \
  SCENE_TRY_PLANE(1);       \
  SCENE_TRY_PLANE(2);       \
  SCENE_TRY_PLANE(3);       \
  SCENE_TRY_PLANE(4);       \
  SCENE_TRY_PLANE(5);       \
  SCENE_TRY_SPHERE(0);      \
  SCENE_TRY_SPHERE(1);      \
  SCENE_TRY_SPHERE(2)

// First hit along a ray, used once per pixel for the denoiser's G-buffer. alb / nrm are the outputs;
// a miss keeps albedo black and the normal pointing at the camera.
inline void first_hit(const Vec3 &o, const Vec3 &d, Vec3 &alb, Vec3 &nrm) {
  float best = BIG;
  int kind = -1;
  Vec3 hNorm(0.0f, 0.0f, 0.0f), hCentre(0.0f, 0.0f, 0.0f);
  Vec3 hAlb(0.0f, 0.0f, 0.0f), hEmit(0.0f, 0.0f, 0.0f);
  int hMat = MAT_DIFF;
  SCENE_HIT_TEST;
  // the G-buffer needs neither the material nor the emission of the first hit
  static_cast<void>(hMat);
  static_cast<void>(hEmit);
  if (kind < 0) return;
  const Vec3 x = o + d * best;
  alb = hAlb;
  if (kind == 1)
    nrm = normalize(x - hCentre);
  else
    nrm = hNorm;
}

// Iterative radiance, equivalent to the recursive implementation of smallpt
inline Vec3 trace(const Vec3 &o0, const Vec3 &d0, Rng &rng) {
  Vec3 o = o0;
  Vec3 d = d0;
  Vec3 L(0.0f, 0.0f, 0.0f);
  Vec3 thr(1.0f, 1.0f, 1.0f);

  for (int depth = 0; depth < MAX_DEPTH; ++depth) {
    float best = BIG;
    int kind = -1;
    Vec3 hNorm(0.0f, 0.0f, 0.0f), hCentre(0.0f, 0.0f, 0.0f);
    Vec3 hAlb(0.0f, 0.0f, 0.0f), hEmit(0.0f, 0.0f, 0.0f);
    int hMat = MAT_DIFF;
    SCENE_HIT_TEST;
    if (kind < 0) break; // miss = black (same as smallpt, there is no sky box)

    const Vec3 x = o + d * best;
    Vec3 n = hNorm;                    // walls: the stored normal already points inward
    if (kind == 1) n = normalize(x - hCentre);
    Vec3 nl = n;                       // normal facing the ray
    if (dot(n, d) >= 0.0f) nl = -n;    // written as a statement: a ?: between a plain object and a
                                       // temporary mixes generic and private address spaces and is
                                       // rejected by C++ for OpenCL
    Vec3 f = hAlb;

    // Emissive sphere: in smallpt every material branch returns obj.e plus the recursive result,
    // which is the thr*e accumulation here
    L += thr * hEmit;

    if (thr.x + thr.y + thr.z <= 0.0f) break; // throughput exhausted (e.g. hitting the black wall)

    // Russian roulette: past depth 5 the maximum reflectance decides whether to continue
    float pmax = f.x;
    if (f.y > pmax) pmax = f.y;
    if (f.z > pmax) pmax = f.z;
    if (depth > 5) {
      if (rng.next() < pmax) {
        f = f * (1.0f / pmax);
      } else {
        break;
      }
    }

    if (hMat == MAT_DIFF) {
      // Cosine-weighted hemisphere sampling (the u/v/w basis of smallpt)
      const float r1 = 2.0f * PI * rng.next();
      const float r2 = rng.next();
      const float r2s = sqrt(r2);
      const Vec3 w = nl;
      Vec3 ax(1.0f, 0.0f, 0.0f);
      if (w.x > 0.1f || w.x < -0.1f) ax = Vec3(0.0f, 1.0f, 0.0f);
      const Vec3 u = normalize(cross(ax, w));
      const Vec3 v = cross(w, u);
      const Vec3 nd = normalize(u * (cos(r1) * r2s) + v * (sin(r1) * r2s) + w * sqrt(1.0f - r2));
      thr = thr * f;
      o = x;
      d = nd;
      continue;
    } else if (hMat == MAT_SPEC) {
      // Perfect mirror reflection
      d = d - n * (2.0f * dot(n, d));
      thr = thr * f;
      o = x;
      continue;
    } else {
      // Glass: Fresnel-weighted choice between reflection and refraction (smallpt picks one by
      // probability instead of evaluating both)
      const bool into = dot(n, nl) > 0.0f;
      const float nc = 1.0f, nt = 1.5f;
      const float nnt = into ? (nc / nt) : (nt / nc);
      const float ddn = dot(d, nl);
      const float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);
      const Vec3 refl = d - n * (2.0f * dot(n, d));
      if (cos2t < 0.0f) { // total internal reflection
        d = refl;
        thr = thr * f;
        o = x;
        continue;
      }
      const float sign = into ? 1.0f : -1.0f;
      const float k = sign * (ddn * nnt + sqrt(cos2t));
      const Vec3 tdir = normalize(d * nnt - n * k);
      const float a = nt - nc, b = nt + nc;
      const float R0 = (a * a) / (b * b);
      const float cc = 1.0f - (into ? (-ddn) : dot(tdir, n));
      const float c2 = cc * cc, c4 = c2 * c2, c5 = c4 * cc;
      const float Re = R0 + (1.0f - R0) * c5; // reflection probability (Schlick)
      const float Tr = 1.0f - Re;
      const float P = 0.25f + 0.5f * Re;
      const float RP = Re / P;
      const float TP = Tr / (1.0f - P);
      if (rng.next() < P) {
        d = refl;
        thr = thr * (f * RP);
      } else {
        d = tdir;
        thr = thr * (f * TP);
      }
      o = x;
      continue;
    }
  }
  return L;
}

// img: 4 floats per pixel (rgb + unused); albOut / nrmOut: 3 floats per pixel (the auxiliary
// images of the OIDN denoiser); params: {width, height, spp, frame}
__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void smallpt(__global float *img, __global float *albOut, __global float *nrmOut,
                      __global const uint *params) {
  const uint w = params[0];
  const uint h = params[1];
  const uint spp = params[2];
  const uint frame = params[3];

  const uint px = get_global_id(0);
  const uint py = get_global_id(1);
  if (px >= w || py >= h) return; // the last column/row may contain idle threads

  // smallpt camera: position + view direction + image plane basis derived from the aspect ratio
  const float aspect = static_cast<float>(w) / static_cast<float>(h);
  const Vec3 camO(50.0f, 52.0f, 295.6f);
  const Vec3 camD = normalize(Vec3(0.0f, -0.042612f, -1.0f));
  const Vec3 cx(aspect * 0.5135f, 0.0f, 0.0f);
  const Vec3 cy = normalize(cross(cx, camD)) * 0.5135f;

  Rng rng;
  rng.s = static_cast<uint>(py * w + px) * 1973u + frame * 9277u + 1u;

  // py is the output row index (PPM row 0 is the top of the image) while the camera +cy axis points
  // up, hence the (h-1-py) conversion
  const float pyUp = static_cast<float>(h - 1u - py);
  Vec3 rad(0.0f, 0.0f, 0.0f);

  // ---- auxiliary images for the denoiser ----------------------------------
  // OIDN wants the albedo and the shading normal of the first surface the primary ray hits (a
  // first-hit G-buffer), so one extra primary ray through the pixel centre fills both. A miss
  // leaves albedo black and the normal pointing at the camera, which is what an empty background
  // should look like. Same arithmetic order as src/smallpt.cl.
  Vec3 alb(0.0f, 0.0f, 0.0f);
  Vec3 nrm(0.0f, 0.0f, 1.0f);
  {
    const Vec3 gDir = normalize(
        (camD + cx * (((static_cast<float>(px) + 0.5f) / static_cast<float>(w)) - 0.5f)) +
        cy * (((pyUp + 0.5f) / static_cast<float>(h)) - 0.5f));
    const Vec3 gStart = camO + gDir * CAM_PUSH;
    first_hit(gStart, gDir, alb, nrm);
  }

  // 2x2 subsamples per pixel, tent-distributed jitter inside each subsample plus spp paths
  for (uint sy = 0; sy < 2; ++sy) {
    for (uint sx = 0; sx < 2; ++sx) {
      Vec3 sub(0.0f, 0.0f, 0.0f);
      for (uint s = 0; s < spp; ++s) {
        // Tent-distributed jitter: dx in [-1,1], the 2*erand48 + sqrt trick of smallpt
        const float r1 = 2.0f * rng.next();
        const float dx = (r1 < 1.0f) ? (sqrt(r1) - 1.0f) : (1.0f - sqrt(2.0f - r1));
        const float r2 = 2.0f * rng.next();
        const float dy = (r2 < 1.0f) ? (sqrt(r2) - 1.0f) : (1.0f - sqrt(2.0f - r2));

        const Vec3 dRaw = cx * ((((static_cast<float>(sx) + 0.5f + dx) * 0.5f) + static_cast<float>(px)) /
                                    static_cast<float>(w) - 0.5f)
                        + cy * ((((static_cast<float>(sy) + 0.5f + dy) * 0.5f) + pyUp) /
                                    static_cast<float>(h) - 0.5f)
                        + camD;
        const Vec3 d = normalize(dRaw);
        // smallpt pushes the camera ray 140 units along its direction: the camera is outside the
        // box, so the origin must be moved inside, otherwise the first ray hits the black front
        // wall (Frnt, albedo = 0) and the whole image is black
        const Vec3 start = camO + d * CAM_PUSH;
        sub += trace(start, d, rng);
      }
      sub = sub * (1.0f / static_cast<float>(spp));
      // smallpt clamps each subsample before averaging, so a bright light cannot blow up the pixel
      rad += Vec3(clamp1(sub.x), clamp1(sub.y), clamp1(sub.z));
    }
  }
  rad = rad * 0.25f;

  const uint idx = (py * w + px) * 4;
  img[idx + 0] = rad.x;
  img[idx + 1] = rad.y;
  img[idx + 2] = rad.z;
  img[idx + 3] = 1.0f;

  // Auxiliary images: the denoiser clamps the albedo to [0, 1] and the normal to [-1, 1] itself,
  // the normal then gets remapped to [0, 1] exactly like OIDN's InputReorder::storeNormal.
  const uint auv = (py * w + px) * 3;
  albOut[auv + 0] = alb.x;
  albOut[auv + 1] = alb.y;
  albOut[auv + 2] = alb.z;
  nrmOut[auv + 0] = nrm.x;
  nrmOut[auv + 1] = nrm.y;
  nrmOut[auv + 2] = nrm.z;
}
