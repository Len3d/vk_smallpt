// smallpt.cl -- the smallpt Cornell-box scene written in OpenCL C and compiled to Vulkan compute
// SPIR-V by clspv.
//
// Differences from the original smallpt (99 lines of C++) are all needed to run correctly on a
// GPU / in OpenCL C / through clspv:
//   1. The original uses recursive radiance(); OpenCL C forbids recursion. This version uses
//      "throughput + iteration" instead: thr tracks the colour attenuation, direct light of a hit
//      is accumulated as thr * emission, and DIFF/SPEC/GLASS only update the ray and continue.
//   2. The original glass sphere evaluates both the reflected and the refracted sub-path (adds
//      them when depth <= 2, picks one by Fresnel probability above that); this version always
//      picks one by Fresnel probability (unbiased, slightly higher variance).
//   3. The original uses double + erand48; this version uses float + one LCG state per thread.
//   4. The original turns the six walls into spheres of radius 1e5 (a sphere that big is almost a
//      plane, which saves code). In float that breaks down: b*b - c subtracts two ~1e10 numbers,
//      so t is off by ~1e-2 and diffuse rays hit the same wall again or hit the wrong place,
//      producing blocky and striped artifacts (raising EPS from 1e-4 to 1e-2 only removes part of
//      them). So the six walls are analytic planes here: same position, colour and orientation as
//      those spheres, only the normal is no longer slightly curved (invisible in the image), and
//      float never self-intersects. The three real spheres (mirror 16.5, glass 16.5, light 600)
//      still use sphere intersection - small radii, so float is accurate enough.
//   5. The scene lives in the kernel source (__constant arrays), so the host passes no geometry:
//      only an output image buffer and a parameter buffer (width/height/spp/frame).
//
// clspv rules (same as the older example):
//   - entry point = kernel function name = smallpt;
//   - n-th __global argument -> descriptor set 0 / binding n;
//   - get_global_id(0/1) -> gl_GlobalInvocationID.xy, dispatch with ceil(width/8), ceil(height/8);
//   - reqd_work_group_size(8,8,1) turns the workgroup size into OpExecutionMode LocalSize 8 8 1
//     instead of the default spec constant 1,1,1 (without it a host dispatching 8x8 would only run
//     one thread per group).

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

// Linear congruential, keeping only the top 24 bits: fast and needs no libclc rand
static float rnd(rng_t *r) {
  r->s = r->s * 1664525u + 1013904223u;
  return (float)(r->s >> 8) * (1.0f / 16777216.0f);
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
  const float aspect = (float)w / (float)h;
  const vec3 camO = V3(50.0f, 52.0f, 295.6f);
  const vec3 camD = norm3(V3(0.0f, -0.042612f, -1.0f));
  const vec3 cx = V3(aspect * 0.5135f, 0.0f, 0.0f);
  const vec3 cy = scl3(norm3(cross3(cx, camD)), 0.5135f);

  rng_t rng;
  rng.s = (uint)(py * w + px) * 1973u + frame * 9277u + 1u;

  // py is the output row index (PPM row 0 is the top of the image) while the camera +cy axis points
  // up, hence the (h-1-py) conversion
  const float pyUp = (float)(h - 1u - py);
  vec3 rad = V3(0.0f, 0.0f, 0.0f);

  // ---- auxiliary images for the denoiser ----------------------------------
  // OIDN wants the albedo and the shading normal of the first surface the primary ray hits
  // (a first-hit G-buffer, see the guide's albedo / normal feature images), so one extra primary
  // ray per pixel through the pixel centre fills both. A miss leaves albedo black and the normal
  // pointing at the camera, which is what an empty background should look like.
  vec3 alb = V3(0.0f, 0.0f, 0.0f);
  vec3 nrm = V3(0.0f, 0.0f, 1.0f);
  {
    vec3 gDir = norm3(add3(add3(camD, scl3(cx, ((float)px + 0.5f) / (float)w - 0.5f)),
                            scl3(cy, ((float)pyUp + 0.5f) / (float)h - 0.5f)));
    vec3 gStart = add3(camO, scl3(gDir, CAM_PUSH));
    float t;
    int id = intersect(gStart, gDir, &t);
    if (id >= 0) {
      vec3 x = add3(gStart, scl3(gDir, t));
      alb = hit_color(id);
      nrm = hit_normal(id, x);
    }
  }

  // 2x2 subsamples per pixel, tent-distributed jitter inside each subsample plus spp paths
  // (exactly like smallpt)
  for (uint sy = 0; sy < 2; ++sy) {
    for (uint sx = 0; sx < 2; ++sx) {
      vec3 sub = V3(0.0f, 0.0f, 0.0f);
      for (uint s = 0; s < spp; ++s) {
        // Tent-distributed jitter: dx in [-1,1], the 2*erand48 + sqrt trick of smallpt
        float r1 = 2.0f * rnd(&rng);
        float dx = (r1 < 1.0f) ? (sqrt(r1) - 1.0f) : (1.0f - sqrt(2.0f - r1));
        float r2 = 2.0f * rnd(&rng);
        float dy = (r2 < 1.0f) ? (sqrt(r2) - 1.0f) : (1.0f - sqrt(2.0f - r2));

        vec3 dRaw = add3(add3(scl3(cx, ((((float)sx + 0.5f + dx) * 0.5f) + (float)px) / (float)w - 0.5f),
                              scl3(cy, ((((float)sy + 0.5f + dy) * 0.5f) + pyUp) / (float)h - 0.5f)),
                         camD);
        vec3 d = norm3(dRaw);
        // smallpt pushes the camera ray 140 units along its direction: the camera is outside the
        // box, so the origin must be moved inside, otherwise the first ray hits the black front
        // wall (Frnt, albedo = 0) and the whole image is black
        vec3 start = add3(camO, scl3(d, CAM_PUSH));
        sub = add3(sub, trace(start, d, &rng));
      }
      sub = scl3(sub, 1.0f / (float)spp);
      // smallpt clamps each subsample before averaging, so a bright light cannot blow up the pixel
      rad = add3(rad, V3(clamp1(sub.x), clamp1(sub.y), clamp1(sub.z)));
    }
  }
  rad = scl3(rad, 0.25f);

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
