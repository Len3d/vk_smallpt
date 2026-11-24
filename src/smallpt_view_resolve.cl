// smallpt_view_resolve.cl -- display pass for the interactive viewer.
//
// Reads the accumulated mean (RGBA float32, one float4 per pixel) and writes the 8-bit image the
// GUI samples: exposure, Reinhard tone mapping, gamma. Packing the four bytes into one uint32 keeps
// the kernel free of 8-bit storage types (clspv would need storageBuffer8BitAccess for those), and
// a plain vkCmdCopyBufferToImage turns the buffer into the sampled image.
//
// Parameter buffer: 0 w, 1 h, 13 exposure. Binding order = argument order (clspv rule).

static float clamp1(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

static float encode(float x) { return pow(clamp1(x), 1.0f / 2.2f); }

static uint pack_rgba(float r, float g, float b) {
  const uint ri = (uint)(encode(r) * 255.0f + 0.5f) & 0xFFu;
  const uint gi = (uint)(encode(g) * 255.0f + 0.5f) & 0xFFu;
  const uint bi = (uint)(encode(b) * 255.0f + 0.5f) & 0xFFu;
  return ri | (gi << 8) | (bi << 16) | (0xFFu << 24);
}

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void smallpt_view_resolve(__global const float4 *acc, __global const float *p,
                                   __global uint *out) {
  const uint w = (uint)p[0];
  const uint h = (uint)p[1];
  const float exposure = p[13];

  const uint px = get_global_id(0);
  const uint py = get_global_id(1);
  if (px >= w || py >= h) return;

  const uint idx = py * w + px;
  float4 c = acc[idx] * exposure;

  // Reinhard: keeps the bright light from clipping to a flat disc at low sample counts
  c = c / (1.0f + c);

  out[idx] = pack_rgba(c.x, c.y, c.z);
}
