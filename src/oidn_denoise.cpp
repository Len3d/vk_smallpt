// oidn_denoise.cpp -- runs the OpenImageDenoise RT U-Net as a chain of Vulkan compute dispatches.
//
// The weights come from OIDN's own rt_hdr.tza blob (converted to raw float32 by
// tools/oidn/export_oidn_weights.py); the topology reproduces core/unet.cpp:
//
//   in -> e0 -> e1 -> p1 -> e2 -> p2 -> e3 -> p3 -> e4 -> p4 -> e5a -> e5b
//      -> up -> cat(p3) -> d4a -> d4b -> up -> cat(p2) -> d3a -> d3b
//      -> up -> cat(p1) -> d2a -> d2b -> up -> cat(in) -> d1a -> d1b -> d0 -> out
//
// Every convolution is 3x3 / padding 1 / stride 1 (core/conv.h) with a fused ReLU except for
// dec_conv0; pooling is 2x2 max with stride 2 (core/pool.h); upsampling is nearest 2x
// (core/upsample.cpp). Tensors are pixel-major (HWC) float32.
//
// Large images are split into tiles exactly like UNetFilter::execute in core/unet.cpp: tile
// buffers are 16-pixel aligned, consecutive tiles overlap by 96 pixels (>= half the 174-pixel
// receptive field), and only the non-overlapping part of each tile is written back. The tile
// size is chosen from a memory budget (OIDN_DENOISE_MAX_MB, default 3000 like OIDN).
//
// Tensors are pixel-major (HWC) and stored as FP16 by default when the device offers 16-bit
// storage buffers (OIDN_STORE=fp32 forces FP32), while the weights, the shared-memory patch and
// every accumulator stay float32 -- see the storage precision comment in oidn_denoise_image and
// the measured table in the README.

#include "oidn_denoise.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#ifndef OIDN_DENOISER_SPV_DIR
#define OIDN_DENOISER_SPV_DIR "."
#endif
#ifndef OIDN_DENOISER_WEIGHTS_PATH
#define OIDN_DENOISER_WEIGHTS_PATH "rt_hdr.bin"
#endif

namespace {

const float PU_A = 1.41283765e+03f;
const float PU_B = 1.64593172e+00f;
const float PU_C = 4.31384981e-01f;
const float PU_D = -2.94139609e-03f;
const float PU_E = 1.92653254e-01f;
const float PU_F = 6.26026094e-03f;
const float PU_G = 9.98620152e-01f;
const float PU_Y0 = 1.57945760e-06f;
const float PU_Y1 = 3.22087631e-02f;
const float HDR_Y_MAX = 65504.0f;

// Same curve as core/color.ispc; used on the host only to derive the normalization scale.
float puForward(float y) {
  if (y <= PU_Y0) return PU_A * y;
  if (y <= PU_Y1) return PU_B * std::pow(y, PU_C) + PU_D;
  return PU_E * std::log(y + PU_F) + PU_G;
}

// core/color.cpp getAutoexposure: average log luminance of a 16x downsampled image.
float compute_autoexposure(const float *rgba, int width, int height) {
  const float key = 0.18f;
  const float eps = 1e-8f;
  const int k = 16;
  const int hk = (height + k / 2) / k;
  const int wk = (width + k / 2) / k;
  double sumLog = 0.0;
  int count = 0;
  for (int i = 0; i < hk; ++i) {
    const int b0 = i * height / hk;
    const int b1 = (i + 1) * height / hk;
    if (b1 <= b0) continue;
    for (int j = 0; j < wk; ++j) {
      const int c0 = j * width / wk;
      const int c1 = (j + 1) * width / wk;
      if (c1 <= c0) continue;
      double lum = 0.0;
      for (int y = b0; y < b1; ++y) {
        for (int x = c0; x < c1; ++x) {
          const float *p = rgba + (static_cast<size_t>(y) * width + x) * 4;
          const double r = (p[0] <= 0.0f || p[0] != p[0]) ? 0.0 : std::min(p[0], HDR_Y_MAX);
          const double g = (p[1] <= 0.0f || p[1] != p[1]) ? 0.0 : std::min(p[1], HDR_Y_MAX);
          const double b = (p[2] <= 0.0f || p[2] != p[2]) ? 0.0 : std::min(p[2], HDR_Y_MAX);
          lum += 0.212671 * r + 0.715160 * g + 0.072169 * b;
        }
      }
      lum /= static_cast<double>((b1 - b0) * (c1 - c0));
      if (lum > eps) {
        sumLog += std::log2(lum);
        ++count;
      }
    }
  }
  if (count == 0) return 1.0f;
  return static_cast<float>(key / std::pow(2.0, sumLog / count));
}

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void *mapped = nullptr;
  size_t bytes = 0;
};

struct Tensor {
  int w = 0, h = 0, c = 0;
  int cPad = 0;  // channel stride of the channel-blocked layout: c rounded up to a multiple of 4
  Buffer buf;
};

// Channel-blocked layout: every tensor's channel stride is padded to a multiple of 4, so a
// 4-channel chunk is one aligned 16-byte (FP32) / 8-byte (FP16) block and every shader can copy a
// whole chunk with a single vec4 load or store. Padding channels are never read as data by the
// convolutions (their loops bound by the real channel count), only by whole-chunk copies.
int pad4i(int c) { return (c + 3) & ~3; }

// A single compute pass. Everything the dispatch needs is described here, so the whole network
// is one static table instead of 30 hand-written dispatch sites.
enum PassKind { PASS_PRE, PASS_CONV, PASS_POOL, PASS_UPSAMPLE, PASS_COPY, PASS_POST };

struct Pass {
  PassKind kind;
  int src;        // tensor index, or -1 for the external rgna buffer
  int dst;        // tensor index, or -1 for the external rgba buffer
  int cin;        // convolution: input channels (also channels for pool/upsample/copy)
  int cout;       // convolution: output channels
  int weightOfs;  // convolution: float offset into the weights buffer
  int relu;
  int dstOfsC;    // concat channel offset
};

// The weight offsets used to be hardcoded here, as printed by export_oidn_weights.py for the
// color-only blob. They are computed at run time now, because two layers depend on the number of
// input channels (see the layer walk in oidn_denoise_image).
const int kWeightRowStride = 12;

// Decode IEEE 754 binary16 to float32. Only used for the OIDN_STORE=fp16 tensor dumps and for
// nothing else on the host, so a small bit-twiddling implementation beats pulling in a header.
float half_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1Fu;
  uint32_t mantissa = h & 0x3FFu;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;  // signed zero
    } else {
      // Subnormal: shift the leading bit up into the normal range.
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3FFu;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1F) {
    bits = sign | 0x7F800000u | (mantissa << 13);  // inf / nan
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// rt_hdr.bin, rt_hdr_alb.bin and rt_hdr_alb_nrm.bin live next to each other: the configured path
// points at the color-only blob and the file name is swapped for the other feature sets.
std::string aux_weights_path(int auxCount) {
  std::string path = oidn_denoiser_weights_path();
  const size_t slash = path.find_last_of("/\\");
  const size_t base = (slash == std::string::npos) ? 0 : slash + 1;
  const char *name = "rt_hdr.bin";
  if (auxCount >= 2) name = "rt_hdr_alb_nrm.bin";
  else if (auxCount == 1) name = "rt_hdr_alb.bin";
  path.erase(base);
  path += name;
  return path;
}

struct PushConv {
  int cin, cout, w, h, relu, weightOfs;
};
struct PushPool {
  int channels, w, h;
};
struct PushUpsample {
  int channels, srcW, srcH, dstW, dstH, dstChannels, dstOfsC;
};
struct PushCopy {
  int channels, w, h, srcChannels, dstChannels, dstOfsC;
};
struct PushPre {
  float inputScale, normScale;
  int tileW, tileH;          // tile buffer size
  int imgW;                  // image width (source stride)
  int srcX, srcY;            // image position of the tile's first valid pixel
  int validW, validH;        // valid content size inside the tile
  int ofsX, ofsY;            // where that content starts inside the tile buffer
  int cin;                   // destination channels (3, 6 or 9)
  int nAux;                  // auxiliary images present (0, 1 = albedo, 2 = albedo + normal)
};
struct PushPost {
  float rcpNormScale, outputScale;
  int tileW, tileH;          // tile buffer size
  int imgW;                  // image width (destination stride)
  int outX, outY;            // image position of the tile's first written pixel
  int outW, outH;            // number of pixels written from this tile
  int ofsX, ofsY;            // where those pixels start inside the tile buffer
};

union PushData {
  PushConv conv;
  PushPool pool;
  PushUpsample upsample;
  PushCopy copy;
  PushPre pre;
  PushPost post;
  unsigned char raw[64];
};

uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties &memProps, VkMemoryRequirements req,
                          bool preferDeviceLocal, bool *ok) {
  const VkMemoryPropertyFlags want =
      preferDeviceLocal
          ? (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
          : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((req.memoryTypeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & want) == want) {
      *ok = true;
      return i;
    }
  }
  // Fall back to a host-visible heap without device-local.
  if (preferDeviceLocal) {
    *ok = false;
    return find_memory_type(memProps, req, false, ok);
  }
  *ok = false;
  return 0;
}

void destroy_buffer(VkDevice device, Buffer *b);

bool create_buffer(VkDevice device, const VkPhysicalDeviceMemoryProperties &memProps,
                   const void *initialData, size_t bytes, Buffer *out, bool preferDeviceLocal,
                   std::string &error,
                   VkBufferUsageFlags extraUsage = 0) {
  out->bytes = bytes;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = bytes;
  info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extraUsage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(device, &info, nullptr, &out->buffer) != VK_SUCCESS) {
    error = "vkCreateBuffer failed";
    return false;
  }
  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(device, out->buffer, &req);
  bool ok = false;
  const uint32_t typeIndex = find_memory_type(memProps, req, preferDeviceLocal, &ok);
  if (!ok) {
    error = "no host-visible coherent memory type for a denoiser buffer";
    destroy_buffer(device, out);
    return false;
  }
  VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc.allocationSize = req.size;
  alloc.memoryTypeIndex = typeIndex;
  if (vkAllocateMemory(device, &alloc, nullptr, &out->memory) != VK_SUCCESS) {
    error = "vkAllocateMemory failed (image too large for the device heap?)";
    destroy_buffer(device, out);
    return false;
  }
  if (vkBindBufferMemory(device, out->buffer, out->memory, 0) != VK_SUCCESS) {
    error = "vkBindBufferMemory failed";
    destroy_buffer(device, out);
    return false;
  }
  if (vkMapMemory(device, out->memory, 0, VK_WHOLE_SIZE, 0, &out->mapped) != VK_SUCCESS) {
    error = "vkMapMemory failed";
    destroy_buffer(device, out);
    return false;
  }
  if (initialData) std::memcpy(out->mapped, initialData, bytes);
  return true;
}

void destroy_buffer(VkDevice device, Buffer *b) {
  if (b->memory != VK_NULL_HANDLE) vkUnmapMemory(device, b->memory);
  if (b->buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, b->buffer, nullptr);
  if (b->memory != VK_NULL_HANDLE) vkFreeMemory(device, b->memory, nullptr);
  *b = Buffer{};
}

bool read_file(const char *path, std::vector<char> *out, std::string &error) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    error = std::string("cannot open ") + path;
    return false;
  }
  const std::size_t size = static_cast<std::size_t>(file.tellg());
  out->resize(size);
  file.seekg(0);
  file.read(out->data(), static_cast<std::streamsize>(size));
  return file.good() || size == 0;
}

// ---------------------------------------------------------------------------
// Tiling, following core/unet.h and UNetFilter::computeTileSize in core/unet.cpp:
//   alignment      = 16   (required spatial alignment in pixels)
//   receptiveField = 174
//   overlap        = round_up(receptiveField / 2, alignment) = 96
// The image is split into tiles whose buffer size is picked so that the tensor set stays inside
// the memory budget; every tile carries `overlap` pixels of real context around the region it
// contributes, which makes the tiled result identical to the untiled one.
// ---------------------------------------------------------------------------
const int kAlignment = 16;
const int kReceptiveField = 174;
const int kOverlap = 96;  // round_up(kReceptiveField / 2, kAlignment)

int round_up_i(int x, int a) { return ((x + a - 1) / a) * a; }
int ceil_div_i(int a, int b) { return (a + b - 1) / b; }

// Channel counts per resolution level, summed over all tensors of the network (see tensorDescs),
// with every tensor's channel stride already padded to the channel-blocked multiple of 4.
// The auxiliary feature sets only add channels at full resolution: enc_conv0 gains 3/6 input
// channels and concat1 gains the same amount; after padding that is +8 channels per auxiliary
// image (input 3->4->8->12 and concat1 68->72->76).
const int kChannelsFull = 4 + 32 + 32 + 68 + 64 + 32 + 4;       // 236
const int kChannelsHalf = 32 + 48 + 128 + 64 + 64;              // 336
const int kChannelsQuarter = 48 + 64 + 160 + 96 + 96;           // 464
const int kChannelsEighth = 64 + 80 + 160 + 112 + 112;          // 528
const int kChannelsSixteenth = 80 + 96 + 96;                    // 272

size_t tensor_bytes_for(int w, int h, size_t elemBytes, int auxCount) {
  const size_t px = static_cast<size_t>(w) * h;
  const int full = kChannelsFull + 8 * auxCount;
  return px * full * elemBytes + (px / 4) * kChannelsHalf * elemBytes +
         (px / 16) * kChannelsQuarter * elemBytes + (px / 64) * kChannelsEighth * elemBytes +
         (px / 256) * kChannelsSixteenth * elemBytes;
}

struct TilePlan {
  int tileW = 0;
  int tileH = 0;
  int countW = 1;
  int countH = 1;
};

TilePlan compute_tile_plan(int width, int height, size_t maxBytes, size_t elemBytes,
                           int auxCount) {
  TilePlan plan;
  plan.tileW = round_up_i(width, kAlignment);
  plan.tileH = round_up_i(height, kAlignment);
  const int minTileSize = 3 * kOverlap;
  while (tensor_bytes_for(plan.tileW, plan.tileH, elemBytes, auxCount) > maxBytes) {
    if (plan.tileH > minTileSize && plan.tileH > plan.tileW) {
      ++plan.countH;
      plan.tileH = std::max(round_up_i(ceil_div_i(height - 2 * kOverlap, plan.countH), kAlignment) +
                                2 * kOverlap,
                            minTileSize);
    } else if (plan.tileW > minTileSize) {
      ++plan.countW;
      plan.tileW = std::max(round_up_i(ceil_div_i(width - 2 * kOverlap, plan.countW), kAlignment) +
                                2 * kOverlap,
                            minTileSize);
    } else {
      break;
    }
  }
  plan.countH = (height > plan.tileH)
                    ? ceil_div_i(height - 2 * kOverlap, plan.tileH - 2 * kOverlap)
                    : 1;
  plan.countW = (width > plan.tileW)
                    ? ceil_div_i(width - 2 * kOverlap, plan.tileW - 2 * kOverlap)
                    : 1;
  return plan;
}

struct Resources {
  VkDevice device = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  std::vector<VkPipeline> pipelines;
  std::vector<Buffer *> buffers;
  std::vector<Buffer> paramsBufs;  // per-pass parameter blocks, only used by the clc backend
};

void release(const Resources &r) {
  if (r.device == VK_NULL_HANDLE) return;
  for (VkPipeline p : r.pipelines) {
    if (p != VK_NULL_HANDLE) vkDestroyPipeline(r.device, p, nullptr);
  }
  if (r.fence != VK_NULL_HANDLE) vkDestroyFence(r.device, r.fence, nullptr);
  if (r.commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(r.device, r.commandPool, nullptr);
  if (r.descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(r.device, r.descriptorPool, nullptr);
  if (r.pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(r.device, r.pipelineLayout, nullptr);
  if (r.dsl != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(r.device, r.dsl, nullptr);
  for (Buffer *b : r.buffers) destroy_buffer(r.device, b);
  for (const Buffer &b : r.paramsBufs) destroy_buffer(r.device, const_cast<Buffer *>(&b));
}

}  // namespace

const char *oidn_denoiser_weights_path() { return OIDN_DENOISER_WEIGHTS_PATH; }
const char *oidn_denoiser_shader_dir() { return OIDN_DENOISER_SPV_DIR; }

bool oidn_denoise_image(const OidnVulkanContext &ctx, const float *rgba, int width, int height,
                        std::vector<float> &outRgba, std::string &error, double *elapsedMs,
                        const OidnAuxImages *aux) {
  if (width <= 0 || height <= 0) {
    error = "the denoiser needs a positive image size";
    return false;
  }
  const VkDevice device = ctx.device;

  // Auxiliary images select the weight set, exactly like UNetFilter::execute's weight selection:
  // color only -> rt_hdr, color + albedo -> rt_hdr_alb, color + albedo + normal -> rt_hdr_alb_nrm.
  // OIDN rejects color + normal without albedo, so this does too.
  const bool hasAlbedo = aux != nullptr && aux->albedo != nullptr;
  const bool hasNormal = aux != nullptr && aux->normal != nullptr;
  if (hasNormal && !hasAlbedo) {
    error = "OIDN needs albedo when normal is given (color + normal is not a valid feature set)";
    return false;
  }
  const int auxCount = (hasNormal ? 2 : (hasAlbedo ? 1 : 0));
  const int inputChannels = 3 + 3 * auxCount;

  // ---- tensor storage precision -------------------------------------------
  // The default is FP16 tensor storage whenever the device offers
  // VkPhysicalDevice16BitStorageFeatures::storageBuffer16BitAccess, because it is the fastest
  // configuration measured here (see the table in the README: -6% to -19% per pass chain, and it
  // halves the footprint, which lets more images run as a single tile). The weights, the
  // shared-memory patch and every accumulator stay FP32 - that is the split OIDN's CUDA backend
  // uses by default too (FP16 tensors and weights, FP32 accumulation, see
  // devices/cuda/cuda_device.cpp and cuda_conv.cu). OIDN_STORE=fp32 forces the FP32 path (the
  // exact configuration the FP32 reference outputs come from), OIDN_STORE=fp16 forces FP16 even if
  // the probe said no, which then fails with a clear message.
  bool halfTensor = ctx.storage16Bit;
  if (const char *env = std::getenv("OIDN_STORE")) {
    const std::string store = env;
    if (store == "fp16" || store == "half") {
      if (!ctx.storage16Bit) {
        error = "OIDN_STORE=fp16 needs storageBuffer16BitAccess, which this device does not offer";
        return false;
      }
      halfTensor = true;
    } else if (store == "fp32") {
      halfTensor = false;
    } else {
      error = "OIDN_STORE must be fp32 or fp16";
      return false;
    }
  }
  const size_t tensorElemBytes = halfTensor ? sizeof(uint16_t) : sizeof(float);

  // ---- weights ------------------------------------------------------------
  std::vector<char> weightBytes;
  const std::string weightsPath = aux_weights_path(auxCount);
  if (!read_file(weightsPath.c_str(), &weightBytes, error)) return false;
  const size_t weightFloatCount = weightBytes.size() / sizeof(float);
  if (weightFloatCount < 1219140u) {
    error = std::string("weights blob is too small: ") + weightsPath.c_str() +
            " (run tools/oidn/export_oidn_weights.py first)";
    return false;
  }

  Resources res;
  res.device = device;
  // ctx.commandPool belongs to the caller and must survive this call, so it is not tracked here.
  res.commandPool = VK_NULL_HANDLE;

  Buffer weightsBuf;
  if (!create_buffer(device, ctx.memoryProperties, weightBytes.data(),
                     weightFloatCount * sizeof(float), &weightsBuf, false, error))
    return false;
  res.buffers.push_back(&weightsBuf);

  Buffer inputBuf, outputBuf;
  const size_t pixelCount = static_cast<size_t>(width) * height;
  if (!create_buffer(device, ctx.memoryProperties, rgba, pixelCount * 4 * sizeof(float), &inputBuf,
                     true, error))
    return false;
  if (!create_buffer(device, ctx.memoryProperties, nullptr, pixelCount * 4 * sizeof(float),
                     &outputBuf, false, error))
    return false;
  res.buffers.push_back(&inputBuf);
  res.buffers.push_back(&outputBuf);

  // Auxiliary images, or a 16-byte placeholder for the bindings nothing reads. The shader reads the
  // aux images with the same 4-float stride as the color image, so the rgb triplets are expanded
  // here.
  Buffer albBuf, nrmBuf;
  const size_t imageBytes = pixelCount * 4 * sizeof(float);
  std::vector<float> albRgba, nrmRgba;
  auto expand_rgb = [pixelCount](const float *src, std::vector<float> &dst) {
    dst.resize(pixelCount * 4);
    for (size_t i = 0; i < pixelCount; ++i) {
      dst[i * 4 + 0] = src[i * 3 + 0];
      dst[i * 4 + 1] = src[i * 3 + 1];
      dst[i * 4 + 2] = src[i * 3 + 2];
      dst[i * 4 + 3] = 1.0f;
    }
  };
  if (hasAlbedo) expand_rgb(aux->albedo, albRgba);
  if (hasNormal) expand_rgb(aux->normal, nrmRgba);
  if (!create_buffer(device, ctx.memoryProperties, hasAlbedo ? albRgba.data() : nullptr,
                     hasAlbedo ? imageBytes : 16, &albBuf, true, error))
    return false;
  if (!create_buffer(device, ctx.memoryProperties, hasNormal ? nrmRgba.data() : nullptr,
                     hasNormal ? imageBytes : 16, &nrmBuf, true, error))
    return false;
  res.buffers.push_back(&albBuf);
  res.buffers.push_back(&nrmBuf);

  // ---- tiling ------------------------------------------------------------
  // Same selection rule as UNetFilter::computeTileSize: shrink the tile until the tensor set fits
  // the memory budget (OIDN defaults to 3000 MB; override with OIDN_DENOISE_MAX_MB).
  size_t maxBytes = 3000ull * 1024 * 1024;
  if (const char *env = std::getenv("OIDN_DENOISE_MAX_MB")) {
    const long mb = std::atol(env);
    if (mb > 0) maxBytes = static_cast<size_t>(mb) * 1024 * 1024;
  }
  const TilePlan plan = compute_tile_plan(width, height, maxBytes, tensorElemBytes, auxCount);
  const int tileW = plan.tileW;
  const int tileH = plan.tileH;

  // ---- weight offsets -----------------------------------------------------
  // Same walk as tools/oidn/export_oidn_weights.py: 16 layers in buildNet order, every 3x3 row
  // padded from 9 to 12 floats and every bias block padded to a multiple of 4 floats. Two layers
  // depend on the auxiliary images: enc_conv0 consumes inputChannels channels, and dec_conv1a
  // consumes the concatenation of the upsampled decoder output (64) with the whole input tensor
  // (concat1 has 64 + inputChannels channels, getConcatDesc({upsample1Desc, inputReorderDesc}) in
  // core/unet.cpp), so every offset after enc_conv0 moves with the feature set.
  enum { L_E0 = 0, L_E1, L_E2, L_E3, L_E4, L_E5A, L_E5B, L_D4A, L_D4B, L_D3A, L_D3B,
         L_D2A, L_D2B, L_D1A, L_D1B, L_D0, L_COUNT };
  static const int layerCin[L_COUNT] = {3, 32, 32, 48, 64, 80, 96, 160, 112, 160, 96, 128, 64, 67, 64, 32};
  static const int layerCout[L_COUNT] = {32, 32, 48, 64, 80, 96, 96, 112, 112, 96, 96, 64, 64, 64, 32, 3};
  int layerWeightOfs[L_COUNT];
  {
    int ofs = 0;
    for (int i = 0; i < L_COUNT; ++i) {
      const int cin = (i == L_E0) ? inputChannels : (i == L_D1A ? 64 + inputChannels : layerCin[i]);
      layerWeightOfs[i] = ofs;
      ofs += layerCout[i] * cin * kWeightRowStride;
      ofs += ((layerCout[i] + 3) / 4) * 4;
    }
  }

  // ---- tensors (sized for one tile, not for the whole image) -------------
  std::vector<Tensor> tensors;
  size_t tensorBytes = 0;
  struct TensorDesc {
    int w, h, c;
    const char *name;
  };
  const int w1 = tileW, h1 = tileH;
  const int w2 = tileW / 2, h2 = tileH / 2;
  const int w4 = tileW / 4, h4 = tileH / 4;
  const int w8 = tileW / 8, h8 = tileH / 8;
  const int w16 = tileW / 16, h16 = tileH / 16;
  const TensorDesc tensorDescs[] = {
      {w1, h1, inputChannels, "input"},
      {w1, h1, 32, "enc_conv0"},
      {w1, h1, 32, "enc_conv1"},
      {w2, h2, 32, "pool1"},
      {w2, h2, 48, "enc_conv2"},
      {w4, h4, 48, "pool2"},
      {w4, h4, 64, "enc_conv3"},
      {w8, h8, 64, "pool3"},
      {w8, h8, 80, "enc_conv4"},
      {w16, h16, 80, "pool4"},
      {w16, h16, 96, "enc_conv5a"},
      {w16, h16, 96, "enc_conv5b"},
      {w8, h8, 160, "concat4"},
      {w8, h8, 112, "dec_conv4a"},
      {w8, h8, 112, "dec_conv4b"},
      {w4, h4, 160, "concat3"},
      {w4, h4, 96, "dec_conv3a"},
      {w4, h4, 96, "dec_conv3b"},
      {w2, h2, 128, "concat2"},
      {w2, h2, 64, "dec_conv2a"},
      {w2, h2, 64, "dec_conv2b"},
      {w1, h1, 64 + inputChannels, "concat1"},
      {w1, h1, 64, "dec_conv1a"},
      {w1, h1, 32, "dec_conv1b"},
      {w1, h1, 3, "dec_conv0"},
  };
  tensors.resize(sizeof(tensorDescs) / sizeof(tensorDescs[0]));
  for (size_t i = 0; i < tensors.size(); ++i) {
    Tensor &t = tensors[i];
    t.w = tensorDescs[i].w;
    t.h = tensorDescs[i].h;
    t.c = tensorDescs[i].c;
    t.cPad = pad4i(t.c);
    const size_t bytes = static_cast<size_t>(t.w) * t.h * t.cPad * tensorElemBytes;
    if (!create_buffer(device, ctx.memoryProperties, nullptr, bytes, &t.buf, true, error)) {
      release(res);
      return false;
    }
    tensorBytes += bytes;
    res.buffers.push_back(&t.buf);
  }

  // Tensor indices (see tensorDescs above).
  enum {
    T_IN = 0, T_E0, T_E1, T_P1, T_E2, T_P2, T_E3, T_P3, T_E4, T_P4,
    T_E5A, T_E5B, T_CAT4, T_D4A, T_D4B, T_CAT3, T_D3A, T_D3B,
    T_CAT2, T_D2A, T_D2B, T_CAT1, T_D1A, T_D1B, T_OUT
  };

  // ---- pass table (mirrors core/unet.cpp buildNet) ------------------------
  static const Pass passes[] = {
      {PASS_PRE, -1, T_IN, 0, 0, 0, 0, 0},
      {PASS_CONV, T_IN, T_E0, 3, 32, L_E0, 1, 0},
      {PASS_CONV, T_E0, T_E1, 32, 32, L_E1, 1, 0},
      {PASS_POOL, T_E1, T_P1, 32, 0, 0, 0, 0},
      {PASS_CONV, T_P1, T_E2, 32, 48, L_E2, 1, 0},
      {PASS_POOL, T_E2, T_P2, 48, 0, 0, 0, 0},
      {PASS_CONV, T_P2, T_E3, 48, 64, L_E3, 1, 0},
      {PASS_POOL, T_E3, T_P3, 64, 0, 0, 0, 0},
      {PASS_CONV, T_P3, T_E4, 64, 80, L_E4, 1, 0},
      {PASS_POOL, T_E4, T_P4, 80, 0, 0, 0, 0},
      {PASS_CONV, T_P4, T_E5A, 80, 96, L_E5A, 1, 0},
      {PASS_CONV, T_E5A, T_E5B, 96, 96, L_E5B, 1, 0},
      {PASS_UPSAMPLE, T_E5B, T_CAT4, 96, 0, 0, 0, 0},
      {PASS_COPY, T_P3, T_CAT4, 64, 0, 0, 0, 96},
      {PASS_CONV, T_CAT4, T_D4A, 160, 112, L_D4A, 1, 0},
      {PASS_CONV, T_D4A, T_D4B, 112, 112, L_D4B, 1, 0},
      {PASS_UPSAMPLE, T_D4B, T_CAT3, 112, 0, 0, 0, 0},
      {PASS_COPY, T_P2, T_CAT3, 48, 0, 0, 0, 112},
      {PASS_CONV, T_CAT3, T_D3A, 160, 96, L_D3A, 1, 0},
      {PASS_CONV, T_D3A, T_D3B, 96, 96, L_D3B, 1, 0},
      {PASS_UPSAMPLE, T_D3B, T_CAT2, 96, 0, 0, 0, 0},
      {PASS_COPY, T_P1, T_CAT2, 32, 0, 0, 0, 96},
      {PASS_CONV, T_CAT2, T_D2A, 128, 64, L_D2A, 1, 0},
      {PASS_CONV, T_D2A, T_D2B, 64, 64, L_D2B, 1, 0},
      {PASS_UPSAMPLE, T_D2B, T_CAT1, 64, 0, 0, 0, 0},
      {PASS_COPY, T_IN, T_CAT1, 0, 0, 0, 0, 64},
      {PASS_CONV, T_CAT1, T_D1A, 67, 64, L_D1A, 1, 0},
      {PASS_CONV, T_D1A, T_D1B, 64, 32, L_D1B, 1, 0},
      {PASS_CONV, T_D1B, T_OUT, 32, 3, L_D0, 0, 0},
      {PASS_POST, T_OUT, -1, 0, 0, 0, 0, 0},
  };
  const size_t passCount = sizeof(passes) / sizeof(passes[0]);

  // Human readable pass names, used only by the per-pass timing below.
  static const char *passNames[] = {
      "pre",   "enc0",  "enc1",  "pool1", "enc2",  "pool2", "enc3", "pool3", "enc4",  "pool4",
      "enc5a", "enc5b", "up1",   "cat4",  "dec4a", "dec4b", "up2",  "cat3",  "dec3a", "dec3b",
      "up3",   "cat2",  "dec2a", "dec2b", "up4",   "cat1",  "dec1a", "dec1b", "dec0",  "post"};

  // ---- backend selection --------------------------------------------------
  // Two complete kernel sets are built into every binary: the GLSL passes compiled by glslc
  // (src/oidn_*.comp) and the C++ for OpenCL passes compiled by clspv (src/oidn_*.cl).
  // OIDN_BACKEND=glsl selects the former; anything else (including no variable at all) selects
  // the latter, which is the default after the two were measured side by side.
  const char *backendEnv = std::getenv("OIDN_BACKEND");
  const bool useGlsl = backendEnv != nullptr && std::strcmp(backendEnv, "glsl") == 0;
  const bool useClc = !useGlsl;
  const bool useTiledConv = std::getenv("OIDN_CONV") == nullptr ||
                            std::strcmp(std::getenv("OIDN_CONV"), "simple") != 0;

  // Register blocking of the tiled convolution (src/oidn_conv_tiled.cl): one work group of 8x8
  // threads covers a (8*PX)x(8*PY) pixel block for `group` output channels, and every thread
  // computes PX*PY pixels. The kernel is compiled for 3x2 pixels over 4 channels (block 24x16),
  // which measured fastest here; OIDN_TILE=PXxPYgGROUP switches to one of the other modules built
  // alongside it. The GLSL pass only implements the 2x2 / 8-channel decomposition, so the GLSL
  // backend always uses that geometry.
  uint32_t tilePX = 3, tilePY = 2, tileGroup = 4;
  const char *tileEnv = std::getenv("OIDN_TILE");
  std::string tileModule = "oidn_conv_tiled";
  if (tileEnv != nullptr) {
    unsigned px = 0, py = 0, group = 0;
    if (std::sscanf(tileEnv, "%ux%ug%u", &px, &py, &group) == 3 && px > 0 && py > 0 && group > 0 &&
        group % 4 == 0) {
      tilePX = px;
      tilePY = py;
      tileGroup = group;
      if (!(px == 3 && py == 2 && group == 4)) tileModule = std::string("oidn_conv_tiled_") + tileEnv;
    } else {
      std::printf("[oidn] OIDN_TILE=%.32s not understood (expected PXxPYgGROUP), using 3x2g4\n",
                  tileEnv);
    }
  }
  const uint32_t convBlockX = useTiledConv ? (useClc ? 8u * tilePX : 16u) : 8u;
  const uint32_t convBlockY = useTiledConv ? (useClc ? 8u * tilePY : 16u) : 8u;
  const uint32_t convGroup = useTiledConv ? (useClc ? tileGroup : 8u) : 8u;

  // Which buffer feeds which binding, per pipeline, for the clc backend. clspv numbers the
  // descriptor bindings by the pointer arguments the kernel actually keeps, so the numbering
  // differs from the GLSL layout (albedo / normal land on 2 and 3 instead of 3 and 4, and every
  // pass carries its parameter block in the last slot because a POD push constant block is not
  // possible in CLC++ mode):
  //   pre                              0 src, 1 dst, 2 albedo, 3 normal, 4 params
  //   conv / conv_tiled                0 src, 1 dst, 2 weights, 3 params
  //   pool / upsample / concat_copy / post   0 src, 1 dst, 2 params
  enum ClcBuf { CB_SRC, CB_DST, CB_WGT, CB_ALB, CB_NRM, CB_PARAMS, CB_NONE };
  static const int clcBinding[7][5] = {
      {CB_SRC, CB_DST, CB_ALB, CB_NRM, CB_PARAMS},    // 0 pre
      {CB_SRC, CB_DST, CB_WGT, CB_PARAMS, CB_NONE},   // 1 conv
      {CB_SRC, CB_DST, CB_PARAMS, CB_NONE, CB_NONE},  // 2 pool
      {CB_SRC, CB_DST, CB_PARAMS, CB_NONE, CB_NONE},  // 3 upsample
      {CB_SRC, CB_DST, CB_PARAMS, CB_NONE, CB_NONE},  // 4 concat_copy
      {CB_SRC, CB_DST, CB_PARAMS, CB_NONE, CB_NONE},  // 5 post
      {CB_SRC, CB_DST, CB_WGT, CB_PARAMS, CB_NONE},   // 6 conv_tiled
  };
  const int pipeOf[6] = {0, useTiledConv ? 6 : 1, 2, 3, 4, 5};  // by PassKind

  // ---- descriptor set layout / pool / sets -------------------------------
  // Five storage buffers: src, dst, weights for every pass, plus the albedo / normal images that
  // only the input reorder reads (bindings 3 and 4). A shader may declare a subset of the layout.
  VkDescriptorSetLayoutBinding bindings[5]{};
  for (uint32_t i = 0; i < 5; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslInfo.bindingCount = 5;
  dslInfo.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &res.dsl) != VK_SUCCESS) {
    error = "vkCreateDescriptorSetLayout failed";
    release(res);
    return false;
  }

  VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 * static_cast<uint32_t>(passCount)};
  VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets = static_cast<uint32_t>(passCount);
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &res.descriptorPool) != VK_SUCCESS) {
    error = "vkCreateDescriptorPool failed";
    release(res);
    return false;
  }

  // Parameter blocks for the clc backend: one 64-byte buffer per pass, bound to the pass's last
  // binding and rewritten before every dispatch (vkCmdUpdateBuffer). The GLSL backend passes the
  // same bytes as a push constant instead.
  if (useClc) {
    res.paramsBufs.resize(passCount);
    for (size_t i = 0; i < passCount; ++i) {
      if (!create_buffer(device, ctx.memoryProperties, nullptr, 64, &res.paramsBufs[i], false,
                         error, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
        release(res);
        return false;
      }
    }
  }

  std::vector<VkDescriptorSet> sets(passCount, VK_NULL_HANDLE);
  std::vector<VkDescriptorSetLayout> setLayouts(passCount, res.dsl);
  VkDescriptorSetAllocateInfo setAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  setAlloc.descriptorPool = res.descriptorPool;
  setAlloc.descriptorSetCount = static_cast<uint32_t>(passCount);
  setAlloc.pSetLayouts = setLayouts.data();
  if (vkAllocateDescriptorSets(device, &setAlloc, sets.data()) != VK_SUCCESS) {
    error = "vkAllocateDescriptorSets failed";
    release(res);
    return false;
  }
  for (size_t i = 0; i < passCount; ++i) {
    const Pass &p = passes[i];
    VkBuffer src = (p.src < 0) ? inputBuf.buffer : tensors[static_cast<size_t>(p.src)].buf.buffer;
    VkBuffer dst = (p.dst < 0) ? outputBuf.buffer : tensors[static_cast<size_t>(p.dst)].buf.buffer;
    VkDescriptorBufferInfo infos[5]{};
    VkWriteDescriptorSet writes[5]{};
    uint32_t writeCount = 0;
    if (useClc) {
      const int pipe = pipeOf[p.kind];
      for (uint32_t b = 0; b < 5; ++b) {
        VkBuffer buf = VK_NULL_HANDLE;
        switch (clcBinding[pipe][b]) {
          case CB_SRC: buf = src; break;
          case CB_DST: buf = dst; break;
          case CB_WGT: buf = weightsBuf.buffer; break;
          case CB_ALB: buf = albBuf.buffer; break;
          case CB_NRM: buf = nrmBuf.buffer; break;
          case CB_PARAMS: buf = res.paramsBufs[i].buffer; break;
          default: continue;
        }
        infos[writeCount] = {buf, 0, VK_WHOLE_SIZE};
        writes[writeCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[writeCount].dstSet = sets[i];
        writes[writeCount].dstBinding = b;
        writes[writeCount].descriptorCount = 1;
        writes[writeCount].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[writeCount].pBufferInfo = &infos[writeCount];
        ++writeCount;
      }
    } else {
      const VkBuffer all[5] = {src, dst, weightsBuf.buffer, albBuf.buffer, nrmBuf.buffer};
      for (uint32_t b = 0; b < 5; ++b) {
        infos[b] = {all[b], 0, VK_WHOLE_SIZE};
        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[b].dstSet = sets[i];
        writes[b].dstBinding = b;
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &infos[b];
      }
      writeCount = 5;
    }
    vkUpdateDescriptorSets(device, writeCount, writes, 0, nullptr);
  }

  // ---- pipelines ---------------------------------------------------------
  VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plInfo.setLayoutCount = 1;
  plInfo.pSetLayouts = &res.dsl;
  VkPushConstantRange range{};
  range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  range.offset = 0;
  range.size = 64;
  plInfo.pushConstantRangeCount = 1;
  plInfo.pPushConstantRanges = &range;
  if (vkCreatePipelineLayout(device, &plInfo, nullptr, &res.pipelineLayout) != VK_SUCCESS) {
    error = "vkCreatePipelineLayout failed";
    release(res);
    return false;
  }

  // Pipeline 6 is the shared-memory / register-blocked convolution. The simple one stays
  // available so the two can be compared on the same input (OIDN_CONV=simple).
  static const char *shaderNames[] = {"oidn_pre.spv",  "oidn_conv.spv", "oidn_pool.spv",
                                     "oidn_upsample.spv", "oidn_concat_copy.spv", "oidn_post.spv",
                                     "oidn_conv_tiled.spv"};
  // The tiled convolution's module depends on the register blocking (clc backend only; the GLSL
  // pass is a single 2x2 build).
  static std::string tiledName;
  if (useClc && useTiledConv) {
    tiledName = tileModule + ".spv";
    shaderNames[6] = tiledName.c_str();
  }
  // clspv cannot emit an entry point called "main": the entry keeps the kernel name.
  static const char *entryNamesClc[] = {"oidn_pre",     "oidn_conv",     "oidn_pool",
                                        "oidn_upsample", "oidn_concat_copy", "oidn_post",
                                        "oidn_conv_tiled"};
  const size_t shaderCount = sizeof(shaderNames) / sizeof(shaderNames[0]);
  res.pipelines.assign(shaderCount, VK_NULL_HANDLE);
  for (size_t i = 0; i < shaderCount; ++i) {
    std::vector<char> code;
    // The FP16 variants are the same sources built with -DOIDN_HALF_TENSOR=1; the clc variants sit
    // next to them with a _cl / _cl_h suffix.
    std::string shaderName = shaderNames[i];
    if (useClc) {
      shaderName = shaderName.substr(0, shaderName.size() - 4) + (halfTensor ? "_cl_h.spv" : "_cl.spv");
    } else if (halfTensor) {
      shaderName.replace(shaderName.size() - 4, 4, "_h.spv");
    }
    const std::string path = std::string(OIDN_DENOISER_SPV_DIR) + "/" + shaderName;
    if (!read_file(path.c_str(), &code, error)) {
      release(res);
      return false;
    }
    VkShaderModuleCreateInfo modInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    modInfo.codeSize = code.size();
    modInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &modInfo, nullptr, &module) != VK_SUCCESS) {
      error = std::string("vkCreateShaderModule failed for ") + path;
      release(res);
      return false;
    }
    VkComputePipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = module;
    pipeInfo.stage.pName = useClc ? entryNamesClc[i] : "main";
    pipeInfo.layout = res.pipelineLayout;
    const VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr,
                                                &res.pipelines[i]);
    vkDestroyShaderModule(device, module, nullptr);
    if (r != VK_SUCCESS) {
      error = std::string("vkCreateComputePipelines failed for ") + path;
      release(res);
      return false;
    }
  }

  // ---- per-pass GPU timing (OIDN_PROFILE=1) -------------------------------
  // One timestamp before every dispatch plus one at the end of each tile chain, so the delta
  // between consecutive stamps is the wall time of exactly one pass. Instrumentation only: it
  // changes no dispatch, so the measured total stays comparable with an unprofiled run.
  const bool profile = std::getenv("OIDN_PROFILE") != nullptr;
  VkQueryPool queryPool = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties physProps{};
  vkGetPhysicalDeviceProperties(ctx.physicalDevice, &physProps);
  const size_t queriesPerTile = passCount + 1;
  const size_t maxQueries = queriesPerTile * static_cast<size_t>(plan.countW) * plan.countH;
  if (profile) {
    VkQueryPoolCreateInfo qpInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpInfo.queryCount = static_cast<uint32_t>(maxQueries);
    if (vkCreateQueryPool(device, &qpInfo, nullptr, &queryPool) != VK_SUCCESS) {
      error = "vkCreateQueryPool failed";
      release(res);
      return false;
    }
  }

  // ---- record ------------------------------------------------------------
  VkCommandBufferAllocateInfo cbInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbInfo.commandPool = ctx.commandPool;
  cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbInfo.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(device, &cbInfo, &res.cmd) != VK_SUCCESS) {
    error = "vkAllocateCommandBuffers failed";
    release(res);
    return false;
  }

  const float inputScale = compute_autoexposure(rgba, width, height);
  const float normScale = 1.0f / puForward(HDR_Y_MAX);
  const float rcpNormScale = puForward(HDR_Y_MAX);
  const float outputScale = 1.0f / inputScale;

  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(res.cmd, &beginInfo) != VK_SUCCESS) {
    error = "vkBeginCommandBuffer failed";
    if (queryPool) vkDestroyQueryPool(device, queryPool, nullptr);
    release(res);
    return false;
  }
  if (queryPool) vkCmdResetQueryPool(res.cmd, queryPool, 0, static_cast<uint32_t>(maxQueries));

  if (useClc) {
    // clspv always emits the OpenCL 2.0 global offset block as a 12-byte module-scope push constant
    // in CLC++ mode. The dispatches use plain global ids, so it stays zero; pushing it once here
    // covers every pass because no other push constant is written on this path.
    const unsigned char zeroOffset[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    vkCmdPushConstants(res.cmd, res.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(zeroOffset), zeroOffset);
  }

  // ---- one dispatch chain per tile (UNetFilter::execute, core/unet.cpp) ------------------
  for (int ti = 0; ti < plan.countH; ++ti) {
    const int h0 = ti * (tileH - 2 * kOverlap);           // tile origin in the image
    const int overlapBeginH = (ti > 0) ? kOverlap : 0;
    const int overlapEndH = (ti < plan.countH - 1) ? kOverlap : 0;
    const int tileH1 = std::min(height - h0, tileH);      // valid rows carried by the tile
    const int tileH2 = tileH1 - overlapBeginH - overlapEndH;  // rows this tile contributes
    const int ofsH = tileH - round_up_i(tileH1, kAlignment);  // content aligned to the bottom

    for (int tj = 0; tj < plan.countW; ++tj) {
      const int w0 = tj * (tileW - 2 * kOverlap);
      const int overlapBeginW = (tj > 0) ? kOverlap : 0;
      const int overlapEndW = (tj < plan.countW - 1) ? kOverlap : 0;
      const int tileW1 = std::min(width - w0, tileW);
      const int tileW2 = tileW1 - overlapBeginW - overlapEndW;
      const int ofsW = tileW - round_up_i(tileW1, kAlignment);

      for (size_t i = 0; i < passCount; ++i) {
        const Pass &p = passes[i];
        PushData push{};
        VkPipeline pipeline = VK_NULL_HANDLE;

        switch (p.kind) {
          case PASS_PRE: {
            push.pre.inputScale = inputScale;
            push.pre.normScale = normScale;
            push.pre.tileW = tileW;
            push.pre.tileH = tileH;
            push.pre.imgW = width;
            push.pre.srcX = w0;
            push.pre.srcY = h0;
            push.pre.validW = tileW1;
            push.pre.validH = tileH1;
            push.pre.ofsX = ofsW;
            push.pre.ofsY = ofsH;
            push.pre.cin = inputChannels;
            push.pre.nAux = auxCount;
            pipeline = res.pipelines[0];
            break;
          }
          case PASS_CONV: {
            // enc_conv0 reads the reorder output, whose channel count depends on the aux images.
            push.conv.cin = tensors[static_cast<size_t>(p.src)].c;
            push.conv.cout = p.cout;
            push.conv.w = tensors[static_cast<size_t>(p.dst)].w;
            push.conv.h = tensors[static_cast<size_t>(p.dst)].h;
            push.conv.relu = p.relu;
            push.conv.weightOfs = layerWeightOfs[p.weightOfs];  // p.weightOfs is a layer index
            pipeline = res.pipelines[useTiledConv ? 6 : 1];
            break;
          }
          case PASS_POOL: {
            push.pool.channels = tensors[static_cast<size_t>(p.src)].c;
            push.pool.w = tensors[static_cast<size_t>(p.dst)].w;
            push.pool.h = tensors[static_cast<size_t>(p.dst)].h;
            pipeline = res.pipelines[2];
            break;
          }
          case PASS_UPSAMPLE: {
            const Tensor &src = tensors[static_cast<size_t>(p.src)];
            const Tensor &dst = tensors[static_cast<size_t>(p.dst)];
            push.upsample.channels = src.c;
            push.upsample.srcW = src.w;
            push.upsample.srcH = src.h;
            push.upsample.dstW = dst.w;
            push.upsample.dstH = dst.h;
            push.upsample.dstChannels = dst.c;
            push.upsample.dstOfsC = p.dstOfsC;
            pipeline = res.pipelines[3];
            break;
          }
          case PASS_COPY: {
            const Tensor &src = tensors[static_cast<size_t>(p.src)];
            const Tensor &dst = tensors[static_cast<size_t>(p.dst)];
            push.copy.channels = src.c;
            push.copy.w = src.w;
            push.copy.h = src.h;
            push.copy.srcChannels = src.c;
            push.copy.dstChannels = dst.c;
            push.copy.dstOfsC = p.dstOfsC;
            pipeline = res.pipelines[4];
            break;
          }
          case PASS_POST: {
            push.post.rcpNormScale = rcpNormScale;
            push.post.outputScale = outputScale;
            push.post.tileW = tileW;
            push.post.tileH = tileH;
            push.post.imgW = width;
            push.post.outX = w0 + overlapBeginW;
            push.post.outY = h0 + overlapBeginH;
            push.post.outW = tileW2;
            push.post.outH = tileH2;
            push.post.ofsX = ofsW + overlapBeginW;
            push.post.ofsY = ofsH + overlapBeginH;
            pipeline = res.pipelines[5];
            break;
          }
        }

        if (queryPool) {
          const size_t qBase = (static_cast<size_t>(ti) * plan.countW + tj) * queriesPerTile;
          vkCmdWriteTimestamp(res.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool,
                              static_cast<uint32_t>(qBase + i));
        }

        vkCmdBindPipeline(res.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(res.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipelineLayout, 0, 1,
                                &sets[i], 0, nullptr);
        if (useClc) {
          // The clc pass reads its parameters from the buffer bound to its last binding, so upload
          // the block with vkCmdUpdateBuffer and make the write visible to the dispatch. The push
          // constant range only ever carries the 12-byte global offset block that clspv emits in
          // CLC++ mode (pushed once, all zero, before the tile loops).
          vkCmdUpdateBuffer(res.cmd, res.paramsBufs[i].buffer, 0, 64, push.raw);
          VkMemoryBarrier uploadBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
          uploadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          uploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
          vkCmdPipelineBarrier(res.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &uploadBarrier, 0,
                               nullptr, 0, nullptr);
        } else {
          vkCmdPushConstants(res.cmd, res.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 64,
                             push.raw);
        }

        uint32_t gx = 0, gy = 1, gz = 1;
        if (p.kind == PASS_CONV) {
          const Tensor &dst = tensors[static_cast<size_t>(p.dst)];
          gx = static_cast<uint32_t>((dst.w + convBlockX - 1) / convBlockX);
          gy = static_cast<uint32_t>((dst.h + convBlockY - 1) / convBlockY);
          gz = static_cast<uint32_t>((p.cout + convGroup - 1) / convGroup);
        } else if (p.kind == PASS_PRE || p.kind == PASS_POST) {
          gx = static_cast<uint32_t>((tileW + 7) / 8);
          gy = static_cast<uint32_t>((tileH + 7) / 8);
        } else {
          const Tensor &src = tensors[static_cast<size_t>(p.src)];
          const Tensor &dst = tensors[static_cast<size_t>(p.dst)];
          gx = static_cast<uint32_t>((dst.w + 7) / 8);
          gy = static_cast<uint32_t>((dst.h + 7) / 8);
          // pool / upsample / copy work one 4-channel blocked chunk of their source per thread
          gz = static_cast<uint32_t>(src.cPad / 4);
        }
        vkCmdDispatch(res.cmd, gx, gy, gz);

        // Every pass reads what the previous pass wrote.
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(res.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                             nullptr);

        // Close the per-tile profile chain after the last pass of the tile.
        if (queryPool && i + 1 == passCount) {
          const size_t qBase = (static_cast<size_t>(ti) * plan.countW + tj) * queriesPerTile;
          vkCmdWriteTimestamp(res.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool,
                              static_cast<uint32_t>(qBase + passCount));
        }
      }
    }
  }

  VkMemoryBarrier hostBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(res.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                       1, &hostBarrier, 0, nullptr, 0, nullptr);
  if (vkEndCommandBuffer(res.cmd) != VK_SUCCESS) {
    error = "vkEndCommandBuffer failed";
    if (queryPool) vkDestroyQueryPool(device, queryPool, nullptr);
    release(res);
    return false;
  }

  // ---- submit ------------------------------------------------------------
  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vkCreateFence(device, &fenceInfo, nullptr, &res.fence) != VK_SUCCESS) {
    error = "vkCreateFence failed";
    release(res);
    return false;
  }
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &res.cmd;

  const auto t0 = std::chrono::high_resolution_clock::now();
  if (vkQueueSubmit(ctx.queue, 1, &submit, res.fence) != VK_SUCCESS ||
      vkWaitForFences(device, 1, &res.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    error = "queue submission failed";
    release(res);
    return false;
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  if (elapsedMs) *elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Per-pass breakdown, averaged over the tiles (the dispatch chain is identical per tile).
  if (queryPool) {
    std::vector<uint64_t> stamps(maxQueries, 0);
    const VkResult qr = vkGetQueryPoolResults(device, queryPool, 0, static_cast<uint32_t>(maxQueries),
                                            maxQueries * sizeof(uint64_t), stamps.data(),
                                            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (qr == VK_SUCCESS) {
      const double msPerTick = static_cast<double>(physProps.limits.timestampPeriod) / 1e6;
      const int tiles = plan.countW * plan.countH;
      std::vector<double> sums(passCount, 0.0);
      for (int t = 0; t < tiles; ++t) {
        const size_t qBase = static_cast<size_t>(t) * queriesPerTile;
        for (size_t i = 0; i < passCount; ++i) {
          sums[i] += static_cast<double>(stamps[qBase + i + 1] - stamps[qBase + i]) * msPerTick;
        }
      }
      double total = 0.0;
      for (size_t i = 0; i < passCount; ++i) total += sums[i];
      std::printf("[profile] per pass, mean over %d tile(s), total %.1f ms\n", tiles, total / tiles);
      for (size_t i = 0; i < passCount; ++i) {
        std::printf("[profile] %-6s %7.2f ms  %5.1f%%\n", passNames[i], sums[i] / tiles,
                    100.0 * sums[i] / total);
      }
    } else {
      std::printf("[profile] query results unavailable (%d)\n", static_cast<int>(qr));
    }
    vkDestroyQueryPool(device, queryPool, nullptr);
  }

  outRgba.resize(pixelCount * 4);
  std::memcpy(outRgba.data(), outputBuf.mapped, pixelCount * 4 * sizeof(float));

  // Debug hook: OIDN_DUMP_TENSOR=<name> writes that tile tensor out as raw float32, which is how
  // the intermediate stages can be compared against the numpy reference one at a time.
  if (const char *dumpName = std::getenv("OIDN_DUMP_TENSOR")) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (std::strcmp(tensorDescs[i].name, dumpName) != 0) continue;
      const Tensor &t = tensors[i];
      const size_t floats = static_cast<size_t>(t.w) * t.h * t.c;
      std::FILE *f = std::fopen((std::string("dump_") + dumpName + ".f32").c_str(), "wb");
      if (f) {
        // Compact the channel-blocked layout back to w*h*c floats, which is the contract the
        // numpy reference comparisons in tools/oidn expect.
        const size_t pixels = static_cast<size_t>(t.w) * t.h;
        std::vector<float> tmp(floats);
        const uint16_t *src16 = halfTensor ? static_cast<const uint16_t *>(t.buf.mapped) : nullptr;
        const float *src32 = halfTensor ? nullptr : static_cast<const float *>(t.buf.mapped);
        for (size_t px = 0; px < pixels; ++px) {
          for (int c = 0; c < t.c; ++c) {
            const size_t k = px * static_cast<size_t>(t.cPad) + static_cast<size_t>(c);
            tmp[px * static_cast<size_t>(t.c) + static_cast<size_t>(c)] =
                halfTensor ? half_to_float(src16[k]) : src32[k];
          }
        }
        std::fwrite(tmp.data(), sizeof(float), floats, f);
        std::fclose(f);
        std::printf("[oidn] dumped tensor %s (%dx%dx%d)\n", dumpName, t.w, t.h, t.c);
      }
      break;
    }
  }

  std::printf("[oidn] kernels %s, weights %.2f MB, tensors %.2f MB %s (%dx%d tile, %zu dispatches per tile)\n",
              useClc ? "clspv C++" : "GLSL",
              static_cast<double>(weightsBuf.bytes) / (1024.0 * 1024.0),
              static_cast<double>(tensorBytes) / (1024.0 * 1024.0),
              halfTensor ? "FP16" : "FP32", tileW, tileH, passCount);
  if (useClc && useTiledConv)
    std::printf("[oidn] tiled convolution: %ux%u pixels per thread, %u channels per work group "
                "(block %ux%u)\n",
                tilePX, tilePY, tileGroup, convBlockX, convBlockY);
  std::printf("[oidn] image %dx%d -> %d tile(s) of at most %dx%d, overlap %d, alignment %d\n",
              width, height, plan.countW * plan.countH, tileW, tileH, kOverlap, kAlignment);
  std::printf("[oidn] autoexposure: inputScale %.4f (mean log2 luminance normalised to key 0.18)\n",
              static_cast<double>(inputScale));

  vkFreeCommandBuffers(device, ctx.commandPool, 1, &res.cmd);
  res.cmd = VK_NULL_HANDLE;
  release(res);
  return true;
}
