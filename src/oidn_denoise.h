#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

// The denoiser borrows the instance/device/queue that the renderer already created; it only
// allocates its own buffers, descriptor pool, pipelines and command buffer.
struct OidnVulkanContext {
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties memoryProperties{};
  // Set by the caller after probing VkPhysicalDevice16BitStorageFeatures::storageBuffer16BitAccess
  // (the feature must also be enabled at device creation). It selects the default tensor storage
  // precision: FP16 when available, FP32 otherwise. OIDN_STORE=fp16/fp32 overrides it.
  bool storage16Bit = false;
};

// Optional auxiliary images (albedo / normal) for the RT filter. Each is a linear float buffer with
// `width * height * 3` floats (RGB, row major), the same dimensions as the color image: albedo is
// clamped to [0, 1] and not passed through the pre-exposure transfer function, normal is clamped to
// [-1, 1] and remapped to [0, 1], exactly like InputReorder::storeAlbedo / storeNormal. Setting only
// `albedo` uses the rt_hdr_alb weights; setting both uses rt_hdr_alb_nrm. Passing null (or a
// structure with null members) keeps the color-only rt_hdr weights.
struct OidnAuxImages {
  const float* albedo = nullptr;
  const float* normal = nullptr;
};

// Runs the OpenImageDenoise RT U-Net (rt_hdr / rt_hdr_alb / rt_hdr_alb_nrm weights, chosen by
// `aux`) on a linear HDR image with Vulkan compute shaders. `rgba` holds 4 floats per pixel; any
// image size is accepted, because large images are split into 16-aligned tiles with a 96 pixel
// overlap exactly like OIDN's UNetFilter. The result is written to `outRgba` with the same layout,
// so it can go straight into the image writer.
//
// Returns false and fills `error` on failure. `elapsedMs`, when non-null, receives the wall-clock
// time of the whole GPU pass chain (dispatch + submit + wait).
bool oidn_denoise_image(const OidnVulkanContext& ctx, const float* rgba, int width, int height,
                        std::vector<float>& outRgba, std::string& error, double* elapsedMs,
                        const OidnAuxImages* aux = nullptr);

// Paths the denoiser uses at run time (useful for diagnostics / error messages).
const char* oidn_denoiser_weights_path();
const char* oidn_denoiser_shader_dir();
