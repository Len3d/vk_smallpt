// oidn_denoise_tool.cpp -- standalone driver for the Vulkan OIDN denoiser.
//
// It reads a raw float32 linear RGB image (HWC, as tools/oidn/ref_forward.py writes), runs the
// denoiser, and writes the result back in the same format, plus an optional gamma-encoded PPM.
// This is the fast iteration path: the same code the smallpt example calls after rendering.
//
// Usage: oidn_denoise_tool.exe <in.f32> <width> <height> <out.f32> [out.ppm] [--alb alb.f32] [--nrm nrm.f32]
// The auxiliary images are raw float32 RGB (width*height*3), like the color image. --alb alone
// selects the rt_hdr_alb weights, --alb plus --nrm selects rt_hdr_alb_nrm.

#include <vulkan/vulkan.h>

#include "oidn_denoise.h"

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void fail(const char *what) {
  std::fprintf(stderr, "[FAIL] %s\n", what);
  std::exit(1);
}

static void vk_check(VkResult r, const char *what) {
  if (r != VK_SUCCESS) {
    std::fprintf(stderr, "[FAIL] %s (VkResult = %d)\n", what, (int)r);
    std::exit(1);
  }
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <in.f32> <width> <height> <out.f32> [out.ppm] [--alb alb.f32] "
                 "[--nrm nrm.f32]\n", argv[0]);
    return 1;
  }
  const char *inPath = argv[1];
  const int width = std::atoi(argv[2]);
  const int height = std::atoi(argv[3]);
  const char *outPath = argv[4];
  const char *albPath = nullptr;
  const char *nrmPath = nullptr;
  const char *ppmPath = nullptr;
  for (int i = 5; i < argc; ++i) {
    if (std::strcmp(argv[i], "--alb") == 0 && i + 1 < argc) albPath = argv[++i];
    else if (std::strcmp(argv[i], "--nrm") == 0 && i + 1 < argc) nrmPath = argv[++i];
    else ppmPath = argv[i];
  }
  if (nrmPath && !albPath) fail("--nrm needs --alb (OIDN rejects color+normal without albedo)");

  std::vector<float> rgb(static_cast<size_t>(width) * height * 3);
  {
    std::ifstream f(inPath, std::ios::binary | std::ios::ate);
    if (!f) fail("cannot open the input image");
    const std::size_t size = static_cast<std::size_t>(f.tellg());
    if (size != rgb.size() * sizeof(float)) fail("input size does not match width*height*3");
    f.seekg(0);
    f.read(reinterpret_cast<char *>(rgb.data()), static_cast<std::streamsize>(size));
  }
  std::vector<float> rgba(static_cast<size_t>(width) * height * 4);
  for (std::size_t i = 0, n = static_cast<std::size_t>(width) * height; i < n; ++i) {
    rgba[i * 4 + 0] = rgb[i * 3 + 0];
    rgba[i * 4 + 1] = rgb[i * 3 + 1];
    rgba[i * 4 + 2] = rgb[i * 3 + 2];
    rgba[i * 4 + 3] = 1.0f;
  }

  // ---- Vulkan setup -------------------------------------------------------
  VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  appInfo.pApplicationName = "oidn-vulkan-denoiser";
  appInfo.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instInfo.pApplicationInfo = &appInfo;
  VkInstance instance = VK_NULL_HANDLE;
  vk_check(vkCreateInstance(&instInfo, nullptr, &instance), "vkCreateInstance");

  uint32_t deviceCount = 0;
  vk_check(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices");
  if (deviceCount == 0) fail("no Vulkan device");
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vk_check(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()),
           "vkEnumeratePhysicalDevices");

  VkPhysicalDevice physical = VK_NULL_HANDLE;
  uint32_t queueFamily = UINT32_MAX;
  int bestScore = -1;
  for (VkPhysicalDevice dev : devices) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(dev, &props);
    int score = 1;
    switch (props.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score = 4; break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 3; break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: score = 2; break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU: score = 0; break;
      default: score = 1; break;
    }
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &familyCount, families.data());
    for (uint32_t i = 0; i < familyCount; ++i) {
      if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
      if (score > bestScore) {
        bestScore = score;
        physical = dev;
        queueFamily = i;
      }
      break;
    }
  }
  if (physical == VK_NULL_HANDLE) fail("no compute queue family");

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physical, &props);
  std::printf("[info] device: %s\n", props.deviceName);

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queueInfo.queueFamilyIndex = queueFamily;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &priority;

  // The denoiser's OIDN_STORE=fp16 path stores its tensors as FP16, which needs 16-bit storage
  // buffers. The extension is optional here: if the device does not advertise it, the FP32 path
  // still works and the fp16 request fails at pipeline creation.
  VkPhysicalDevice16BitStorageFeatures supported16{};
  supported16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  VkPhysicalDeviceShaderFloat16Int8Features supportedF16{};
  supportedF16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  supported16.pNext = &supportedF16;
  VkPhysicalDeviceFeatures2 queryFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  queryFeatures.pNext = &supported16;
  vkGetPhysicalDeviceFeatures2(physical, &queryFeatures);
  VkPhysicalDevice16BitStorageFeatures enabled16{};
  enabled16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  enabled16.storageBuffer16BitAccess = supported16.storageBuffer16BitAccess;
  enabled16.uniformAndStorageBuffer16BitAccess = supported16.uniformAndStorageBuffer16BitAccess;
  VkPhysicalDeviceShaderFloat16Int8Features enabledF16{};
  enabledF16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  enabledF16.shaderFloat16 = supportedF16.shaderFloat16;
  enabled16.pNext = &enabledF16;

  VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  devInfo.queueCreateInfoCount = 1;
  devInfo.pQueueCreateInfos = &queueInfo;
  devInfo.pNext = &enabled16;
  VkDevice device = VK_NULL_HANDLE;
  vk_check(vkCreateDevice(physical, &devInfo, nullptr, &device), "vkCreateDevice");

  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(device, queueFamily, 0, &queue);

  VkCommandPoolCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpInfo.queueFamilyIndex = queueFamily;
  VkCommandPool cmdPool = VK_NULL_HANDLE;
  vk_check(vkCreateCommandPool(device, &cpInfo, nullptr, &cmdPool), "vkCreateCommandPool");

  OidnVulkanContext ctx{};
  ctx.physicalDevice = physical;
  ctx.device = device;
  ctx.queue = queue;
  ctx.commandPool = cmdPool;
  vkGetPhysicalDeviceMemoryProperties(physical, &ctx.memoryProperties);
  // FP16 tensor storage is the denoiser's default (fastest configuration), so the tool has to
  // report whether the 16-bit storage feature was enabled above.
  ctx.storage16Bit = enabled16.storageBuffer16BitAccess;

  // ---- run ----------------------------------------------------------------
  const size_t pixelCount = static_cast<size_t>(width) * height;
  std::vector<float> albedo, normal;
  auto load_rgb = [pixelCount](const char *path, std::vector<float> &dst, const char *what) {
    dst.resize(pixelCount * 3);
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) fail(what);
    const std::size_t size = static_cast<std::size_t>(f.tellg());
    if (size != dst.size() * sizeof(float)) fail(what);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(dst.data()), static_cast<std::streamsize>(size));
  };
  if (albPath) load_rgb(albPath, albedo, "cannot read the albedo image (width*height*3 floats)");
  if (nrmPath) load_rgb(nrmPath, normal, "cannot read the normal image (width*height*3 floats)");
  OidnAuxImages aux;
  aux.albedo = albPath ? albedo.data() : nullptr;
  aux.normal = nrmPath ? normal.data() : nullptr;

  std::vector<float> out;
  std::string error;
  double ms = 0.0;
  if (!oidn_denoise_image(ctx, rgba.data(), width, height, out, error, &ms,
                          (albPath || nrmPath) ? &aux : nullptr)) {
    std::fprintf(stderr, "[FAIL] %s\n", error.c_str());
    return 1;
  }
  std::printf("[info] denoised in %.1f ms\n", ms);

  std::vector<float> outRgb(static_cast<size_t>(width) * height * 3);
  for (std::size_t i = 0, n = static_cast<std::size_t>(width) * height; i < n; ++i) {
    for (int c = 0; c < 3; ++c) outRgb[i * 3 + c] = out[i * 4 + c];
  }
  {
    std::ofstream f(outPath, std::ios::binary);
    if (!f) fail("cannot write the output image");
    f.write(reinterpret_cast<const char *>(outRgb.data()),
            static_cast<std::streamsize>(outRgb.size() * sizeof(float)));
  }

  if (ppmPath) {
    std::ofstream f(ppmPath, std::ios::binary);
    if (!f) fail("cannot write the preview image");
    f << "P6\n" << width << " " << height << "\n255\n";
    for (std::size_t i = 0, n = static_cast<std::size_t>(width) * height; i < n; i += 1) {
      for (int c = 0; c < 3; ++c) {
        float v = outRgb[i * 3 + c];
        if (!(v == v) || v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        f.put(static_cast<char>(std::pow(v, 1.0f / 2.2f) * 255.0f + 0.5f));
      }
    }
  }

  vkDestroyCommandPool(device, cmdPool, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  std::printf("[info] wrote %s\n", outPath);
  return 0;
}
