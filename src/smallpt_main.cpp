// smallpt_main.cpp -- runs the smallpt kernel compiled by clspv and writes the render to an image file.
//
// It uses the same minimal Vulkan compute flow as the vadd example, except:
//   * the kernel takes four arguments: binding 0 = output image buffer (4 floats per pixel),
//     binding 1 = albedo image, binding 2 = normal image (the auxiliary inputs of the denoiser),
//     binding 3 = parameter buffer;
//   * the default module build/smallpt.spv is built from src/smallpt_cpp.cl (C++ for OpenCL); the
//     OpenCL C source src/smallpt.cl is still built as build/smallpt_c.spv and renders the same
//     image, so either module works with this host;
//   * the dispatch is 2D: ceil(width/8) x ceil(height/8) workgroups, 8x8 = 64 threads each;
//   * after readback the values go through sRGB gamma and are written as PNG
//     (8-bit RGB, encoded in-process by png_writer.cpp); a .ppm output path
//     still selects the plain PPM(P6) writer, which is handy for byte-level
//     image comparison.
//
// Usage:
//   smallpt.exe                                   # compile-time defaults: 640x480 / 64 spp
//   smallpt.exe <spv>                             # replace the SPIR-V module only
//   smallpt.exe <spv> <width> <height> <spp> [out.png|out.ppm] [entryPoint]
//
// After the render the image is passed through the Vulkan OpenImageDenoise denoiser
// (src/oidn_denoise.cpp) unless SMALLPT_NO_DENOISE=1 is set. Large images are denoised in
// tiles, so any resolution works; OIDN_DENOISE_MAX_MB tunes the tile memory budget.
//
// Validation layers: set CLSPV_EXAMPLE_VALIDATION=1 first.

#include <vulkan/vulkan.h>

#include "oidn_denoise.h"
#include "png_writer.h"

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef SMALLPT_SPV_PATH
#define SMALLPT_SPV_PATH "smallpt.spv"
#endif
#ifndef SMALLPT_OUT_PATH
#define SMALLPT_OUT_PATH "smallpt.png"
#endif

static const uint32_t kGroupSize = 8; // matches reqd_work_group_size(8,8,1) in the kernel

[[noreturn]] static void fail(const char *what) {
  std::fprintf(stderr, "[FAIL] %s\n", what);
  std::exit(1);
}

static void vk_check(VkResult r, const char *what) {
  if (r != VK_SUCCESS) {
    std::fprintf(stderr, "[FAIL] %s (VkResult = %d)\n", what, (int)r);
    std::exit(1);
  }
}

static uint32_t find_memory_type(VkPhysicalDeviceMemoryProperties memProps,
                                 VkMemoryRequirements req, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((req.memoryTypeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & want) == want) {
      return i;
    }
  }
  fail("no memory type satisfies HOST_VISIBLE | HOST_COHERENT");
}

static std::vector<char> read_spirv(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::fprintf(stderr, "[FAIL] cannot open SPIR-V: %s\n(build it with the clspv target in CMakeLists.txt)\n", path);
    std::exit(1);
  }
  const std::size_t size = static_cast<std::size_t>(file.tellg());
  std::vector<char> data(size);
  file.seekg(0);
  file.read(data.data(), static_cast<std::streamsize>(size));
  if (size == 0 || size % 4 != 0) fail("unexpected SPIR-V size (must be a multiple of 4 bytes)");
  return data;
}

struct HostBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void *mapped = nullptr;
  std::size_t bytes = 0;
};

// Workgroup size discovery, done by scanning the module directly instead of hard-coding it:
//   - OpenCL C modules carry it as OpExecutionMode LocalSize (from reqd_work_group_size);
//   - modules built with clspv --cl-std=CLC++2021 (C++ for OpenCL) IGNORE reqd_work_group_size:
//     there is no LocalSize, and the workgroup size is three spec constants decorated with
//     SpecId 0/1/2 instead. The host has to pass 8/8/1 through VkSpecializationInfo, otherwise
//     every workgroup runs a single thread and only the first groupCount elements are written.
// Returns bit0 = "has LocalSize", bits 1..3 = "SpecId 0/1/2 is decorated".
static uint32_t scan_workgroup_size(const std::vector<char> &code) {
  uint32_t found = 0;
  const uint32_t *words = reinterpret_cast<const uint32_t *>(code.data());
  const std::size_t count = code.size() / sizeof(uint32_t);
  std::size_t i = 5; // skip the 5-word SPIR-V header
  while (i < count) {
    const uint32_t word = words[i];
    const uint32_t opcode = word >> 16;
    const uint32_t length = word & 0xFFFFu;
    if (length == 0 || i + length > count) break;
    if (opcode == 16 && length >= 3) {          // OpExecutionMode: <target> <mode> <literals>
      if (words[i + 2] == 17u) found |= 1u;     // LocalSize
    } else if (opcode == 71 && length >= 4) {   // OpDecorate: <target> <decoration> <literals>
      if (words[i + 2] == 1u) {                 // Decoration SpecId
        const uint32_t id = words[i + 3];
        if (id < 3u) found |= 2u << id;
      }
    }
    i += length;
  }
  return found;
}

static HostBuffer create_host_buffer(VkDevice device, VkPhysicalDeviceMemoryProperties memProps,
                                     std::size_t bytes) {
  HostBuffer out;
  out.bytes = bytes;
  VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufInfo.size = bytes;
  bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vk_check(vkCreateBuffer(device, &bufInfo, nullptr, &out.buffer), "vkCreateBuffer");

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(device, out.buffer, &req);
  VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocInfo.allocationSize = req.size;
  allocInfo.memoryTypeIndex = find_memory_type(
      memProps, req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vk_check(vkAllocateMemory(device, &allocInfo, nullptr, &out.memory), "vkAllocateMemory");
  vk_check(vkBindBufferMemory(device, out.buffer, out.memory, 0), "vkBindBufferMemory");
  vk_check(vkMapMemory(device, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped), "vkMapMemory");
  return out;
}

static VkDebugUtilsMessengerEXT g_messenger = VK_NULL_HANDLE;

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT *data, void * /*user*/) {
  std::fprintf(stderr, "[validation] %s: %s\n",
               (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" : "WARN",
               data->pMessage);
  return VK_FALSE;
}

// sRGB gamma 1/2.2 then clamp, matching the toInt() of the original smallpt
static void convert_to_rgb8(const float *rgba, uint32_t w, uint32_t h,
                            std::vector<unsigned char> &rgb8) {
  rgb8.resize(static_cast<std::size_t>(w) * h * 3);
  for (std::size_t i = 0, n = static_cast<std::size_t>(w) * h; i < n; ++i) {
    const float *p = rgba + i * 4;
    for (int c = 0; c < 3; ++c) {
      float v = p[c];
      if (!(v == v) || v < 0.0f) v = 0.0f; // NaN / negative
      if (v > 1.0f) v = 1.0f;
      rgb8[i * 3 + c] = static_cast<unsigned char>(std::pow(v, 1.0f / 2.2f) * 255.0f + 0.5f);
    }
  }
}

// PPM(P6) writer: kept because a raw PPM is the easiest thing to diff byte by byte.
static bool write_ppm(const std::string &path, const std::vector<unsigned char> &rgb8, uint32_t w,
                      uint32_t h) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out << "P6\n" << w << " " << h << "\n255\n";
  out.write(reinterpret_cast<const char *>(rgb8.data()),
            static_cast<std::streamsize>(rgb8.size()));
  return out.good();
}

// The output format follows the file extension: .ppm keeps the uncompressed
// PPM, everything else (and the default) writes PNG.
static void write_image(const std::string &path, const float *rgba, uint32_t w, uint32_t h) {
  std::vector<unsigned char> rgb8;
  convert_to_rgb8(rgba, w, h, rgb8);

  const std::size_t dot = path.find_last_of('.');
  std::string ext;
  if (dot != std::string::npos) {
    ext = path.substr(dot);
    for (std::size_t i = 0; i < ext.size(); ++i) ext[i] = static_cast<char>(std::tolower(ext[i]));
  }

  bool ok;
  const char *what;
  if (ext == ".ppm") {
    ok = write_ppm(path, rgb8, w, h);
    what = "PPM/P6";
  } else {
    ok = write_png_rgb(path, rgb8.data(), w, h);
    what = "PNG/RGB8";
  }
  if (!ok) {
    std::fprintf(stderr, "[FAIL] cannot write output file: %s\n", path.c_str());
    std::exit(1);
  }
  std::printf("[info] wrote %s (%ux%u, %s)\n", path.c_str(), w, h, what);
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  const char *spvPath = (argc > 1) ? argv[1] : SMALLPT_SPV_PATH;
  uint32_t width = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 640u;
  uint32_t height = (argc > 3) ? static_cast<uint32_t>(std::atoi(argv[3])) : 480u;
  uint32_t spp = (argc > 4) ? static_cast<uint32_t>(std::atoi(argv[4])) : 64u;
  const char *outPath = (argc > 5) ? argv[5] : SMALLPT_OUT_PATH;
  // The 7th argument is the SPIR-V entry point name: a clspv module uses the kernel function
  // name ("smallpt"), a glslc-built GLSL module uses "main". The default follows clspv.
  const char *entryName = (argc > 6) ? argv[6] : "smallpt";
  if (width == 0 || height == 0 || spp == 0) fail("width / height / spp must be greater than 0");

  std::vector<char> spirv = read_spirv(spvPath);
  const bool wantValidation = std::getenv("CLSPV_EXAMPLE_VALIDATION") != nullptr;

  // ---- 1. instance --------------------------------------------------------
  VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  appInfo.pApplicationName = "clspv-smallpt";
  appInfo.apiVersion = VK_API_VERSION_1_1;

  std::vector<const char *> instanceExts, instanceLayers;
  if (wantValidation) {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> layers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
    for (const auto &l : layers) {
      if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
        instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
        break;
      }
    }
    if (instanceLayers.empty()) {
      std::fprintf(stderr, "[warn] VK_LAYER_KHRONOS_validation not found, skipping validation\n");
    } else {
      instanceExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
  }

  VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instInfo.pApplicationInfo = &appInfo;
  instInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExts.size());
  instInfo.ppEnabledExtensionNames = instanceExts.empty() ? nullptr : instanceExts.data();
  instInfo.enabledLayerCount = static_cast<uint32_t>(instanceLayers.size());
  instInfo.ppEnabledLayerNames = instanceLayers.empty() ? nullptr : instanceLayers.data();

  VkInstance instance = VK_NULL_HANDLE;
  vk_check(vkCreateInstance(&instInfo, nullptr, &instance), "vkCreateInstance");

  if (!instanceLayers.empty()) {
    auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (createMessenger) {
      VkDebugUtilsMessengerCreateInfoEXT ci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      ci.pfnUserCallback = debug_callback;
      createMessenger(instance, &ci, nullptr, &g_messenger);
    }
  }

  // ---- 2. device (discrete > integrated > other > CPU, then Vulkan version) --
  uint32_t deviceCount = 0;
  vk_check(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices");
  if (deviceCount == 0) fail("no Vulkan device on this system");
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vk_check(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices");

  VkPhysicalDevice physical = VK_NULL_HANDLE;
  uint32_t queueFamily = UINT32_MAX;
  std::string deviceName;
  int bestTypeScore = -1;
  uint32_t bestApiVersion = 0, bestQueueCount = 0;

  for (VkPhysicalDevice dev : devices) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(dev, &props);
    int typeScore = 1;
    switch (props.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: typeScore = 4; break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: typeScore = 3; break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: typeScore = 2; break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU: typeScore = 0; break;
      default: typeScore = 1; break;
    }
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &familyCount, families.data());
    for (uint32_t i = 0; i < familyCount; ++i) {
      if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
      const bool better =
          typeScore > bestTypeScore ||
          (typeScore == bestTypeScore &&
           (props.apiVersion > bestApiVersion ||
            (props.apiVersion == bestApiVersion && families[i].queueCount > bestQueueCount)));
      if (better) {
        bestTypeScore = typeScore;
        bestApiVersion = props.apiVersion;
        bestQueueCount = families[i].queueCount;
        physical = dev;
        queueFamily = i;
        deviceName = props.deviceName;
      }
      break;
    }
  }
  if (physical == VK_NULL_HANDLE) fail("no physical device with a compute queue family");

  std::printf("[info] device: %s (queue family %u)\n", deviceName.c_str(), queueFamily);
  std::printf("[info] SPIR-V: %s (%zu bytes)\n", spvPath, spirv.size());
  std::printf("[info] resolution: %ux%u, 2x2 subsamples x %u spp = %u paths/pixel\n", width, height,
              spp, 4 * spp);

  // ---- 3. device + queue --------------------------------------------------
  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queueInfo.queueFamilyIndex = queueFamily;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &priority;

  // A clspv module declares the SPIR-V Capability Int64 (needed for 64-bit byte addressing of
  // storage buffers), which maps to VkPhysicalDeviceFeatures::shaderInt64 and must be enabled
  // explicitly, otherwise the validation layer reports VUID-VkShaderModuleCreateInfo-pCode-08740.
  VkPhysicalDeviceFeatures supported{};
  vkGetPhysicalDeviceFeatures(physical, &supported);
  VkPhysicalDeviceFeatures enabled{};
  if (supported.shaderInt64) {
    enabled.shaderInt64 = VK_TRUE;
  } else {
    std::fprintf(stderr, "[warn] device has no shaderInt64, the clspv module may fail to load\n");
  }

  // The denoiser can store its tensors as FP16 (OIDN_STORE=fp16), which needs 16-bit storage
  // buffers. Enable it whenever the device advertises it; the FP32 path is unaffected.
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
  if (enabled16.storageBuffer16BitAccess)
    std::printf("[info] denoiser FP16 tensor storage available\n");

  VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  devInfo.queueCreateInfoCount = 1;
  devInfo.pQueueCreateInfos = &queueInfo;
  devInfo.pEnabledFeatures = &enabled;
  devInfo.pNext = &enabled16;
  VkDevice device = VK_NULL_HANDLE;
  vk_check(vkCreateDevice(physical, &devInfo, nullptr, &device), "vkCreateDevice");

  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(device, queueFamily, 0, &queue);
  VkPhysicalDeviceMemoryProperties memProps{};
  vkGetPhysicalDeviceMemoryProperties(physical, &memProps);

  // ---- 4. buffers: output image + parameters ------------------------------
  const std::size_t imageBytes = static_cast<std::size_t>(width) * height * 4 * sizeof(float);
  HostBuffer imgBuf = create_host_buffer(device, memProps, imageBytes);
  // Auxiliary images (albedo / normal) for the denoiser: 3 floats per pixel each
  const std::size_t auxBytes = static_cast<std::size_t>(width) * height * 3 * sizeof(float);
  HostBuffer albBuf = create_host_buffer(device, memProps, auxBytes);
  HostBuffer nrmBuf = create_host_buffer(device, memProps, auxBytes);
  // Parameter buffer: the first 4 uints are w/h/spp/frame, followed by the scene data (75 floats)
  // used by smallpt_buf.cl, the variant that reads the scene from a buffer. smallpt.cl and
  // smallpt.comp embed the scene in shader constants and ignore those 75 floats (the extra
  // 300 bytes are negligible).
  static const float kScene[75] = {
      // 6 planes: nx,ny,nz,d
      1.0f, 0.0f, 0.0f, 1.0f,
      -1.0f, 0.0f, 0.0f, -99.0f,
      0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, -1.0f, -170.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, -1.0f, 0.0f, -81.6f,
      // albedo of the 6 planes
      0.75f, 0.25f, 0.25f,
      0.25f, 0.25f, 0.75f,
      0.75f, 0.75f, 0.75f,
      0.0f, 0.0f, 0.0f,
      0.75f, 0.75f, 0.75f,
      0.75f, 0.75f, 0.75f,
      // 3 spheres: px,py,pz,radius
      27.0f, 16.5f, 47.0f, 16.5f,
      73.0f, 16.5f, 78.0f, 16.5f,
      50.0f, 681.6f - 0.27f, 81.6f, 600.0f,
      // albedo of the 3 spheres
      0.999f, 0.999f, 0.999f,
      0.999f, 0.999f, 0.999f,
      0.0f, 0.0f, 0.0f,
      // emission of the 3 spheres
      0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f,
      12.0f, 12.0f, 12.0f,
      // material of the 3 spheres
      1.0f, 2.0f, 0.0f,
  };
  const std::size_t paramBytes = 4 * sizeof(uint32_t) + sizeof(kScene);
  HostBuffer paramBuf = create_host_buffer(device, memProps, paramBytes);

  std::memset(imgBuf.mapped, 0, imageBytes);
  std::memset(albBuf.mapped, 0, auxBytes);
  std::memset(nrmBuf.mapped, 0, auxBytes);
  uint32_t *params = static_cast<uint32_t *>(paramBuf.mapped);
  params[0] = width;
  params[1] = height;
  params[2] = spp;
  params[3] = 0; // frame (single frame here, reserved for progressive accumulation)
  std::memcpy(reinterpret_cast<char *>(paramBuf.mapped) + 4 * sizeof(uint32_t), kScene, sizeof(kScene));

  std::printf("[info] buffers: image %.2f MiB + params %zu B = %.2f MiB (identical for both compilers)\n",
              static_cast<double>(imageBytes) / (1024.0 * 1024.0), paramBuf.bytes,
              static_cast<double>(imageBytes + paramBuf.bytes) / (1024.0 * 1024.0));
  std::printf("[info] SPIR-V module: %.1f KiB\n", static_cast<double>(spirv.size()) / 1024.0);

  // Optional: report how much device memory is really allocated (VK_EXT_memory_budget)
  struct MemBudget {
    bool ok = false;
    uint64_t heapSize = 0, heapUsage = 0, heapBudget = 0;
  } budgetInfo;
  {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &extCount, exts.data());
    for (const auto &e : exts) {
      if (std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
        budgetInfo.ok = true;
        break;
      }
    }
    if (budgetInfo.ok) {
      VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
      VkPhysicalDeviceMemoryProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
      props2.pNext = &budget;
      auto getProps2 = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
          vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties2"));
      if (getProps2) {
        getProps2(physical, &props2);
        // Take the largest device-local heap as the reference for memory usage
        for (uint32_t i = 0; i < props2.memoryProperties.memoryHeapCount; ++i) {
          const VkMemoryHeap &heap = props2.memoryProperties.memoryHeaps[i];
          if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
          if (heap.size > budgetInfo.heapSize) {
            budgetInfo.heapSize = heap.size;
            budgetInfo.heapUsage = budget.heapUsage[i];
            budgetInfo.heapBudget = budget.heapBudget[i];
          }
        }
      }
    }
  }

  // ---- 5. descriptor set layout / pool / set ------------------------------
  // binding 0 = image, 1 = albedo, 2 = normal, 3 = parameters
  VkDescriptorSetLayoutBinding bindings[4]{};
  for (uint32_t i = 0; i < 4; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslInfo.bindingCount = 4;
  dslInfo.pBindings = bindings;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  vk_check(vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &dsl), "vkCreateDescriptorSetLayout");

  VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
  VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets = 1;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  vk_check(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");

  VkDescriptorSetAllocateInfo allocSet{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocSet.descriptorPool = pool;
  allocSet.descriptorSetCount = 1;
  allocSet.pSetLayouts = &dsl;
  VkDescriptorSet set = VK_NULL_HANDLE;
  vk_check(vkAllocateDescriptorSets(device, &allocSet, &set), "vkAllocateDescriptorSets");

  VkDescriptorBufferInfo infos[4] = {{imgBuf.buffer, 0, VK_WHOLE_SIZE},
                                     {albBuf.buffer, 0, VK_WHOLE_SIZE},
                                     {nrmBuf.buffer, 0, VK_WHOLE_SIZE},
                                     {paramBuf.buffer, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet writes[4]{};
  for (uint32_t i = 0; i < 4; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);

  // ---- 6. compute pipeline -----------------------------------------------
  VkShaderModuleCreateInfo modInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  modInfo.codeSize = spirv.size();
  modInfo.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
  VkShaderModule module = VK_NULL_HANDLE;
  vk_check(vkCreateShaderModule(device, &modInfo, nullptr, &module), "vkCreateShaderModule");

  VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plInfo.setLayoutCount = 1;
  plInfo.pSetLayouts = &dsl;

  // C++ for OpenCL (and OpenCL 2.0) semantics add the OpenCL "global offset" to get_global_id, and
  // clspv reads that offset from a 12-byte push constant block (clspv-reflection prints it as
  // `pushconstant,name,region_offset,offset,0,size,12`). This example always dispatches with offset
  // 0, so the block is declared here and filled with zeros before the dispatch. Modules built with
  // --cl-std=CL1.2 simply never touch it.
  VkPushConstantRange pcRange{};
  pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcRange.offset = 0;
  pcRange.size = 12;
  plInfo.pushConstantRangeCount = 1;
  plInfo.pPushConstantRanges = &pcRange;

  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  vk_check(vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout), "vkCreatePipelineLayout");

  VkComputePipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeInfo.stage.module = module;
  pipeInfo.stage.pName = entryName; // clspv: "smallpt"; glslc-built GLSL module: "main"
  pipeInfo.layout = pipelineLayout;

  // Workgroup size. When the module has no OpExecutionMode LocalSize it declares the size as
  // spec constants 0/1/2 (clspv's C++ for OpenCL path ignores reqd_work_group_size), so the 8/8/1
  // that the dispatch assumes has to be supplied here.
  const uint32_t wgScan = scan_workgroup_size(spirv);
  uint32_t wgValues[3] = {8u, 8u, 1u};
  VkSpecializationMapEntry wgEntries[3]{};
  VkSpecializationInfo wgInfo{};
  const bool useSpec = (wgScan & 1u) == 0u;
  if (useSpec) {
    uint32_t used = 0;
    for (uint32_t id = 0; id < 3u; ++id) {
      if ((wgScan & (2u << id)) == 0u) continue; // this SpecId is not used by the module
      wgEntries[used].constantID = id;
      wgEntries[used].offset = used * static_cast<uint32_t>(sizeof(uint32_t));
      wgEntries[used].size = sizeof(uint32_t);
      wgValues[used] = wgValues[id];
      ++used;
    }
    if (used > 0) {
      wgInfo.mapEntryCount = used;
      wgInfo.pMapEntries = wgEntries;
      wgInfo.dataSize = used * static_cast<uint32_t>(sizeof(uint32_t));
      wgInfo.pData = wgValues;
      pipeInfo.stage.pSpecializationInfo = &wgInfo;
      std::printf("[info] workgroup size: spec constants 0/1/2 = 8 8 1 (module has no LocalSize)\n");
    } else {
      // spirv-opt --freeze-spec-const already turned the size into a plain constant WorkgroupSize
      // builtin, so the module itself carries 8/8/1 and nothing has to be supplied here.
      std::printf("[info] workgroup size: constant WorkgroupSize builtin = 8 8 1 (no spec constant left)\n");
    }
  } else {
    std::printf("[info] workgroup size: fixed by the module (OpExecutionMode LocalSize)\n");
  }
  VkPipeline pipeline = VK_NULL_HANDLE;
  vk_check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline),
           "vkCreateComputePipelines");
  vkDestroyShaderModule(device, module, nullptr);

  // ---- 7. command buffer --------------------------------------------------
  VkCommandPoolCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpInfo.queueFamilyIndex = queueFamily;
  VkCommandPool cmdPool = VK_NULL_HANDLE;
  vk_check(vkCreateCommandPool(device, &cpInfo, nullptr, &cmdPool), "vkCreateCommandPool");

  VkCommandBufferAllocateInfo cbInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbInfo.commandPool = cmdPool;
  cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbInfo.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vk_check(vkAllocateCommandBuffers(device, &cbInfo, &cmd), "vkAllocateCommandBuffers");

  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk_check(vkBeginCommandBuffer(cmd, &beginInfo), "vkBeginCommandBuffer");
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);

  // Global offset = 0 (see the push constant range above). Ignored by CL1.2 modules.
  const uint32_t regionOffset[3] = {0u, 0u, 0u};
  vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     static_cast<uint32_t>(sizeof(regionOffset)), regionOffset);

  const uint32_t gx = (width + kGroupSize - 1) / kGroupSize;
  const uint32_t gy = (height + kGroupSize - 1) / kGroupSize;
  vkCmdDispatch(cmd, gx, gy, 1);

  VkMemoryBarrier memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                       &memBarrier, 0, nullptr, 0, nullptr);
  vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

  // ---- 8. submit ---------------------------------------------------------
  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  vk_check(vkCreateFence(device, &fenceInfo, nullptr, &fence), "vkCreateFence");
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;

  const auto t0 = std::chrono::high_resolution_clock::now();
  vk_check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
  vk_check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // ---- 9. write the image ------------------------------------------------
  // The OpenImageDenoise U-Net runs right after the path tracer, on the same device and queue.
  // It processes large images in 16-aligned tiles with a 96 pixel overlap (core/unet.cpp), so
  // any resolution works; SMALLPT_NO_DENOISE=1 switches it off.
  const float *raw = static_cast<const float *>(imgBuf.mapped);
  std::vector<float> denoised;
  const float *finalRgba = raw;
  if (std::getenv("SMALLPT_NO_DENOISE") == nullptr) {
    OidnVulkanContext octx;
    octx.physicalDevice = physical;
    octx.device = device;
    octx.queue = queue;
    octx.commandPool = cmdPool;
    octx.memoryProperties = memProps;
    // The denoiser defaults to FP16 tensor storage, which is the fastest configuration, so it has
    // to be told whether the 16-bit storage feature was actually enabled above.
    octx.storage16Bit = enabled16.storageBuffer16BitAccess;
    std::string denoiseError;
    double denoiseMs = 0.0;
    // The albedo / normal images the kernel wrote select the weight set: rt_hdr_alb_nrm when both
    // are used, rt_hdr_alb for albedo only, rt_hdr when SMALLPT_NO_AUX=1 (the A/B baseline).
    OidnAuxImages aux;
    const bool useAux = std::getenv("SMALLPT_NO_AUX") == nullptr;
    aux.albedo = useAux ? static_cast<const float *>(albBuf.mapped) : nullptr;
    aux.normal = useAux ? static_cast<const float *>(nrmBuf.mapped) : nullptr;
    if (oidn_denoise_image(octx, raw, static_cast<int>(width), static_cast<int>(height), denoised,
                           denoiseError, &denoiseMs, useAux ? &aux : nullptr)) {
      finalRgba = denoised.data();
      std::printf("[info] OIDN denoiser: %.1f ms (%s weights)\n", denoiseMs,
                  useAux ? "rt_hdr_alb_nrm" : "rt_hdr");
    } else {
      std::fprintf(stderr, "[warn] denoiser skipped: %s\n", denoiseError.c_str());
    }
  }

  write_image(outPath, finalRgba, width, height);

  // Debug hook: SMALLPT_DUMP_RGBA=<path> writes the final (denoised) linear RGB as raw float32.
  if (const char *rgbaPath = std::getenv("SMALLPT_DUMP_RGBA")) {
    std::FILE *f = std::fopen(rgbaPath, "wb");
    if (f) {
      for (std::size_t i = 0, n = static_cast<std::size_t>(width) * height; i < n; ++i)
        std::fwrite(finalRgba + i * 4, sizeof(float), 3, f);
      std::fclose(f);
      std::printf("[info] wrote %s\n", rgbaPath);
    }
  }

  // Debug hook: SMALLPT_DUMP_AUX=<prefix> writes the albedo / normal auxiliary images as raw
  // float32, which is how they can be inspected without a second render pass.
  if (const char *auxPrefix = std::getenv("SMALLPT_DUMP_AUX")) {    for (const auto &item : {std::make_pair("alb", &albBuf), std::make_pair("nrm", &nrmBuf)}) {
      const std::string path = std::string(auxPrefix) + "_" + item.first + ".f32";
      std::FILE *f = std::fopen(path.c_str(), "wb");
      if (!f) continue;
      std::fwrite(item.second->mapped, sizeof(float), static_cast<std::size_t>(width) * height * 3, f);
      std::fclose(f);
      std::printf("[info] wrote %s\n", path.c_str());
    }
  }

  const double paths = static_cast<double>(width) * height * 4.0 * spp;
  std::printf("[info] dispatch: %u x %u workgroups x 64 threads = %u pixel threads\n", gx, gy, gx * gy * 64);
  std::printf("[info] elapsed: %.1f ms (%.2f Mpaths, %.2f Mpaths/s)\n", ms, paths / 1e6,
              paths / (ms * 1e3));
  std::printf("[info] top-left pixel (linear RGB): %.4f %.4f %.4f\n", finalRgba[0], finalRgba[1],
              finalRgba[2]);
  if (budgetInfo.heapSize != 0) {
    std::printf("[info] device memory (largest device-local heap): used %.2f MiB / budget %.2f MiB / size %.2f MiB\n",
                static_cast<double>(budgetInfo.heapUsage) / (1024.0 * 1024.0),
                static_cast<double>(budgetInfo.heapBudget) / (1024.0 * 1024.0),
                static_cast<double>(budgetInfo.heapSize) / (1024.0 * 1024.0));
  }

  // ---- 10. teardown ------------------------------------------------------
  vkDestroyFence(device, fence, nullptr);
  vkDestroyCommandPool(device, cmdPool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  vkDestroyDescriptorPool(device, pool, nullptr);
  vkDestroyDescriptorSetLayout(device, dsl, nullptr);
  for (HostBuffer *b : {&imgBuf, &albBuf, &nrmBuf, &paramBuf}) {
    vkUnmapMemory(device, b->memory);
    vkDestroyBuffer(device, b->buffer, nullptr);
    vkFreeMemory(device, b->memory, nullptr);
  }
  vkDestroyDevice(device, nullptr);

  if (g_messenger) {
    auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroyMessenger) destroyMessenger(instance, g_messenger, nullptr);
  }
  vkDestroyInstance(instance, nullptr);
  return 0;
}
