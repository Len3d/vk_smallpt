// main.cpp -- minimalist Vulkan compute example that runs a kernel compiled by clspv.
//
// Flow (each step is marked with a comment below):
//   1. create instance / pick a physical device with a compute queue family / create device + queue
//   2. create three host-visible SSBOs as input and output buffers, fill in a and b
//   3. descriptor set layout: binding 0/1/2 = storage buffer (the three __global kernel args)
//   4. build the compute pipeline from vadd.spv
//   5. record the command buffer: bind pipeline -> bind descriptor set -> dispatch(count / 64)
//   6. submit, wait on the fence, read the result back and compare it element by element on the CPU
//
// SPIR-V path: defaults to the compile-time VADD_SPV_PATH (CMake points it at the vadd.spv
// in the build directory). The first command line argument overrides it:
//   vkcompute.exe <path-to-vadd.spv>
//
// To turn on the Vulkan validation layer, set the environment variable
// CLSPV_EXAMPLE_VALIDATION=1 before running.

#include <vulkan/vulkan.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef VADD_SPV_PATH
#define VADD_SPV_PATH "vadd.spv"
#endif

// ---- Problem size for this example ----------------------------------------
static const uint32_t kElementCount = 1u << 20; // 1M floats = 4 MB
static const uint32_t kWorkgroupSize = 64;      // matches local_size_x in the kernel

// ---- Small helpers --------------------------------------------------------

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
                                 VkMemoryRequirements req,
                                 VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((req.memoryTypeBits & (1u << i)) &&
        (memProps.memoryTypes[i].propertyFlags & want) == want) {
      return i;
    }
  }
  fail("no memory type satisfies HOST_VISIBLE | HOST_COHERENT");
}

static std::vector<char> read_spirv(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::fprintf(stderr,
                 "[FAIL] cannot open SPIR-V file: %s\n"
                 "       build it first with the clspv target (see CMakeLists.txt), "
                 "or pass the path of an existing vadd.spv as the first argument.\n",
                 path);
    std::exit(1);
  }
  const std::size_t size = static_cast<std::size_t>(file.tellg());
  std::vector<char> data(size);
  file.seekg(0);
  file.read(data.data(), static_cast<std::streamsize>(size));
  if (size == 0 || size % 4 != 0) fail("unexpected SPIR-V size (must be a multiple of 4 bytes)");
  return data;
}

// A host-visible storage buffer whose mapped pointer stays alive for the whole run
struct HostBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void *mapped = nullptr;
  std::size_t bytes = 0;
};

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

// ---- Optional validation layer --------------------------------------------
static VkDebugUtilsMessengerEXT g_messenger = VK_NULL_HANDLE;

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT *data, void * /*user*/) {
  std::fprintf(stderr, "[validation] %s: %s\n",
               (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" : "WARN",
               data->pMessage);
  return VK_FALSE;
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0); // unbuffered stdout so a crash keeps the log
  const char *spvPath = (argc > 1) ? argv[1] : VADD_SPV_PATH;
  std::vector<char> spirv = read_spirv(spvPath);
  const bool wantValidation = std::getenv("CLSPV_EXAMPLE_VALIDATION") != nullptr;

  // ---- 1. instance --------------------------------------------------------
  VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  appInfo.pApplicationName = "clspv-vulkan-compute-demo";
  appInfo.apiVersion = VK_API_VERSION_1_1;

  std::vector<const char *> instanceExts;
  std::vector<const char *> instanceLayers;

  // Attach the validation layer if it exists (probed with vkEnumerateInstanceLayerProperties;
  // a missing layer is not an error)
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

  // ---- 2. pick a device (must expose a compute queue family) --------------
  uint32_t deviceCount = 0;
  vk_check(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr), "vkEnumeratePhysicalDevices");
  if (deviceCount == 0) fail("no Vulkan device on this system");
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vk_check(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices");

  VkPhysicalDevice physical = VK_NULL_HANDLE;
  uint32_t queueFamily = UINT32_MAX;
  std::string deviceName;
  int bestTypeScore = -1;
  uint32_t bestApiVersion = 0;
  uint32_t bestQueueCount = 0;

  for (VkPhysicalDevice dev : devices) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(dev, &props);

    // Preference order: discrete > integrated > other > CPU; ties broken by Vulkan version and
    // queue count. On a machine that exposes both a vendor driver and a translation/emulation
    // device (e.g. Dozen on top of D3D12) this prefers the vendor driver.
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
      break; // only the first compute queue family of each device is considered
    }
  }
  if (physical == VK_NULL_HANDLE) fail("no physical device with a compute queue family");

  std::printf("[info] device: %s (queue family %u)\n", deviceName.c_str(), queueFamily);
  std::printf("[info] SPIR-V: %s (%zu bytes)\n", spvPath, spirv.size());

  // ---- 3. device + queue --------------------------------------------------
  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queueInfo.queueFamilyIndex = queueFamily;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &priority;

  VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  devInfo.queueCreateInfoCount = 1;
  devInfo.pQueueCreateInfos = &queueInfo;
  devInfo.enabledExtensionCount = 0; // no swapchain and no device extension is needed

  // clspv uses Vulkan 1.1 variablePointers when a kernel does pointer-to-pointer work or
  // pointer arithmetic, so the feature is probed and enabled when available:
  //   * supported -> enable it, so more complex kernels need no host change later;
  //   * unsupported -> skip it, this example's kernel has no pointer arithmetic.
  VkPhysicalDeviceVulkan11Features supported11{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features2.pNext = &supported11;
  vkGetPhysicalDeviceFeatures2(physical, &features2);

  VkPhysicalDeviceVulkan11Features enable11{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  if (supported11.variablePointersStorageBuffer == VK_TRUE) {
    enable11.variablePointersStorageBuffer = VK_TRUE;
    enable11.variablePointers = supported11.variablePointers;
    devInfo.pNext = &enable11;
    std::printf("[info] enabled variablePointersStorageBuffer%s\n",
                supported11.variablePointers ? " / variablePointers" : "");
  }

  VkDevice device = VK_NULL_HANDLE;
  vk_check(vkCreateDevice(physical, &devInfo, nullptr, &device), "vkCreateDevice");

  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(device, queueFamily, 0, &queue);

  VkPhysicalDeviceMemoryProperties memProps{};
  vkGetPhysicalDeviceMemoryProperties(physical, &memProps);

  // ---- 4. three buffers, fill the inputs ----------------------------------
  const std::size_t bytes = static_cast<std::size_t>(kElementCount) * sizeof(float);
  HostBuffer bufA = create_host_buffer(device, memProps, bytes);
  HostBuffer bufB = create_host_buffer(device, memProps, bytes);
  HostBuffer bufOut = create_host_buffer(device, memProps, bytes);

  std::vector<float> hostA(kElementCount), hostB(kElementCount), expected(kElementCount);
  for (uint32_t i = 0; i < kElementCount; ++i) {
    hostA[i] = static_cast<float>(i % 100) * 0.5f;
    hostB[i] = static_cast<float>(i % 37) * 2.0f;
    expected[i] = hostA[i] + hostB[i]; // CPU reference
  }
  std::memcpy(bufA.mapped, hostA.data(), bytes);
  std::memcpy(bufB.mapped, hostB.data(), bytes);
  std::memset(bufOut.mapped, 0, bytes);

  // ---- 5. descriptor set layout / pool / set ------------------------------
  VkDescriptorSetLayoutBinding bindings[3]{};
  for (uint32_t i = 0; i < 3; ++i) {
    bindings[i].binding = i;                                  // i-th __global kernel argument
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslInfo.bindingCount = 3;
  dslInfo.pBindings = bindings;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  vk_check(vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &dsl), "vkCreateDescriptorSetLayout");

  VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
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

  VkDescriptorBufferInfo infos[3] = {
      {bufA.buffer, 0, VK_WHOLE_SIZE},
      {bufB.buffer, 0, VK_WHOLE_SIZE},
      {bufOut.buffer, 0, VK_WHOLE_SIZE},
  };
  VkWriteDescriptorSet writes[3]{};
  for (uint32_t i = 0; i < 3; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

  // ---- 6. compute pipeline -----------------------------------------------
  VkShaderModuleCreateInfo modInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  modInfo.codeSize = spirv.size();
  modInfo.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
  VkShaderModule module = VK_NULL_HANDLE;
  vk_check(vkCreateShaderModule(device, &modInfo, nullptr, &module), "vkCreateShaderModule");

  VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plInfo.setLayoutCount = 1;
  plInfo.pSetLayouts = &dsl; // this kernel uses no push constants
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  vk_check(vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout), "vkCreatePipelineLayout");

  VkComputePipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeInfo.stage.module = module;
  pipeInfo.stage.pName = "vadd"; // clspv keeps the kernel function name as the entry point
  pipeInfo.layout = pipelineLayout;
  VkPipeline pipeline = VK_NULL_HANDLE;
  vk_check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline),
           "vkCreateComputePipelines");
  vkDestroyShaderModule(device, module, nullptr); // the module can go once the pipeline exists

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

  const uint32_t groupCount = kElementCount / kWorkgroupSize; // 1M / 64 = 16384 workgroups
  vkCmdDispatch(cmd, groupCount, 1, 1);

  // The host has to see the writes done by the device, so insert a compute -> host barrier
  VkMemoryBarrier memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                       1, &memBarrier, 0, nullptr, 0, nullptr);

  vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

  // ---- 8. submit and wait -------------------------------------------------
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

  // ---- 9. verify the result ----------------------------------------------
  const float *gpu = static_cast<const float *>(bufOut.mapped);
  uint32_t mismatch = 0;
  float maxErr = 0.0f;
  for (uint32_t i = 0; i < kElementCount; ++i) {
    const float err = std::fabs(gpu[i] - expected[i]);
    if (err > 1e-5f) {
      if (mismatch < 4) {
        std::printf("[diff] i=%u  gpu=%g  cpu=%g\n", i, gpu[i], expected[i]);
      }
      ++mismatch;
    }
    if (err > maxErr) maxErr = err;
  }

  std::printf("[info] dispatch: %u workgroups x %u threads = %u work items\n",
              groupCount, kWorkgroupSize, groupCount * kWorkgroupSize);
  std::printf("[info] elapsed: %.3f ms\n",
              std::chrono::duration<double, std::milli>(t1 - t0).count());
  std::printf("[info] first 4 elements: gpu = %g %g %g %g\n", gpu[0], gpu[1], gpu[2], gpu[3]);
  std::printf("[info] last element: gpu = %g (expected %g)\n", gpu[kElementCount - 1],
              expected[kElementCount - 1]);

  if (mismatch == 0) {
    std::printf("\nPASS: all %u elements match (max |err| = %g)\n", kElementCount, maxErr);
  } else {
    std::printf("\nFAIL: %u / %u elements differ (max |err| = %g)\n", mismatch, kElementCount, maxErr);
  }

  // ---- 10. teardown -------------------------------------------------------
  vkDestroyFence(device, fence, nullptr);
  vkDestroyCommandPool(device, cmdPool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  vkDestroyDescriptorPool(device, pool, nullptr);
  vkDestroyDescriptorSetLayout(device, dsl, nullptr);
  for (HostBuffer *b : {&bufA, &bufB, &bufOut}) {
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

  return mismatch == 0 ? 0 : 1;
}
