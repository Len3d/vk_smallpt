// conv_probe.cpp -- run a single 3x3 convolution compute pass and compare it against a CPU
// reference computed with the same weight layout. Used to iterate on the convolution kernel
// without running the whole 30-pass chain.
//
// Usage: conv_probe.exe <conv.spv> <rt_hdr.bin> <width> <height> <cin> <cout> <weightOfs> <block>
//   weightOfs : float offset of this layer inside the weight blob
//   block     : output pixels covered by one work group in each dimension (8 for oidn_conv.spv,
//               16 for oidn_conv_tiled.spv)
// The work group always computes 8 output channels, and gl_GlobalInvocationID.z selects which
// group, so the z dispatch count is ceil(cout / 8) for both kernels.

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

static void fail(const char *what) {
  std::fprintf(stderr, "[FAIL] %s\n", what);
  std::exit(1);
}

static void vk_check(VkResult r, const char *what) {
  if (r != VK_SUCCESS) {
    std::fprintf(stderr, "[FAIL] %s (VkResult = %d)\n", what, static_cast<int>(r));
    std::exit(1);
  }
}

static std::vector<char> read_file(const char *path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) fail("cannot open file");
  const std::size_t size = static_cast<std::size_t>(f.tellg());
  std::vector<char> data(size);
  f.seekg(0);
  f.read(data.data(), static_cast<std::streamsize>(size));
  return data;
}

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void *mapped = nullptr;
  std::size_t size = 0;
};

static Buffer create_buffer(VkDevice device, const VkPhysicalDeviceMemoryProperties &memProps,
                            std::size_t size, VkBufferUsageFlags usage) {
  Buffer b;
  b.size = size;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = size;
  info.usage = usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vk_check(vkCreateBuffer(device, &info, nullptr, &b.buffer), "vkCreateBuffer");

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(device, b.buffer, &req);
  uint32_t typeIndex = UINT32_MAX;
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    const bool hostVisible = (memProps.memoryTypes[i].propertyFlags &
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    const bool hostCoherent = (memProps.memoryTypes[i].propertyFlags &
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    if ((req.memoryTypeBits & (1u << i)) && hostVisible && hostCoherent) {
      typeIndex = i;
      break;
    }
  }
  if (typeIndex == UINT32_MAX) fail("no host visible memory type");

  VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocInfo.allocationSize = req.size;
  allocInfo.memoryTypeIndex = typeIndex;
  vk_check(vkAllocateMemory(device, &allocInfo, nullptr, &b.memory), "vkAllocateMemory");
  vk_check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "vkBindBufferMemory");
  vk_check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "vkMapMemory");
  return b;
}

static void destroy_buffer(VkDevice device, Buffer &b) {
  if (b.mapped) vkUnmapMemory(device, b.memory);
  if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
  if (b.memory) vkFreeMemory(device, b.memory, nullptr);
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 9) {
    std::fprintf(stderr,
                 "usage: %s <conv.spv> <rt_hdr.bin> <width> <height> <cin> <cout> <weightOfs> "
                 "<block> [cout group] [repeat]\n",
                 argv[0]);
    return 1;
  }
  const char *spvPath = argv[1];
  const char *wgtPath = argv[2];
  const int w = std::atoi(argv[3]);
  const int h = std::atoi(argv[4]);
  const int cin = std::atoi(argv[5]);
  const int cout = std::atoi(argv[6]);
  const int weightOfs = std::atoi(argv[7]);
  const int block = std::atoi(argv[8]);
  // A register-blocked kernel can have a rectangular block (e.g. 24x16 for 3x2 pixels per thread).
  const int blockY = (argc > 12) ? std::atoi(argv[12]) : block;
  const int group = (argc > 9) ? std::atoi(argv[9]) : 8;
  const int repeat = (argc > 10) ? std::atoi(argv[10]) : 1;
  // The clspv (C++ for OpenCL) build of the pass takes its scalar arguments from a parameter
  // buffer instead of push constants, and has its own entry point name.
  const bool clcMode = (argc > 11) && std::strcmp(argv[11], "--clc") == 0;
  const bool relu = true;

  const std::size_t pixelCount = static_cast<std::size_t>(w) * h;
  const std::size_t srcFloats = pixelCount * cin;
  const std::size_t dstFloats = pixelCount * cout;
  const std::size_t wgtFloats = static_cast<std::size_t>(cout) * cin * 12 + cout;

  // ---- inputs -------------------------------------------------------------
  std::vector<float> src(srcFloats);
  for (std::size_t i = 0; i < srcFloats; ++i) {
    src[i] = std::sin(static_cast<double>(i) * 0.731f) * 0.5f;
  }
  const std::vector<char> blob = read_file(wgtPath);
  if (blob.size() < (static_cast<std::size_t>(weightOfs) + wgtFloats) * sizeof(float)) {
    fail("weight blob too small for the requested layer");
  }
  std::vector<float> wgt(wgtFloats);
  std::memcpy(wgt.data(), blob.data() + static_cast<std::size_t>(weightOfs) * sizeof(float),
              wgtFloats * sizeof(float));

  // ---- Vulkan setup -------------------------------------------------------
  VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  appInfo.pApplicationName = "oidn-conv-probe";
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
    VkPhysicalDeviceProperties devProps{};
    vkGetPhysicalDeviceProperties(dev, &devProps);
    int score = 1;
    switch (devProps.deviceType) {
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
  VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  devInfo.queueCreateInfoCount = 1;
  devInfo.pQueueCreateInfos = &queueInfo;
  VkDevice device = VK_NULL_HANDLE;
  vk_check(vkCreateDevice(physical, &devInfo, nullptr, &device), "vkCreateDevice");
  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(device, queueFamily, 0, &queue);

  VkPhysicalDeviceMemoryProperties memProps{};
  vkGetPhysicalDeviceMemoryProperties(physical, &memProps);

  VkCommandPoolCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpInfo.queueFamilyIndex = queueFamily;
  VkCommandPool cmdPool = VK_NULL_HANDLE;
  vk_check(vkCreateCommandPool(device, &cpInfo, nullptr, &cmdPool), "vkCreateCommandPool");

  const std::vector<char> code = read_file(spvPath);
  VkShaderModuleCreateInfo modInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  modInfo.codeSize = code.size();
  modInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule module = VK_NULL_HANDLE;
  vk_check(vkCreateShaderModule(device, &modInfo, nullptr, &module), "vkCreateShaderModule");

  const uint32_t bindingCount = clcMode ? 4u : 3u;
  VkDescriptorSetLayoutBinding bindings[4]{};
  for (uint32_t i = 0; i < bindingCount; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslInfo.bindingCount = bindingCount;
  dslInfo.pBindings = bindings;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  vk_check(vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &dsl), "vkCreateDescriptorSetLayout");

  VkPushConstantRange range{};
  range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  // clspv's CLC++ mode always emits a 12-byte module-scope push constant holding the
  // PushConstantRegionOffset, so the range has to cover it even though nothing is read there.
  range.size = clcMode ? 12u : static_cast<uint32_t>(6 * sizeof(int));
  VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plInfo.setLayoutCount = 1;
  plInfo.pSetLayouts = &dsl;
  plInfo.pushConstantRangeCount = 1;
  plInfo.pPushConstantRanges = &range;
  VkPipelineLayout pl = VK_NULL_HANDLE;
  vk_check(vkCreatePipelineLayout(device, &plInfo, nullptr, &pl), "vkCreatePipelineLayout");

  VkComputePipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeInfo.stage.module = module;
  pipeInfo.stage.pName = clcMode ? "oidn_conv_tiled" : "main";
  pipeInfo.layout = pl;
  VkPipeline pipeline = VK_NULL_HANDLE;
  vk_check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline),
           "vkCreateComputePipelines");

  Buffer srcBuf = create_buffer(device, memProps, srcFloats * sizeof(float),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  Buffer dstBuf = create_buffer(device, memProps, dstFloats * sizeof(float),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  Buffer wgtBuf = create_buffer(device, memProps, wgtFloats * sizeof(float),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  std::memcpy(srcBuf.mapped, src.data(), srcFloats * sizeof(float));
  std::memcpy(wgtBuf.mapped, wgt.data(), wgtFloats * sizeof(float));
  // Parameter buffer for the clspv build: the six scalar words the GLSL pass gets as push
  // constants. The remaining words are left zero.
  Buffer parBuf = create_buffer(device, memProps, 64 * sizeof(uint32_t),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  if (clcMode) {
    const uint32_t words[6] = {static_cast<uint32_t>(cin),
                               static_cast<uint32_t>(cout),
                               static_cast<uint32_t>(w),
                               static_cast<uint32_t>(h),
                               static_cast<uint32_t>(relu ? 1 : 0),
                               static_cast<uint32_t>(weightOfs)};
    std::memcpy(parBuf.mapped, words, sizeof(words));
  }
  // Sentinel so an unwritten output is visible in the comparison.
  std::memset(dstBuf.mapped, 0x5A, dstFloats * sizeof(float));

  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = bindingCount;
  VkDescriptorPoolCreateInfo dpInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpInfo.maxSets = 1;
  dpInfo.poolSizeCount = 1;
  dpInfo.pPoolSizes = &poolSize;
  VkDescriptorPool dp = VK_NULL_HANDLE;
  vk_check(vkCreateDescriptorPool(device, &dpInfo, nullptr, &dp), "vkCreateDescriptorPool");

  VkDescriptorSetAllocateInfo dsInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsInfo.descriptorPool = dp;
  dsInfo.descriptorSetCount = 1;
  dsInfo.pSetLayouts = &dsl;
  VkDescriptorSet ds = VK_NULL_HANDLE;
  vk_check(vkAllocateDescriptorSets(device, &dsInfo, &ds), "vkAllocateDescriptorSets");

  VkDescriptorBufferInfo bufInfos[4] = {{srcBuf.buffer, 0, VK_WHOLE_SIZE},
                                        {dstBuf.buffer, 0, VK_WHOLE_SIZE},
                                        {wgtBuf.buffer, 0, VK_WHOLE_SIZE},
                                        {parBuf.buffer, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet writes[4]{};
  for (uint32_t i = 0; i < bindingCount; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufInfos[i];
  }
  vkUpdateDescriptorSets(device, bindingCount, writes, 0, nullptr);

  VkCommandBufferAllocateInfo cbInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbInfo.commandPool = cmdPool;
  cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbInfo.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  vk_check(vkAllocateCommandBuffers(device, &cbInfo, &cb), "vkAllocateCommandBuffers");

  struct {
    int cin, cout, width, height, relu, weightOfs;
  } pc = {cin, cout, w, h, relu ? 1 : 0, 0};

  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vk_check(vkBeginCommandBuffer(cb, &beginInfo), "vkBeginCommandBuffer");
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
  if (clcMode) {
    const uint32_t zeros[3] = {0, 0, 0};
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(zeros), zeros);
  } else {
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  }
  const uint32_t gx = static_cast<uint32_t>((w + block - 1) / block);
  const uint32_t gy = static_cast<uint32_t>((h + blockY - 1) / blockY);
  const uint32_t gz = static_cast<uint32_t>((cout + group - 1) / group);
  std::printf("[info] dispatch %ux%ux%u (block %d, cout group %d, repeat %d)\n", gx, gy, gz, block,
              group, repeat);
  for (int r = 0; r < repeat; ++r) vkCmdDispatch(cb, gx, gy, gz);
  vk_check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");

  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  vk_check(vkCreateFence(device, &fenceInfo, nullptr, &fence), "vkCreateFence");
  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cb;
  const auto t0 = std::chrono::steady_clock::now();
  vk_check(vkQueueSubmit(queue, 1, &submitInfo, fence), "vkQueueSubmit");
  vk_check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
  std::printf("[info] kernel wall time %.3f ms per dispatch\n", ms);

  const float *out = static_cast<const float *>(dstBuf.mapped);
  float outMin = out[0], outMax = out[0];
  std::size_t sentinelCount = 0;
  for (std::size_t i = 0; i < dstFloats; ++i) {
    if (!(out[i] == out[i])) continue;
    if (out[i] < outMin) outMin = out[i];
    if (out[i] > outMax) outMax = out[i];
    const uint32_t bits = *reinterpret_cast<const uint32_t *>(&out[i]);
    if (bits == 0x5A5A5A5A) ++sentinelCount;
  }
  std::printf("[info] gpu output range [%g, %g], sentinel floats left: %zu / %zu\n", outMin, outMax,
              sentinelCount, dstFloats);

  // ---- CPU reference ------------------------------------------------------
  std::vector<float> ref(dstFloats);
  const float *wp = wgt.data();
  const float *bias = wp + static_cast<std::size_t>(cout) * cin * 12;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t p = static_cast<std::size_t>(y) * w + x;
      for (int o = 0; o < cout; ++o) {
        float acc = bias[o];
        for (int i = 0; i < cin; ++i) {
          const float *k = wp + (static_cast<std::size_t>(o) * cin + i) * 12;
          float s = 0.0f;
          for (int t = 0; t < 9; ++t) {
            const int sx = x + (t % 3) - 1;
            const int sy = y + (t / 3) - 1;
            float v = 0.0f;
            if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
              v = src[(static_cast<std::size_t>(sy) * w + sx) * cin + i];
            }
            s += k[t] * v;
          }
          acc += s;
        }
        ref[p * cout + o] = acc > 0.0f ? acc : 0.0f;
      }
    }
  }

  double sumAbs = 0.0;
  float maxAbs = 0.0f;
  double sumRef = 0.0;
  std::size_t bad = 0;
  for (std::size_t i = 0; i < dstFloats; ++i) {
    const float d = std::fabs(out[i] - ref[i]);
    sumAbs += d;
    if (std::fabs(ref[i]) > 1e-3f) sumRef += std::fabs(ref[i]);
    if (d > maxAbs) maxAbs = d;
    if (!(out[i] == out[i]) || d > 1e-3f) ++bad;
  }
  std::printf("[result] mean abs diff %.6e  max abs diff %.6e  rel %.3e  bad floats %zu / %zu\n",
              sumAbs / static_cast<double>(dstFloats), maxAbs,
              sumAbs / (sumRef > 0.0 ? sumRef : 1.0), bad, dstFloats);

  destroy_buffer(device, srcBuf);
  destroy_buffer(device, dstBuf);
  destroy_buffer(device, wgtBuf);
  vkDestroyFence(device, fence, nullptr);
  vkDestroyDescriptorPool(device, dp, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pl, nullptr);
  vkDestroyDescriptorSetLayout(device, dsl, nullptr);
  vkDestroyShaderModule(device, module, nullptr);
  vkDestroyCommandPool(device, cmdPool, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  return bad == 0 ? 0 : 2;
}
