// viewer_main.cpp -- interactive front end for the smallpt kernel: Win32 window, Vulkan swapchain,
// ImGui overlay, and the same clspv-compiled path tracer used by the CLI renderer.
//
// Flow per frame:
//   1. one accumulate dispatch (src/smallpt_view.cl) adds one jittered sample per pixel to the
//      running mean, unless accumulation is paused or the camera just moved (then it restarts);
//   2. a resolve dispatch (src/smallpt_view_resolve.cl) applies exposure, Reinhard tone mapping and
//      gamma into a packed RGBA8 buffer;
//   3. the buffer is copied into a sampled image which the ImGui background draw list shows, and the
//      control window is drawn on top in the same render pass.
//
// Mouse: left drag orbits, right drag pans, wheel zooms -- the camera is a position/target pair, so
// the kernel needs no rebuild; the sample index is simply reset to 0 on every change.
//
// Options:
//   --frames N        render N frames and exit (used for automated checks)
//   --width W --height H   initial window size
//   --save path       write the viewport to a PNG when exiting (or after --frames)
//   --samples N       stop accumulating at N samples per pixel (default 4096)
//
// Validation layers: set CLSPV_EXAMPLE_VALIDATION=1.

#define VK_USE_PLATFORM_WIN32_KHR
#define NOMINMAX
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <vulkan/vulkan.h>
#include <windows.h>

#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_vulkan.h>

#include "oidn_denoise.h"
#include "png_writer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef SMALLPT_VIEW_ACCUM_SPV
#define SMALLPT_VIEW_ACCUM_SPV "smallpt_view.spv"
#endif
#ifndef SMALLPT_VIEW_RESOLVE_SPV
#define SMALLPT_VIEW_RESOLVE_SPV "smallpt_view_resolve.spv"
#endif

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam,
                                                            LPARAM lParam);

// ---------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------

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

static std::vector<char> read_binary(const char *path) {
  std::FILE *f = std::fopen(path, "rb");
  if (!f) fail(path);
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<char> data(static_cast<std::size_t>(size));
  if (size > 0 && std::fread(data.data(), 1, static_cast<std::size_t>(size), f) !=
                      static_cast<std::size_t>(size))
    fail(path);
  std::fclose(f);
  return data;
}

static uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties &mp, uint32_t bits,
                                VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  }
  fail("no memory type satisfies the request");
}

struct GpuBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void *mapped = nullptr;
  std::size_t bytes = 0;
};

// ---------------------------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------------------------

struct App {
  HWND hwnd = nullptr;
  HINSTANCE hinst = nullptr;
  bool running = true;
  bool resized = false;

  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t family = 0;
  VkPhysicalDeviceMemoryProperties memProps{};
  bool storage16Bit = false;

  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkExtent2D extent{};
  std::vector<VkImage> images;
  std::vector<VkImageView> views;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  std::vector<VkFramebuffer> framebuffers;

  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkSemaphore semImage = VK_NULL_HANDLE;
  VkSemaphore semRender = VK_NULL_HANDLE;

  VkDescriptorPool descPool = VK_NULL_HANDLE;
  VkDescriptorSetLayout layoutAcc = VK_NULL_HANDLE;
  VkDescriptorSetLayout layoutRes = VK_NULL_HANDLE;
  VkDescriptorSet setAcc = VK_NULL_HANDLE;
  VkDescriptorSet setRes = VK_NULL_HANDLE;
  VkPipelineLayout pipeLayoutAcc = VK_NULL_HANDLE;
  VkPipelineLayout pipeLayoutRes = VK_NULL_HANDLE;
  VkPipeline pipeAcc = VK_NULL_HANDLE;
  VkPipeline pipeRes = VK_NULL_HANDLE;
  VkShaderModule shaderAcc = VK_NULL_HANDLE;
  VkShaderModule shaderRes = VK_NULL_HANDLE;

  GpuBuffer accBuf{};
  GpuBuffer albBuf{};
  GpuBuffer nrmBuf{};
  GpuBuffer paramBuf{};
  GpuBuffer outBuf{};

  VkImage viewImage = VK_NULL_HANDLE;
  VkDeviceMemory viewImageMemory = VK_NULL_HANDLE;
  VkImageView viewImageView = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;
  VkDescriptorSet viewTexture = VK_NULL_HANDLE;
  bool viewImageReadLayout = false;

  uint32_t imageIndex = 0;
  uint32_t viewW = 0, viewH = 0;

  // camera (room centre by default, the classic smallpt framing: the eye sits just inside the box)
  float target[3] = {50.0f, 40.0f, 85.0f};
  float distance = 72.0f;
  float yaw = 0.0f, pitch = 0.17f;
  float fovTan = 0.5135f;
  float exposure = 1.0f;

  uint32_t sampleIndex = 0;
  uint32_t maxSamples = 4096;
  bool accumulate = true;
  bool wantDenoise = false;
  bool wantSave = false;
  bool denoisedView = false;
  bool runDenoiseOnExit = false;
  bool wantReset = false;

  double lastDenoiseMs = 0.0;
  std::string status;
  std::string status2;

  // options
  int framesToRun = -1;
  std::string savePath;
  int windowW = 1280, windowH = 800;
};

static App g;

static void camera_position(const App &a, float out[3]) {
  const float cp = std::cos(a.pitch), sp = std::sin(a.pitch);
  out[0] = a.target[0] + a.distance * (std::sin(a.yaw) * cp);
  out[1] = a.target[1] + a.distance * sp;
  out[2] = a.target[2] + a.distance * (std::cos(a.yaw) * cp);
}

static void reset_samples() {
  g.sampleIndex = 0;
  g.denoisedView = false;
  g.status2.clear();
}

static void reset_camera() {
  g.target[0] = 50.0f;
  g.target[1] = 40.0f;
  g.target[2] = 85.0f;
  g.distance = 72.0f;
  g.yaw = 0.0f;
  g.pitch = 0.17f;
  g.fovTan = 0.5135f;
  reset_samples();
}

// ---------------------------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------------------------

static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
  // Keyboard shortcuts are handled here rather than through ImGui so they work no
  // matter which widget has focus.
  if (msg == WM_KEYDOWN) {
    switch (wparam) {
    case VK_SPACE:
      g.accumulate = !g.accumulate;
      return 0;
    case 'R':
      g.wantReset = true;
      return 0;
    case 'D':
      g.wantDenoise = true;
      return 0;
    case 'S':
      g.wantSave = true;
      return 0;
    }
  }
  if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam)) return 1;
  switch (msg) {
  case WM_SIZE:
    if (wparam != SIZE_MINIMIZED) g.resized = true;
    return 0;
  case WM_SYSCOMMAND:
    if ((wparam & 0xfff0) == SC_KEYMENU) return 0; // no alt-menu
    break;
  case WM_CLOSE:
    g.running = false;
    return 0;
  case WM_DESTROY:
    PostQuitMessage(0);
    return 0;
  }
  return DefWindowProc(hwnd, msg, wparam, lparam);
}

static void create_window() {
  ImGui_ImplWin32_EnableDpiAwareness();

  g.hinst = GetModuleHandle(nullptr);
  WNDCLASSEX wc{};
  wc.cbSize = sizeof(wc);
  wc.style = CS_OWNDC;
  wc.lpfnWndProc = wnd_proc;
  wc.hInstance = g.hinst;
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wc.lpszClassName = L"vk_smallpt_viewer";
  RegisterClassEx(&wc);

  RECT rect{0, 0, g.windowW, g.windowH};
  AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
  g.hwnd = CreateWindowEx(0, wc.lpszClassName, L"vk_smallpt -- clspv path tracer + OIDN",
                          WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left,
                          rect.bottom - rect.top, nullptr, nullptr, g.hinst, nullptr);
  if (!g.hwnd) fail("CreateWindowEx");
  ShowWindow(g.hwnd, SW_SHOW);

  // The executable is a console app so that the [info] lines stay visible; park the
  // console in the taskbar instead of letting it cover the viewport.
  if (HWND console = GetConsoleWindow()) ShowWindow(console, SW_MINIMIZE);
}

// ---------------------------------------------------------------------------------------------
// Vulkan setup
// ---------------------------------------------------------------------------------------------

static void create_instance() {
  VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  appInfo.pApplicationName = "vk_smallpt viewer";
  appInfo.apiVersion = VK_API_VERSION_1_2;

  std::vector<const char *> layers;
  if (std::getenv("CLSPV_EXAMPLE_VALIDATION")) layers.push_back("VK_LAYER_KHRONOS_validation");

  std::vector<const char *> exts{VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};

  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ci.pApplicationInfo = &appInfo;
  ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
  ci.ppEnabledExtensionNames = exts.data();
  ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
  ci.ppEnabledLayerNames = layers.data();
  vk_check(vkCreateInstance(&ci, nullptr, &g.instance), "vkCreateInstance");

  VkWin32SurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
  sci.hinstance = g.hinst;
  sci.hwnd = g.hwnd;
  vk_check(vkCreateWin32SurfaceKHR(g.instance, &sci, nullptr, &g.surface), "vkCreateWin32SurfaceKHR");
}

static void pick_device() {
  uint32_t count = 0;
  vk_check(vkEnumeratePhysicalDevices(g.instance, &count, nullptr), "vkEnumeratePhysicalDevices");
  if (count == 0) fail("no Vulkan device");
  std::vector<VkPhysicalDevice> devices(count);
  vk_check(vkEnumeratePhysicalDevices(g.instance, &count, devices.data()), "vkEnumeratePhysicalDevices");

  for (VkPhysicalDevice dev : devices) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &n, fams.data());
    for (uint32_t i = 0; i < n; ++i) {
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, g.surface, &present);
      const VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
      if ((fams[i].queueFlags & need) == need && present) {
        g.physical = dev;
        g.family = i;
        break;
      }
    }
    if (g.physical) break;
  }
  if (!g.physical) fail("no queue family with graphics + compute + present");

  vkGetPhysicalDeviceMemoryProperties(g.physical, &g.memProps);

  VkPhysicalDevice16BitStorageFeatures storage{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  query.pNext = &storage;
  vkGetPhysicalDeviceFeatures2(g.physical, &query);
  g.storage16Bit = storage.storageBuffer16BitAccess != 0;

  VkPhysicalDevice16BitStorageFeatures enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  enable.storageBuffer16BitAccess = g.storage16Bit ? VK_TRUE : VK_FALSE;

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = g.family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;

  std::vector<const char *> exts{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = g.storage16Bit ? &enable : nullptr;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
  dci.ppEnabledExtensionNames = exts.data();
  vk_check(vkCreateDevice(g.physical, &dci, nullptr, &g.device), "vkCreateDevice");
  vkGetDeviceQueue(g.device, g.family, 0, &g.queue);
}

static void create_swapchain() {
  VkSurfaceCapabilitiesKHR caps{};
  vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g.physical, g.surface, &caps),
           "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

  uint32_t formatCount = 0;
  vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(g.physical, g.surface, &formatCount, nullptr),
           "vkGetPhysicalDeviceSurfaceFormatsKHR");
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(g.physical, g.surface, &formatCount, formats.data()),
           "vkGetPhysicalDeviceSurfaceFormatsKHR");

  VkSurfaceFormatKHR chosen = formats[0];
  for (const auto &f : formats) {
    if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) {
      chosen = f;
      break;
    }
  }
  g.format = chosen.format;

  VkExtent2D ext = caps.currentExtent;
  if (ext.width == 0xFFFFFFFFu) {
    RECT rc{};
    GetClientRect(g.hwnd, &rc);
    ext.width = static_cast<uint32_t>(std::max(1L, rc.right - rc.left));
    ext.height = static_cast<uint32_t>(std::max(1L, rc.bottom - rc.top));
    ext.width = std::max(caps.minImageExtent.width, std::min(caps.maxImageExtent.width, ext.width));
    ext.height = std::max(caps.minImageExtent.height, std::min(caps.maxImageExtent.height, ext.height));
  }
  g.extent = ext;

  uint32_t imageCount = caps.minImageCount + 1;
  if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) imageCount = caps.maxImageCount;

  VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  sci.surface = g.surface;
  sci.minImageCount = imageCount;
  sci.imageFormat = chosen.format;
  sci.imageColorSpace = chosen.colorSpace;
  sci.imageExtent = ext;
  sci.imageArrayLayers = 1;
  sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  sci.preTransform = caps.currentTransform;
  sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  sci.clipped = VK_TRUE;
  vk_check(vkCreateSwapchainKHR(g.device, &sci, nullptr, &g.swapchain), "vkCreateSwapchainKHR");

  uint32_t n = 0;
  vk_check(vkGetSwapchainImagesKHR(g.device, g.swapchain, &n, nullptr), "vkGetSwapchainImagesKHR");
  g.images.resize(n);
  vk_check(vkGetSwapchainImagesKHR(g.device, g.swapchain, &n, g.images.data()),
           "vkGetSwapchainImagesKHR");

  g.views.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = g.images[i];
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = g.format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vk_check(vkCreateImageView(g.device, &vci, nullptr, &g.views[i]), "vkCreateImageView");
  }
}

static void create_render_pass() {
  VkAttachmentDescription color{};
  color.format = g.format;
  color.samples = VK_SAMPLE_COUNT_1_BIT;
  color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sub{};
  sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  sub.colorAttachmentCount = 1;
  sub.pColorAttachments = &ref;

  VkSubpassDependency dep{};
  dep.srcSubpass = VK_SUBPASS_EXTERNAL;
  dep.dstSubpass = 0;
  dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.srcAccessMask = 0;
  dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpci.attachmentCount = 1;
  rpci.pAttachments = &color;
  rpci.subpassCount = 1;
  rpci.pSubpasses = &sub;
  rpci.dependencyCount = 1;
  rpci.pDependencies = &dep;
  vk_check(vkCreateRenderPass(g.device, &rpci, nullptr, &g.renderPass), "vkCreateRenderPass");
}

static void create_framebuffers() {
  g.framebuffers.resize(g.views.size());
  for (std::size_t i = 0; i < g.views.size(); ++i) {
    VkImageView att[] = {g.views[i]};
    VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fci.renderPass = g.renderPass;
    fci.attachmentCount = 1;
    fci.pAttachments = att;
    fci.width = g.extent.width;
    fci.height = g.extent.height;
    fci.layers = 1;
    vk_check(vkCreateFramebuffer(g.device, &fci, nullptr, &g.framebuffers[i]), "vkCreateFramebuffer");
  }
}

static GpuBuffer create_buffer(std::size_t bytes, VkBufferUsageFlags usage) {
  GpuBuffer b{};
  b.bytes = bytes;
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = bytes;
  bci.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vk_check(vkCreateBuffer(g.device, &bci, nullptr, &b.buffer), "vkCreateBuffer");

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(g.device, b.buffer, &req);

  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(
      g.memProps, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vk_check(vkAllocateMemory(g.device, &ai, nullptr, &b.memory), "vkAllocateMemory");
  vk_check(vkBindBufferMemory(g.device, b.buffer, b.memory, 0), "vkBindBufferMemory");
  vk_check(vkMapMemory(g.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "vkMapMemory");
  std::memset(b.mapped, 0, bytes);
  return b;
}

static void destroy_buffer(GpuBuffer &b) {
  if (b.mapped) vkUnmapMemory(g.device, b.memory);
  if (b.buffer) vkDestroyBuffer(g.device, b.buffer, nullptr);
  if (b.memory) vkFreeMemory(g.device, b.memory, nullptr);
  b = GpuBuffer{};
}

static VkShaderModule load_shader(const char *path) {
  std::vector<char> code = read_binary(path);
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
  VkShaderModule mod = VK_NULL_HANDLE;
  vk_check(vkCreateShaderModule(g.device, &ci, nullptr, &mod), "vkCreateShaderModule");
  return mod;
}

static VkPipeline create_compute(VkPipelineLayout layout, VkShaderModule mod, const char *name) {
  VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  ci.stage.module = mod;
  ci.stage.pName = name;
  ci.layout = layout;
  VkPipeline pipe = VK_NULL_HANDLE;
  vk_check(vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe),
           "vkCreateComputePipelines");
  return pipe;
}

static void create_compute_objects() {
  // accumulate: acc / albedo / normal / params ; resolve: acc / params / out
  const VkDescriptorType sb = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

  auto make_layout = [&](uint32_t count) {
    std::vector<VkDescriptorSetLayoutBinding> b(count);
    for (uint32_t i = 0; i < count; ++i) {
      b[i].binding = i;
      b[i].descriptorType = sb;
      b[i].descriptorCount = 1;
      b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = count;
    ci.pBindings = b.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    vk_check(vkCreateDescriptorSetLayout(g.device, &ci, nullptr, &layout), "vkCreateDescriptorSetLayout");
    return layout;
  };
  g.layoutAcc = make_layout(4);
  g.layoutRes = make_layout(3);

  VkPipelineLayoutCreateInfo plAcc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plAcc.setLayoutCount = 1;
  plAcc.pSetLayouts = &g.layoutAcc;
  vk_check(vkCreatePipelineLayout(g.device, &plAcc, nullptr, &g.pipeLayoutAcc), "vkCreatePipelineLayout");

  VkPipelineLayoutCreateInfo plRes{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plRes.setLayoutCount = 1;
  plRes.pSetLayouts = &g.layoutRes;
  vk_check(vkCreatePipelineLayout(g.device, &plRes, nullptr, &g.pipeLayoutRes), "vkCreatePipelineLayout");

  g.shaderAcc = load_shader(SMALLPT_VIEW_ACCUM_SPV);
  g.shaderRes = load_shader(SMALLPT_VIEW_RESOLVE_SPV);
  g.pipeAcc = create_compute(g.pipeLayoutAcc, g.shaderAcc, "smallpt_view");
  g.pipeRes = create_compute(g.pipeLayoutRes, g.shaderRes, "smallpt_view_resolve");

  VkDescriptorPoolSize poolSize{sb, 8};
  VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pci.maxSets = 2;
  pci.poolSizeCount = 1;
  pci.pPoolSizes = &poolSize;
  vk_check(vkCreateDescriptorPool(g.device, &pci, nullptr, &g.descPool), "vkCreateDescriptorPool");
}

// ---------------------------------------------------------------------------------------------
// Viewport resources (depend on the window size)
// ---------------------------------------------------------------------------------------------

static void destroy_view_resources() {
  if (g.viewTexture) {
    ImGui_ImplVulkan_RemoveTexture(g.viewTexture);
    g.viewTexture = VK_NULL_HANDLE;
  }
  if (g.viewImageView) vkDestroyImageView(g.device, g.viewImageView, nullptr);
  if (g.viewImage) vkDestroyImage(g.device, g.viewImage, nullptr);
  if (g.viewImageMemory) vkFreeMemory(g.device, g.viewImageMemory, nullptr);
  g.viewImageView = VK_NULL_HANDLE;
  g.viewImage = VK_NULL_HANDLE;
  g.viewImageMemory = VK_NULL_HANDLE;
  g.viewImageReadLayout = false;

  destroy_buffer(g.accBuf);
  destroy_buffer(g.albBuf);
  destroy_buffer(g.nrmBuf);
  destroy_buffer(g.paramBuf);
  destroy_buffer(g.outBuf);
  g.setAcc = VK_NULL_HANDLE;
  g.setRes = VK_NULL_HANDLE;
  if (g.descPool) vkResetDescriptorPool(g.device, g.descPool, 0);
}

static void create_view_resources(uint32_t w, uint32_t h) {
  g.viewW = w;
  g.viewH = h;
  const std::size_t pixels = static_cast<std::size_t>(w) * h;

  g.accBuf = create_buffer(pixels * 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  g.albBuf = create_buffer(pixels * 12, 0);
  g.nrmBuf = create_buffer(pixels * 12, 0);
  g.paramBuf = create_buffer(16 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  g.outBuf = create_buffer(pixels * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  // sampled image for the ImGui draw list
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = VK_FORMAT_R8G8B8A8_UNORM;
  ici.extent = {w, h, 1};
  ici.mipLevels = 1;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  vk_check(vkCreateImage(g.device, &ici, nullptr, &g.viewImage), "vkCreateImage");

  VkMemoryRequirements req{};
  vkGetImageMemoryRequirements(g.device, g.viewImage, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_memory_type(g.memProps, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vk_check(vkAllocateMemory(g.device, &ai, nullptr, &g.viewImageMemory), "vkAllocateMemory(image)");
  vk_check(vkBindImageMemory(g.device, g.viewImage, g.viewImageMemory, 0), "vkBindImageMemory");

  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = g.viewImage;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = VK_FORMAT_R8G8B8A8_UNORM;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vk_check(vkCreateImageView(g.device, &vci, nullptr, &g.viewImageView), "vkCreateImageView(view)");

  if (!g.sampler) {
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    vk_check(vkCreateSampler(g.device, &sci, nullptr, &g.sampler), "vkCreateSampler");
  }
  g.viewTexture = ImGui_ImplVulkan_AddTexture(g.viewImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  // descriptor sets: the same buffers, two layouts
  VkDescriptorSetAllocateInfo acc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  acc.descriptorPool = g.descPool;
  acc.descriptorSetCount = 1;
  acc.pSetLayouts = &g.layoutAcc;
  vk_check(vkAllocateDescriptorSets(g.device, &acc, &g.setAcc), "vkAllocateDescriptorSets");

  VkDescriptorSetAllocateInfo res{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  res.descriptorPool = g.descPool;
  res.descriptorSetCount = 1;
  res.pSetLayouts = &g.layoutRes;
  vk_check(vkAllocateDescriptorSets(g.device, &res, &g.setRes), "vkAllocateDescriptorSets");

  VkDescriptorBufferInfo accInfo{g.accBuf.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo albInfo{g.albBuf.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo nrmInfo{g.nrmBuf.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo parInfo{g.paramBuf.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo outInfo{g.outBuf.buffer, 0, VK_WHOLE_SIZE};

  VkWriteDescriptorSet writes[7]{};
  const VkDescriptorBufferInfo *accSrc[7] = {&accInfo, &albInfo, &nrmInfo, &parInfo,
                                             &accInfo, &parInfo, &outInfo};
  const VkDescriptorSet dstSets[7] = {g.setAcc, g.setAcc, g.setAcc, g.setAcc,
                                      g.setRes, g.setRes, g.setRes};
  for (int i = 0; i < 7; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = dstSets[i];
    writes[i].dstBinding = (i < 4) ? static_cast<uint32_t>(i) : static_cast<uint32_t>(i - 4);
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = accSrc[i];
  }
  vkUpdateDescriptorSets(g.device, 7, writes, 0, nullptr);

  g.sampleIndex = 0;
  g.viewImageReadLayout = false;
}

static void recreate_swapchain() {
  vkDeviceWaitIdle(g.device);
  for (VkFramebuffer fb : g.framebuffers) vkDestroyFramebuffer(g.device, fb, nullptr);
  g.framebuffers.clear();
  for (VkImageView v : g.views) vkDestroyImageView(g.device, v, nullptr);
  g.views.clear();
  g.images.clear();
  if (g.swapchain) vkDestroySwapchainKHR(g.device, g.swapchain, nullptr);
  g.swapchain = VK_NULL_HANDLE;

  create_swapchain();
  create_framebuffers();
  destroy_view_resources();
  create_view_resources(g.extent.width, g.extent.height);
  g.resized = false;
}

// ---------------------------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------------------------

static void update_camera(ImGuiIO &io) {
  if (io.WantCaptureMouse) return;
  if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
    g.yaw -= io.MouseDelta.x * 0.006f;
    g.pitch = std::clamp(g.pitch + io.MouseDelta.y * 0.006f, -1.45f, 1.45f);
    reset_samples();
  }
  if (ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.0f)) {
    // pan the target in the camera plane
    float pos[3];
    camera_position(g, pos);
    float fwd[3] = {g.target[0] - pos[0], g.target[1] - pos[1], g.target[2] - pos[2]};
    float len = std::sqrt(fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]);
    for (float &v : fwd) v /= len;
    float right[3] = {-fwd[2], 0.0f, fwd[0]};
    float rlen = std::sqrt(right[0] * right[0] + right[2] * right[2]);
    if (rlen < 1e-6f) {
      right[0] = 1.0f;
      right[2] = 0.0f;
      rlen = 1.0f;
    }
    right[0] /= rlen;
    right[2] /= rlen;
    float up[3] = {right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2],
                   right[0] * fwd[1] - right[1] * fwd[0]};
    const float k = g.distance * 0.0016f;
    for (int i = 0; i < 3; ++i)
      g.target[i] += -right[i] * io.MouseDelta.x * k + up[i] * io.MouseDelta.y * k;
    reset_samples();
  }
  if (io.MouseWheel != 0.0f) {
    g.distance = std::clamp(g.distance * std::pow(1.15f, -io.MouseWheel), 6.0f, 900.0f);
    reset_samples();
  }
}

static void draw_ui(ImGuiIO &io) {
  // the path-traced viewport fills the window, behind every window ImGui draws
  if (g.viewTexture) {
    ImGui::GetBackgroundDrawList()->AddImage(reinterpret_cast<ImTextureID>(g.viewTexture),
                                             ImVec2(0, 0), ImVec2((float)g.viewW, (float)g.viewH));
  }

  ImGui::SetNextWindowPos(ImVec2(14, 14), ImGuiCond_FirstUseEver);
  // the status line appears and disappears, so the panel has to follow the content
  // height or it ends up with a scrollbar
  ImGui::SetNextWindowSizeConstraints(ImVec2(372, 0), ImVec2(FLT_MAX, FLT_MAX));
  ImGui::Begin("vk_smallpt ", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::Text("%.1f fps   %ux%u", 1.0f / std::max(io.DeltaTime, 1e-6f), g.viewW, g.viewH);
  if (g.denoisedView) ImGui::Text("%u spp   denoised", g.sampleIndex);
  else ImGui::Text("%u spp%s", g.sampleIndex, g.accumulate ? "" : "   paused");
  if (!g.status2.empty()) ImGui::TextWrapped("%s", g.status2.c_str());
  ImGui::Separator();

  ImGui::SetNextItemWidth(160.0f);
  ImGui::SliderFloat("exposure", &g.exposure, 0.1f, 4.0f, "%.2f");
  ImGui::Checkbox("Accumulate", &g.accumulate);
  ImGui::SameLine();
  ImGui::BeginDisabled(g.sampleIndex == 0);
  const bool wantDenoise = ImGui::Button("Denoise");
  ImGui::EndDisabled();
  const bool wantSave = ImGui::Button("Save PNG");
  ImGui::SameLine();
  if (ImGui::Button("Reset view")) reset_camera();
  if (g.wantReset) {
    reset_camera();
    g.wantReset = false;
  }

  ImGui::Separator();
  ImGui::TextUnformatted("LMB drag: orbit");
  ImGui::TextUnformatted("RMB drag: pan");
  ImGui::TextUnformatted("wheel: zoom");
  ImGui::TextUnformatted("keys: R reset  D denoise");
  ImGui::TextUnformatted("      S save   space pause");
  ImGui::Text("camera distance %.1f", g.distance);
  ImGui::End();

  // a click and a keyboard shortcut can both request the same action in one frame
  g.wantDenoise = g.wantDenoise || wantDenoise;
  g.wantSave = g.wantSave || wantSave;
}

static void tonemap_to_rgb(const float *rgba, std::vector<uint8_t> &rgb, uint32_t w, uint32_t h,
                           float exposure) {
  rgb.resize(static_cast<std::size_t>(w) * h * 3);
  for (std::size_t i = 0, n = static_cast<std::size_t>(w) * h; i < n; ++i) {
    for (int c = 0; c < 3; ++c) {
      float v = rgba[i * 4 + c] * exposure;
      v = v / (1.0f + v);
      v = std::clamp(v, 0.0f, 1.0f);
      v = std::pow(v, 1.0f / 2.2f);
      rgb[i * 3 + c] = static_cast<uint8_t>(v * 255.0f + 0.5f);
    }
  }
}

static bool save_png(const std::string &path) {
  vkDeviceWaitIdle(g.device);
  std::vector<uint8_t> rgb;
  tonemap_to_rgb(static_cast<const float *>(g.accBuf.mapped), rgb, g.viewW, g.viewH, g.exposure);
  return write_png_rgb(path, rgb.data(), g.viewW, g.viewH);
}

static void run_denoiser() {
  vkDeviceWaitIdle(g.device);
  const auto t0 = std::chrono::high_resolution_clock::now();
  OidnVulkanContext ctx;
  ctx.physicalDevice = g.physical;
  ctx.device = g.device;
  ctx.queue = g.queue;
  ctx.commandPool = g.commandPool;
  ctx.memoryProperties = g.memProps;
  ctx.storage16Bit = g.storage16Bit;

  std::vector<float> src(static_cast<std::size_t>(g.viewW) * g.viewH * 4);
  std::memcpy(src.data(), g.accBuf.mapped, src.size() * sizeof(float));

  OidnAuxImages aux;
  aux.albedo = static_cast<const float *>(g.albBuf.mapped);
  aux.normal = static_cast<const float *>(g.nrmBuf.mapped);

  std::vector<float> dst;
  std::string err;
  double ms = 0.0;
  if (!oidn_denoise_image(ctx, src.data(), static_cast<int>(g.viewW), static_cast<int>(g.viewH), dst,
                          err, &ms, &aux)) {
    g.status2 = "denoise failed: " + err;
    return;
  }
  std::memcpy(g.accBuf.mapped, dst.data(), dst.size() * sizeof(float));
  g.sampleIndex = 0;
  g.accumulate = false;
  g.denoisedView = true;
  g.lastDenoiseMs = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0)
                        .count();
  char buf[160];
  std::snprintf(buf, sizeof(buf), "denoised in %.0f ms (U-Net on the GPU)", ms);
  g.status2 = buf;
}

static void write_params() {
  float *p = static_cast<float *>(g.paramBuf.mapped);
  float pos[3];
  camera_position(g, pos);
  p[0] = static_cast<float>(g.viewW);
  p[1] = static_cast<float>(g.viewH);
  p[2] = static_cast<float>(g.sampleIndex);
  p[3] = 0.0f;
  p[4] = pos[0];
  p[5] = pos[1];
  p[6] = pos[2];
  p[7] = 0.0f;
  p[8] = g.target[0];
  p[9] = g.target[1];
  p[10] = g.target[2];
  p[11] = 0.0f;
  p[12] = g.fovTan;
  p[13] = g.exposure;
  p[14] = 0.0f;
  p[15] = 0.0f;
}

static void record_and_submit(bool doAccumulate) {
  vk_check(vkResetCommandBuffer(g.cmd, 0), "vkResetCommandBuffer");
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk_check(vkBeginCommandBuffer(g.cmd, &bi), "vkBeginCommandBuffer");

  const uint32_t gx = (g.viewW + 7) / 8, gy = (g.viewH + 7) / 8;

  if (doAccumulate) {
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipeAcc);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipeLayoutAcc, 0, 1, &g.setAcc, 0,
                            nullptr);
    vkCmdDispatch(g.cmd, gx, gy, 1);

    VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bb.buffer = g.accBuf.buffer;
    bb.offset = 0;
    bb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &bb, 0, nullptr);
  }

  vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipeRes);
  vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipeLayoutRes, 0, 1, &g.setRes, 0,
                          nullptr);
  vkCmdDispatch(g.cmd, gx, gy, 1);

  VkBufferMemoryBarrier outBar{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  outBar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  outBar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  outBar.buffer = g.outBuf.buffer;
  outBar.offset = 0;
  outBar.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                       0, nullptr, 1, &outBar, 0, nullptr);

  VkImageMemoryBarrier toDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  toDst.oldLayout =
      g.viewImageReadLayout ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
  toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  toDst.srcAccessMask = g.viewImageReadLayout ? VK_ACCESS_SHADER_READ_BIT : 0;
  toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toDst.image = g.viewImage;
  toDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &toDst);

  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {g.viewW, g.viewH, 1};
  vkCmdCopyBufferToImage(g.cmd, g.outBuf.buffer, g.viewImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                         &region);

  VkImageMemoryBarrier imgBar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  imgBar.oldLayout = g.viewImageReadLayout ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                           : VK_IMAGE_LAYOUT_UNDEFINED;
  imgBar.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  imgBar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  imgBar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  imgBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  imgBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  imgBar.image = g.viewImage;
  imgBar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                       0, nullptr, 0, nullptr, 1, &imgBar);

  VkClearValue clear{};
  clear.color = {{0.02f, 0.02f, 0.025f, 1.0f}};
  VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  rp.renderPass = g.renderPass;
  rp.framebuffer = g.framebuffers[g.imageIndex];
  rp.renderArea = {{0, 0}, g.extent};
  rp.clearValueCount = 1;
  rp.pClearValues = &clear;
  vkCmdBeginRenderPass(g.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

  ImDrawData *draw = ImGui::GetDrawData();
  if (draw) ImGui_ImplVulkan_RenderDrawData(draw, g.cmd);

  vkCmdEndRenderPass(g.cmd);
  vk_check(vkEndCommandBuffer(g.cmd), "vkEndCommandBuffer");

  VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.waitSemaphoreCount = 1;
  si.pWaitSemaphores = &g.semImage;
  si.pWaitDstStageMask = &waitStage;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &g.cmd;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &g.semRender;
  vk_check(vkQueueSubmit(g.queue, 1, &si, g.fence), "vkQueueSubmit");
}

// ---------------------------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------------------------

static void parse_args(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) g.framesToRun = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--width") && i + 1 < argc) g.windowW = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--height") && i + 1 < argc) g.windowH = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--save") && i + 1 < argc) g.savePath = argv[++i];
    else if (!std::strcmp(argv[i], "--samples") && i + 1 < argc) g.maxSamples = (uint32_t)std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--denoise")) g.runDenoiseOnExit = true;
  }
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  create_window();
  create_instance();
  pick_device();
  create_swapchain();
  create_render_pass();
  create_framebuffers();

  // command pool / buffer / sync
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = g.family;
  vk_check(vkCreateCommandPool(g.device, &pci, nullptr, &g.commandPool), "vkCreateCommandPool");

  VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbi.commandPool = g.commandPool;
  cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbi.commandBufferCount = 1;
  vk_check(vkAllocateCommandBuffers(g.device, &cbi, &g.cmd), "vkAllocateCommandBuffers");

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  vk_check(vkCreateFence(g.device, &fci, nullptr, &g.fence), "vkCreateFence");

  VkSemaphoreCreateInfo semci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  vk_check(vkCreateSemaphore(g.device, &semci, nullptr, &g.semImage), "vkCreateSemaphore");
  vk_check(vkCreateSemaphore(g.device, &semci, nullptr, &g.semRender), "vkCreateSemaphore");

  create_compute_objects();

  // ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.IniFilename = nullptr;
  ImFontConfig cfg;
  cfg.SizePixels = 16.0f;
  io.Fonts->AddFontDefault(&cfg);
  ImGui_ImplWin32_Init(g.hwnd);

  ImGui_ImplVulkan_InitInfo vii{};
  vii.Instance = g.instance;
  vii.PhysicalDevice = g.physical;
  vii.Device = g.device;
  vii.QueueFamily = g.family;
  vii.Queue = g.queue;
  vii.DescriptorPoolSize = 64;
  vii.MinImageCount = 2;
  vii.ImageCount = static_cast<uint32_t>(g.images.size());
  vii.PipelineInfoMain.RenderPass = g.renderPass;
  vii.PipelineInfoMain.Subpass = 0;
  vii.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  vii.ApiVersion = VK_API_VERSION_1_2;
  if (!ImGui_ImplVulkan_Init(&vii)) fail("ImGui_ImplVulkan_Init");

  create_view_resources(g.extent.width, g.extent.height);
  g.wantDenoise = false;
  g.wantSave = false;

  std::printf("[info] device: %s\n", "Vulkan");
  std::printf("[info] viewport %ux%u, %s\n", g.viewW, g.viewH, g.format == VK_FORMAT_B8G8R8A8_UNORM ? "B8G8R8A8" : "other");
  {
    RECT rc{};
    GetClientRect(g.hwnd, &rc);
    std::printf("[info] client %ldx%ld, swapchain %ux%u\n", rc.right - rc.left, rc.bottom - rc.top,
                g.extent.width, g.extent.height);
  }

  int frameIndex = 0;
  auto lastTime = std::chrono::high_resolution_clock::now();

  while (g.running) {
    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
      TranslateMessage(&msg);
      DispatchMessage(&msg);
      if (msg.message == WM_QUIT) g.running = false;
    }
    if (!g.running) break;
    if (g.resized) recreate_swapchain();

    vk_check(vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    vk_check(vkResetFences(g.device, 1, &g.fence), "vkResetFences");

    VkResult acquired = vkAcquireNextImageKHR(g.device, g.swapchain, UINT64_MAX, g.semImage,
                                             VK_NULL_HANDLE, &g.imageIndex);
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
      recreate_swapchain();
      continue;
    }
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR)
      vk_check(acquired, "vkAcquireNextImageKHR");

    ImGui_ImplWin32_NewFrame();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    update_camera(io);
    draw_ui(io);
    ImGui::Render();

    bool doAccumulate = g.accumulate && g.sampleIndex < g.maxSamples;
    write_params(); // frame index must be the pre-dispatch sample count
    record_and_submit(doAccumulate);
    if (doAccumulate) ++g.sampleIndex;

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &g.semRender;
    pi.swapchainCount = 1;
    pi.pSwapchains = &g.swapchain;
    pi.pImageIndices = &g.imageIndex;
    VkResult present = vkQueuePresentKHR(g.queue, &pi);
    if (present == VK_ERROR_OUT_OF_DATE_KHR || present == VK_SUBOPTIMAL_KHR) g.resized = true;
    g.viewImageReadLayout = true;

    if (g.wantDenoise) {
      g.wantDenoise = false;
      run_denoiser();
    }
    if (g.wantSave) {
      g.wantSave = false;
      char path[512];
      // keep the working tree tidy: renders/ when it exists, otherwise the cwd
      const DWORD attr = GetFileAttributesA("renders");
      if (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY))
        std::snprintf(path, sizeof(path), "renders/viewer_%05d.png", frameIndex);
      else
        std::snprintf(path, sizeof(path), "viewer_%05d.png", frameIndex);      if (save_png(path)) {
        char buf[560];
        std::snprintf(buf, sizeof(buf), "saved %s (%u spp)", path, g.sampleIndex);
        g.status2 = buf;
      } else {
        g.status2 = "PNG write failed";
      }
    }

    ++frameIndex;
    if (g.framesToRun > 0 && frameIndex >= g.framesToRun) g.running = false;
  }

  if (g.runDenoiseOnExit && g.sampleIndex > 0) run_denoiser();

  if (!g.savePath.empty()) {
    if (save_png(g.savePath)) std::printf("[info] wrote %s (%u spp)\n", g.savePath.c_str(), g.sampleIndex);
    else std::printf("[warn] could not write %s\n", g.savePath.c_str());
  }
  std::printf("[info] %d frames, %u spp\n", frameIndex, g.sampleIndex);

  vkDeviceWaitIdle(g.device);
  destroy_view_resources();
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplWin32_Shutdown();
  ImGui::DestroyContext();
  vkDestroySemaphore(g.device, g.semImage, nullptr);
  vkDestroySemaphore(g.device, g.semRender, nullptr);
  vkDestroyFence(g.device, g.fence, nullptr);
  vkDestroyCommandPool(g.device, g.commandPool, nullptr);
  if (g.sampler) vkDestroySampler(g.device, g.sampler, nullptr);
  vkDestroyPipeline(g.device, g.pipeAcc, nullptr);
  vkDestroyPipeline(g.device, g.pipeRes, nullptr);
  vkDestroyShaderModule(g.device, g.shaderAcc, nullptr);
  vkDestroyShaderModule(g.device, g.shaderRes, nullptr);
  vkDestroyPipelineLayout(g.device, g.pipeLayoutAcc, nullptr);
  vkDestroyPipelineLayout(g.device, g.pipeLayoutRes, nullptr);
  vkDestroyDescriptorPool(g.device, g.descPool, nullptr);
  vkDestroyDescriptorSetLayout(g.device, g.layoutAcc, nullptr);
  vkDestroyDescriptorSetLayout(g.device, g.layoutRes, nullptr);
  for (VkFramebuffer fb : g.framebuffers) vkDestroyFramebuffer(g.device, fb, nullptr);
  for (VkImageView v : g.views) vkDestroyImageView(g.device, v, nullptr);
  vkDestroyRenderPass(g.device, g.renderPass, nullptr);
  vkDestroySwapchainKHR(g.device, g.swapchain, nullptr);
  vkDestroySurfaceKHR(g.instance, g.surface, nullptr);
  vkDestroyDevice(g.device, nullptr);
  vkDestroyInstance(g.instance, nullptr);
  DestroyWindow(g.hwnd);
  return 0;
}
