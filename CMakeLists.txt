cmake_minimum_required(VERSION 3.20)

project(clspv_vulkan_compute LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Vulkan REQUIRED)

set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/vadd.cl)
set(SPV_NAME  vadd.spv)
set(SPV_OUT   ${CMAKE_CURRENT_BINARY_DIR}/${SPV_NAME})

# ---------------------------------------------------------------------------
# Step 1: compile src/vadd.cl into vadd.spv with clspv
# ---------------------------------------------------------------------------
find_program(CLSPV_EXECUTABLE NAMES clspv
  HINTS $ENV{CLSPV_ROOT}/bin
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/clspv/bin
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/clspv/build)

if(CLSPV_EXECUTABLE)
  message(STATUS "clspv: ${CLSPV_EXECUTABLE}")
  add_custom_command(
    OUTPUT  ${SPV_OUT}
    COMMAND ${CLSPV_EXECUTABLE} ${KERNEL_SRC} -o ${SPV_OUT} -O3
    DEPENDS ${KERNEL_SRC}
    COMMENT "clspv: src/vadd.cl -> ${SPV_NAME}"
    VERBATIM)
else()
  message(WARNING
    "clspv was not found in PATH or CLSPV_ROOT, so the SPIR-V cannot be built from src/vadd.cl.\n"
    "         Falling back to the pre-built shaders/${SPV_NAME} kept in the repository (see the README\n"
    "         section 'About shaders/vadd.spv'). Re-run configure after installing clspv to compile the\n"
    "         kernel instead.")
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SPV_NAME})
    message(FATAL_ERROR "Neither clspv nor shaders/${SPV_NAME} is available, cannot continue.")
  endif()
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SPV_NAME} ${SPV_OUT} COPYONLY)
endif()

# ---------------------------------------------------------------------------
# Step 2: the host program
# ---------------------------------------------------------------------------
add_executable(vkcompute src/main.cpp ${SPV_OUT})

target_link_libraries(vkcompute PRIVATE Vulkan::Vulkan)

# Bake the absolute .spv path into the executable so it works from any working directory
file(TO_CMAKE_PATH "${SPV_OUT}" SPV_OUT_CMAKE)
target_compile_definitions(vkcompute PRIVATE VADD_SPV_PATH="${SPV_OUT_CMAKE}")

if(MSVC)
  target_compile_options(vkcompute PRIVATE /W4 /utf-8)
  target_compile_definitions(vkcompute PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_property(TARGET vkcompute PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Add the kernel source to the project so it can be edited in the IDE
target_sources(vkcompute PRIVATE ${KERNEL_SRC})
set_source_files_properties(${KERNEL_SRC} PROPERTIES VS_TOOL_OVERRIDE "None")

# ---------------------------------------------------------------------------
# Step 3: smallpt -- path tracing on the same framework, kernel written in C++ for OpenCL
# (src/smallpt_cpp.cl). src/smallpt.cl (OpenCL C) is kept as the reference implementation and is
# built as smallpt_c.spv by the reference targets below.
#
# Two clspv specifics of the C++ mode are handled here:
#   - --cl-std=CLC++2021 requires -inline-entry-points (generic address space);
#   - reqd_work_group_size is ignored in that mode, so the workgroup size arrives as spec constants
#     0/1/2. spirv-opt freezes them to 8/8/1, which leaves a constant WorkgroupSize builtin and no
#     spec constant at all - the same shape glslc produces for the GLSL kernels.
#
# The default module is built for speed, not for bit-exactness with the OpenCL C reference:
#   --use-native-builtins=sin,cos   cos/sin map to the hardware ops instead of the inlined libclc
#                                   polynomials (196 instructions per call site)
#   --cl-fast-relaxed-math          drops the denormal/range scaling guards around division
# Those two together take 480x360/32spp from ~1.6 s to ~0.26 s on this machine, and change the
# last bits of the image. `smallpt_exact` (below) is the same source built with plain -O3, which
# still renders bit-identical to src/smallpt.cl and stays available for comparisons.
# ---------------------------------------------------------------------------
set(SMALLPT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_cpp.cl)
set(SMALLPT_RAW ${CMAKE_CURRENT_BINARY_DIR}/smallpt_fast_raw.spv)
set(SMALLPT_PLAIN_RAW ${CMAKE_CURRENT_BINARY_DIR}/smallpt_cpp_raw.spv)
set(SMALLPT_SPV ${CMAKE_CURRENT_BINARY_DIR}/smallpt.spv)
set(SMALLPT_OUT ${CMAKE_CURRENT_SOURCE_DIR}/renders/smallpt.png)

find_program(SPIRV_OPT_EXECUTABLE NAMES spirv-opt
HINTS $ENV{VULKAN_SDK}/Bin
"C:/VulkanSDK/1.4.357.0/Bin"
${CMAKE_CURRENT_SOURCE_DIR}/tools/VulkanSDK/Bin)

if(CLSPV_EXECUTABLE AND NOT TARGET smallpt)
  if(NOT SPIRV_OPT_EXECUTABLE)
    message(FATAL_ERROR "spirv-opt was not found, so the C++ for OpenCL kernel cannot be built.\n"
      "         Install the Vulkan SDK (spirv-opt ships in its Bin directory) and re-configure.")
  endif()
  add_custom_command(
    OUTPUT  ${SMALLPT_RAW}
    COMMAND ${CLSPV_EXECUTABLE} ${SMALLPT_SRC} -o ${SMALLPT_RAW} -O3
            --cl-std=CLC++2021 -inline-entry-points
            --use-native-builtins=sin,cos --cl-fast-relaxed-math
    DEPENDS ${SMALLPT_SRC}
    COMMENT "clspv: src/smallpt_cpp.cl -> smallpt_cpp_raw.spv (C++ for OpenCL, native sin/cos + fast math)"
    VERBATIM)
  add_custom_command(
    OUTPUT  ${SMALLPT_SPV}
    COMMAND ${SPIRV_OPT_EXECUTABLE} ${SMALLPT_RAW} -o ${SMALLPT_SPV}
            --set-spec-const-default-value "0:8 1:8 2:1" --freeze-spec-const
    DEPENDS ${SMALLPT_RAW} ${SMALLPT_SRC}
    COMMENT "spirv-opt: freeze the workgroup size spec constants to 8 8 1"
    VERBATIM)

  add_executable(smallpt src/smallpt_main.cpp src/png_writer.cpp ${SMALLPT_SPV})
  target_link_libraries(smallpt PRIVATE Vulkan::Vulkan)

  file(TO_CMAKE_PATH "${SMALLPT_SPV}" SMALLPT_SPV_CMAKE)
  file(TO_CMAKE_PATH "${SMALLPT_OUT}" SMALLPT_OUT_CMAKE)
  target_compile_definitions(smallpt PRIVATE
    SMALLPT_SPV_PATH="${SMALLPT_SPV_CMAKE}"
    SMALLPT_OUT_PATH="${SMALLPT_OUT_CMAKE}")

  if(MSVC)
    target_compile_options(smallpt PRIVATE /W4 /utf-8)
    target_compile_definitions(smallpt PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()

  set_property(TARGET smallpt PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  target_sources(smallpt PRIVATE ${SMALLPT_SRC})
  set_source_files_properties(${SMALLPT_SRC} PROPERTIES VS_TOOL_OVERRIDE "None")

  # Reference build of the same scene in OpenCL C, so the two language modes can be compared on the
  # same machine: smallpt.exe build/smallpt_c.spv <w> <h> <spp> <out>
  set(SMALLPT_C_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt.cl)
  set(SMALLPT_C_SPV ${CMAKE_CURRENT_BINARY_DIR}/smallpt_c.spv)
  add_custom_command(
    OUTPUT  ${SMALLPT_C_SPV}
    COMMAND ${CLSPV_EXECUTABLE} ${SMALLPT_C_SRC} -o ${SMALLPT_C_SPV} -O3
    DEPENDS ${SMALLPT_C_SRC}
    COMMENT "clspv: src/smallpt.cl -> smallpt_c.spv (OpenCL C reference)"
    VERBATIM)
  add_custom_target(smallpt_c DEPENDS ${SMALLPT_C_SPV})
  add_dependencies(smallpt smallpt_c)
  target_sources(smallpt PRIVATE ${SMALLPT_C_SRC})
  set_source_files_properties(${SMALLPT_C_SRC} PROPERTIES VS_TOOL_OVERRIDE "None")

  # Precision build of the C++ kernel: the same source without the native-builtin and relaxed-math
  # flags, so the image stays bit-identical to src/smallpt.cl. Use it to check that a change did not
  # alter the arithmetic:
  #   smallpt.exe build/smallpt_exact.spv 320 240 8 out.png
  # The plain clspv step lives here and Step 7 reuses it for smallpt_cpp.spv, so the plain module is
  # compiled once and both precision modules come from the same raw file.
  set(SMALLPT_EXACT_SPV ${CMAKE_CURRENT_BINARY_DIR}/smallpt_exact.spv)
  add_custom_command(
    OUTPUT  ${SMALLPT_PLAIN_RAW}
    COMMAND ${CLSPV_EXECUTABLE} ${SMALLPT_SRC} -o ${SMALLPT_PLAIN_RAW} -O3
            --cl-std=CLC++2021 -inline-entry-points
    DEPENDS ${SMALLPT_SRC}
    COMMENT "clspv: src/smallpt_cpp.cl -> smallpt_cpp_raw.spv (no fast-math flags)"
    VERBATIM)
  add_custom_command(
    OUTPUT  ${SMALLPT_EXACT_SPV}
    COMMAND ${SPIRV_OPT_EXECUTABLE} ${SMALLPT_PLAIN_RAW} -o ${SMALLPT_EXACT_SPV}
            --set-spec-const-default-value "0:8 1:8 2:1" --freeze-spec-const
    DEPENDS ${SMALLPT_PLAIN_RAW} ${SMALLPT_SRC}
    COMMENT "spirv-opt: freeze the workgroup size spec constants to 8 8 1 (smallpt_exact)"
    VERBATIM)
  add_custom_target(smallpt_exact DEPENDS ${SMALLPT_EXACT_SPV})
elseif(NOT CLSPV_EXECUTABLE)
  message(WARNING "No clspv, skipping the smallpt target (it must be compiled from src/smallpt_cpp.cl)")
endif()

# ---------------------------------------------------------------------------
# Step 4: two reference builds of the same scene, used to compare compilers
# (not the main rendering path)
#   smallpt_buf.cl  -- built by clspv, scene data read from the parameter buffer
#                      (the counter-experiment for the "__constant demotion" problem)
#   smallpt.comp    -- GLSL built by glslc (equivalent implementation, entry point is main)
# Both are run through the same smallpt.exe:
#   smallpt.exe <spv> <width> <height> <spp> <output> <entryPointName>
# ---------------------------------------------------------------------------
if(CLSPV_EXECUTABLE)
  set(SMALLPT_BUF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_buf.cl)
  set(SMALLPT_BUF_SPV ${CMAKE_CURRENT_BINARY_DIR}/smallpt_buf.spv)
  add_custom_command(
    OUTPUT  ${SMALLPT_BUF_SPV}
    COMMAND ${CLSPV_EXECUTABLE} ${SMALLPT_BUF_SRC} -o ${SMALLPT_BUF_SPV} -O3
    DEPENDS ${SMALLPT_BUF_SRC}
    COMMENT "clspv: src/smallpt_buf.cl -> smallpt_buf.spv"
    VERBATIM)
  add_custom_target(smallpt_buf DEPENDS ${SMALLPT_BUF_SPV})
  if(TARGET smallpt)
    add_dependencies(smallpt smallpt_buf)
  endif()
endif()

find_program(GLSLC_EXECUTABLE NAMES glslc
  HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
find_program(SPIRV_OPT_EXECUTABLE NAMES spirv-opt
  HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
if(GLSLC_EXECUTABLE)
  set(SMALLPT_GLSL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt.comp)
  set(SMALLPT_GLSL_SPV ${CMAKE_CURRENT_BINARY_DIR}/smallpt_glsl.spv)
  add_custom_command(
    OUTPUT  ${SMALLPT_GLSL_SPV}
    COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 -O ${SMALLPT_GLSL_SRC} -o ${SMALLPT_GLSL_SPV}
    DEPENDS ${SMALLPT_GLSL_SRC}
    COMMENT "glslc: src/smallpt.comp -> smallpt_glsl.spv"
    VERBATIM)
  add_custom_target(smallpt_glsl DEPENDS ${SMALLPT_GLSL_SPV})
  if(TARGET smallpt)
    add_dependencies(smallpt smallpt_glsl)
  endif()

  # The literal-subscript trick applies to GLSL as well: glslc materializes a
  # `const vec4 P[6]` into `OpVariable %_ptr_Function__arr_v4float_uint_6 Function`, so
  # P[i] inside the loop is a dynamic OpAccessChain. Unrolling with macros and literal
  # subscripts makes both the arrays and the AccessChains disappear.
  set(SMALLPT_LIT_GLSL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_lit.comp)
  set(SMALLPT_LIT_GLSL_SPV ${CMAKE_CURRENT_BINARY_DIR}/smallpt_lit_glsl.spv)
  add_custom_command(
    OUTPUT  ${SMALLPT_LIT_GLSL_SPV}
    COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 -O ${SMALLPT_LIT_GLSL_SRC} -o ${SMALLPT_LIT_GLSL_SPV}
    DEPENDS ${SMALLPT_LIT_GLSL_SRC}
    COMMENT "glslc: src/smallpt_lit.comp -> smallpt_lit_glsl.spv (literal indices)"
    VERBATIM)
  add_custom_target(smallpt_lit_glsl DEPENDS ${SMALLPT_LIT_GLSL_SPV})
  if(TARGET smallpt)
    add_dependencies(smallpt smallpt_lit_glsl)
  endif()
else()
  message(WARNING "No glslc, skipping the GLSL reference builds")
endif()

# ---------------------------------------------------------------------------
# Step 5: smallpt_native -- the fastest clspv configuration (scene read from a buffer)
# The point is to keep sin/cos away from the libclc software implementations and to relax the
# correctly-rounded division guards:
#   --use-native-builtins=sin,cos   -> sin/cos map to the hardware OpExtInst Sin/Cos
#   --cl-fast-relaxed-math          -> drop the division scaling / denormal guards, allow FMA
# With smallpt_buf.cl (scene in a __global parameter buffer) this measured 302 ms at 480x360/32spp
# against 1599 ms for the baseline (src/smallpt.cl with plain -O3) and 258 ms for the GLSL build.
# ---------------------------------------------------------------------------
if(CLSPV_EXECUTABLE)
  set(SMALLPT_NATIVE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_buf.cl)
  set(SMALLPT_NATIVE_SPV ${CMAKE_CURRENT_BINARY_DIR}/smallpt_native.spv)
  add_custom_command(
    OUTPUT  ${SMALLPT_NATIVE_SPV}
    COMMAND ${CLSPV_EXECUTABLE} ${SMALLPT_NATIVE_SRC} -o ${SMALLPT_NATIVE_SPV} -O3
            --use-native-builtins=sin,cos --cl-fast-relaxed-math
    DEPENDS ${SMALLPT_NATIVE_SRC}
    COMMENT "clspv: src/smallpt_buf.cl -> smallpt_native.spv (native sin/cos + fast math)"
    VERBATIM)
  add_custom_target(smallpt_native DEPENDS ${SMALLPT_NATIVE_SPV})
  if(TARGET smallpt)
    add_dependencies(smallpt smallpt_native)
  endif()
endif()

# ---------------------------------------------------------------------------
# Step 6: smallpt_lit -- the literal-subscript build (fastest clspv variant, needs no extra
# descriptor). clspv has no __constant address space, so a module-scope __constant array is demoted
# to a per-thread Private array; what actually decides whether Private is used is whether the
# subscript is a constant. The `#pragma clang loop unroll(full)` in smallpt.cl has no effect on
# clspv (the loop survives and i stays a run-time value), so PLANES[i*7+k] can only index Private
# dynamically. Here the 6 walls and the 3 spheres are unrolled by hand with macros and literal
# subscripts, which turns every subscript into a constant and lets clspv fold the whole scene
# table into immediates:
#   Private accesses 52 -> 0, OpLoad 58 -> 6, module 26044 -> 24944 bytes,
#   480x360/32spp 310.3 -> 197.7 ms (GLSL 220.8 ms).
# ---------------------------------------------------------------------------
if(CLSPV_EXECUTABLE)
  set(SMALLPT_LIT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_lit.cl)
  set(SMALLPT_LIT_SPV ${CMAKE_CURRENT_BINARY_DIR}/smallpt_lit.spv)
  add_custom_command(
    OUTPUT  ${SMALLPT_LIT_SPV}
    COMMAND ${CLSPV_EXECUTABLE} ${SMALLPT_LIT_SRC} -o ${SMALLPT_LIT_SPV} -O3
            --use-native-builtins=sin,cos --cl-fast-relaxed-math
    DEPENDS ${SMALLPT_LIT_SRC}
    COMMENT "clspv: src/smallpt_lit.cl -> smallpt_lit.spv (literal indices, Private=0)"
    VERBATIM)
  add_custom_target(smallpt_lit DEPENDS ${SMALLPT_LIT_SPV})
  if(TARGET smallpt)
    add_dependencies(smallpt smallpt_lit)
  endif()
endif()

# ---------------------------------------------------------------------------
# Step 7: smallpt_cpp -- the same kernel written in C++ for OpenCL, with the vector math expressed
# through operator overloading (Vec3 + - * dot cross normalize) instead of add3/sub3/scl3/dot3/...
#
# C++ mode needs two extra flags:
#   --cl-std=CLC++2021      select C++ for OpenCL
#   -inline-entry-points    mandatory: C++ uses the generic address space, the entry point has to be
#                           inlined before the address spaces can be resolved
#
# Two quirks of this mode have to be worked around, both visible in the generated module:
#   * the OpenCL global offset becomes a 12-byte push constant block that clspv reads to build
#     get_global_id(); src/smallpt_main.cpp declares that range and pushes zeros before the dispatch;
#   * reqd_work_group_size is ignored: no OpExecutionMode LocalSize is emitted, the workgroup size
#     ends up as spec constants SpecId 0/1/2 whose defaults are 1/1/1. A driver that ignores the
#     specialization info therefore runs one thread per workgroup. spirv-opt is used below to give
#     those constants the real 8/8/1 values and to freeze them into ordinary constants, which is
#     exactly what glslc emits for a GLSL workgroup size and what every driver honours.
# ---------------------------------------------------------------------------
if(CLSPV_EXECUTABLE)
  set(SMALLPT_CPP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_cpp.cl)
  set(SMALLPT_CPP_SPV ${CMAKE_CURRENT_BINARY_DIR}/smallpt_cpp.spv)
  # The plain clspv step is defined once in Step 3 (it also feeds smallpt_exact.spv)
  if(SPIRV_OPT_EXECUTABLE)
    add_custom_command(
      OUTPUT  ${SMALLPT_CPP_SPV}
      COMMAND ${SPIRV_OPT_EXECUTABLE} ${SMALLPT_PLAIN_RAW} -o ${SMALLPT_CPP_SPV}
              --set-spec-const-default-value "0:8 1:8 2:1" --freeze-spec-const
      DEPENDS ${SMALLPT_PLAIN_RAW} ${SMALLPT_CPP_SRC}
      COMMENT "spirv-opt: freeze the workgroup size to a constant 8/8/1 in smallpt_cpp.spv"
      VERBATIM)
    set(SMALLPT_CPP_OUTPUT ${SMALLPT_CPP_SPV})
  else()
    message(WARNING "spirv-opt not found: smallpt_cpp.spv keeps 1/1/1 spec-constant workgroup size")
    set(SMALLPT_CPP_OUTPUT ${SMALLPT_PLAIN_RAW})
  endif()

  add_custom_target(smallpt_cpp DEPENDS ${SMALLPT_CPP_OUTPUT})
  if(TARGET smallpt)
    add_dependencies(smallpt smallpt_cpp)
  endif()
endif()

# ---------------------------------------------------------------------------
# Step 8: the OpenImageDenoise (OIDN) denoiser, executed as Vulkan compute
#
# The U-Net of OIDN's RT filter is rebuilt from its published weights and run with plain
# compute shaders, so no OIDN library, SYCL or oneDNN backend is involved:
#   * tools/oidn/rt_hdr.bin holds the float32 weights, converted from OIDN's own
#     weights/rt_hdr.tza by tools/oidn/export_oidn_weights.py (Apache-2.0, (c) Intel);
#   * src/oidn_{pre,conv,pool,upsample,concat_copy,post}.comp are the six passes;
#   * src/oidn_denoise.cpp dispatches the 30-pass chain on the same device and queue that
#     the path tracer uses.
# ---------------------------------------------------------------------------
if(GLSLC_EXECUTABLE AND TARGET smallpt)
  set(OIDN_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/oidn)
  file(MAKE_DIRECTORY ${OIDN_SHADER_DIR})
  set(OIDN_SHADER_SRCS
    src/oidn_pre.comp
    src/oidn_conv.comp
    src/oidn_conv_tiled.comp
    src/oidn_pool.comp
    src/oidn_upsample.comp
    src/oidn_concat_copy.comp
    src/oidn_post.comp)
  set(OIDN_CL_SRCS
    src/oidn_pre.cl
    src/oidn_conv.cl
    src/oidn_conv_tiled.cl
    src/oidn_pool.cl
    src/oidn_upsample.cl
    src/oidn_concat_copy.cl
    src/oidn_post.cl)
  set(OIDN_SHADER_SPVS)
  # Every pass is built twice: the FP32 tensor variant (<name>.spv) and the FP16 storage variant
  # (<name>_h.spv, -DOIDN_HALF_TENSOR=1). oidn_denoise.cpp picks the set at run time with
  # OIDN_STORE=fp16, so the two can be compared on the same input and machine.
  foreach(src ${OIDN_SHADER_SRCS})
    get_filename_component(name ${src} NAME_WE)
    set(spv ${OIDN_SHADER_DIR}/${name}.spv)
    add_custom_command(
      OUTPUT  ${spv}
      COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 -O
              ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${spv}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
      COMMENT "glslc: ${src} -> oidn/${name}.spv"
      VERBATIM)
    list(APPEND OIDN_SHADER_SPVS ${spv})

    set(spvHalf ${OIDN_SHADER_DIR}/${name}_h.spv)
    add_custom_command(
      OUTPUT  ${spvHalf}
      COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 -O -DOIDN_HALF_TENSOR=1
              ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${spvHalf}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
      COMMENT "glslc: ${src} -> oidn/${name}_h.spv (FP16 tensors)"
      VERBATIM)
    list(APPEND OIDN_SHADER_SPVS ${spvHalf})
  endforeach()
  # The same seven passes also exist as C++ for OpenCL sources (src/oidn_*.cl) compiled by clspv,
  # again twice (FP32 / FP16 tensors). clspv needs --cl-std=CLC++2021 -inline-entry-points, and it
  # ignores reqd_work_group_size in that mode, so the workgroup size arrives as spec constants that
  # spirv-opt freezes to 8/8/1 (same trick as the smallpt kernel).
  foreach(src ${OIDN_CL_SRCS})
    get_filename_component(name ${src} NAME_WE)
    set(spv ${OIDN_SHADER_DIR}/${name}_cl.spv)
    set(raw ${OIDN_SHADER_DIR}/${name}_cl_raw.spv)
    add_custom_command(
      OUTPUT  ${spv}
      COMMAND ${CLSPV_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${raw} -O3
              --cl-std=CLC++2021 -inline-entry-points
      COMMAND ${SPIRV_OPT_EXECUTABLE} ${raw} -o ${spv}
              --set-spec-const-default-value "0:8 1:8 2:1" --freeze-spec-const
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
      COMMENT "clspv: ${src} -> oidn/${name}_cl.spv"
      VERBATIM)
    list(APPEND OIDN_SHADER_SPVS ${spv})

    set(spvHalf ${OIDN_SHADER_DIR}/${name}_cl_h.spv)
    set(rawHalf ${OIDN_SHADER_DIR}/${name}_cl_h_raw.spv)
    add_custom_command(
      OUTPUT  ${spvHalf}
      COMMAND ${CLSPV_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${rawHalf} -O3
              --cl-std=CLC++2021 -inline-entry-points -DOIDN_HALF_TENSOR=1
      COMMAND ${SPIRV_OPT_EXECUTABLE} ${rawHalf} -o ${spvHalf}
              --set-spec-const-default-value "0:8 1:8 2:1" --freeze-spec-const
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
      COMMENT "clspv: ${src} -> oidn/${name}_cl_h.spv (FP16 tensors)"
      VERBATIM)
    list(APPEND OIDN_SHADER_SPVS ${spvHalf})
  endforeach()

  # The tiled convolution is register blocked, and the blocking (pixels per thread, output
  # channels per work group) is a compile-time choice. The module built above uses the kernel's
  # own defaults, which are the configuration that measured fastest on this GPU (3x2 pixels over 4
  # channels, block 24x16). These alternates are built too, so OIDN_TILE=PXxPYgGROUP can switch
  # between them on the same input without a rebuild:
  #   name PX PY GROUP
  set(OIDN_TILE_VARIANTS "2x2g8 2 2 8" "4x2g4 4 2 4" "4x3g4 4 3 4")
  foreach(v ${OIDN_TILE_VARIANTS})
    separate_arguments(parts UNIX_COMMAND "${v}")
    list(GET parts 0 vname)
    list(GET parts 1 vpx)
    list(GET parts 2 vpy)
    list(GET parts 3 vgroup)
    set(defs -DOIDN_PX=${vpx} -DOIDN_PY=${vpy} -DOIDN_GROUP=${vgroup})
    foreach(half 0 1)
      if(half)
        set(suffix _h)
        set(extra -DOIDN_HALF_TENSOR=1)
        set(label "FP16 tensors")
      else()
        set(suffix "")
        set(extra "")
        set(label "FP32 tensors")
      endif()
      set(spv ${OIDN_SHADER_DIR}/oidn_conv_tiled_${vname}_cl${suffix}.spv)
      set(raw ${OIDN_SHADER_DIR}/oidn_conv_tiled_${vname}_cl${suffix}_raw.spv)
      add_custom_command(
        OUTPUT  ${spv}
        COMMAND ${CLSPV_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/oidn_conv_tiled.cl -o ${raw}
                -O3 --cl-std=CLC++2021 -inline-entry-points
                -DOIDN_PX=${vpx} -DOIDN_PY=${vpy} -DOIDN_GROUP=${vgroup} ${extra}
        COMMAND ${SPIRV_OPT_EXECUTABLE} ${raw} -o ${spv}
                --set-spec-const-default-value "0:8 1:8 2:1" --freeze-spec-const
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/oidn_conv_tiled.cl
        COMMENT "clspv: oidn_conv_tiled.cl (${vname}, ${label}) -> oidn/oidn_conv_tiled_${vname}_cl${suffix}.spv"
        VERBATIM)
      list(APPEND OIDN_SHADER_SPVS ${spv})
    endforeach()
  endforeach()

  add_custom_target(oidn_shaders DEPENDS ${OIDN_SHADER_SPVS})
  add_dependencies(smallpt oidn_shaders)

  target_sources(smallpt PRIVATE src/oidn_denoise.cpp ${OIDN_SHADER_SRCS})

  set(OIDN_WEIGHTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/oidn/rt_hdr.bin)
  file(TO_CMAKE_PATH "${OIDN_SHADER_DIR}" OIDN_SHADER_DIR_CMAKE)
  file(TO_CMAKE_PATH "${OIDN_WEIGHTS}" OIDN_WEIGHTS_CMAKE)
  target_compile_definitions(smallpt PRIVATE
    OIDN_DENOISER_SPV_DIR="${OIDN_SHADER_DIR_CMAKE}"
    OIDN_DENOISER_WEIGHTS_PATH="${OIDN_WEIGHTS_CMAKE}")

  foreach(src ${OIDN_SHADER_SRCS} ${OIDN_CL_SRCS})
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/${src} PROPERTIES VS_TOOL_OVERRIDE "None")
  endforeach()

  if(NOT EXISTS ${OIDN_WEIGHTS})
    message(WARNING
      "tools/oidn/rt_hdr.bin is missing: run\n"
      "  python tools/oidn/export_oidn_weights.py <oidn>/weights/rt_hdr.tza tools/oidn/rt_hdr.bin\n"
      "otherwise smallpt builds but skips the denoiser at run time.")
  endif()

  # Standalone driver: raw float32 in, raw float32 out. Same passes, no path tracer, which makes
  # it the quickest way to diff the Vulkan output against tools/oidn/ref_forward.py.
  add_executable(oidn_denoise_tool src/oidn_denoise_tool.cpp src/oidn_denoise.cpp)
  target_link_libraries(oidn_denoise_tool PRIVATE Vulkan::Vulkan)
  target_compile_definitions(oidn_denoise_tool PRIVATE
    OIDN_DENOISER_SPV_DIR="${OIDN_SHADER_DIR_CMAKE}"
    OIDN_DENOISER_WEIGHTS_PATH="${OIDN_WEIGHTS_CMAKE}")
  add_dependencies(oidn_denoise_tool oidn_shaders)
  if(MSVC)
    target_compile_options(oidn_denoise_tool PRIVATE /W4 /utf-8)
    target_compile_definitions(oidn_denoise_tool PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
  set_property(TARGET oidn_denoise_tool PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

  # Single-pass driver used to tune the convolution: it runs exactly one 3x3 pass and diffs it
  # against a CPU reference, so kernel changes can be timed and validated without the other six
  # passes in the way.
  add_executable(conv_probe tools/oidn/conv_probe.cpp)
  target_link_libraries(conv_probe PRIVATE Vulkan::Vulkan)
  if(MSVC)
    target_compile_options(conv_probe PRIVATE /W4 /utf-8)
    target_compile_definitions(conv_probe PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
elseif(NOT GLSLC_EXECUTABLE)
  message(WARNING "No glslc, the OIDN denoiser passes cannot be built; smallpt will skip them")
endif()

# ---------------------------------------------------------------------------
# Step 10: vk_viewer -- interactive front end: Win32 window, Vulkan swapchain, ImGui overlay and
# the same clspv-compiled path tracer, accumulating one sample per pixel per frame.
#
# src/smallpt_view.cl is plain OpenCL C, so reqd_work_group_size becomes OpExecutionMode LocalSize
# (no spec constants, no spirv-opt step) and the fast-math flags are the ones measured fastest in
# Step 6. src/smallpt_view_resolve.cl applies exposure, Reinhard and gamma into a packed RGBA8
# buffer, which is copied into a sampled image for the ImGui draw list.
#
# Dear ImGui is vendored in third_party/imgui (MIT, see third_party/imgui/LICENSE.txt).
# ---------------------------------------------------------------------------
if(CLSPV_EXECUTABLE AND WIN32 AND TARGET oidn_shaders)
  set(VIEW_ACCUM_SPV ${CMAKE_CURRENT_BINARY_DIR}/smallpt_view.spv)
  set(VIEW_RESOLVE_SPV ${CMAKE_CURRENT_BINARY_DIR}/smallpt_view_resolve.spv)

  add_custom_command(
    OUTPUT  ${VIEW_ACCUM_SPV}
    COMMAND ${CLSPV_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_view.cl
            -o ${VIEW_ACCUM_SPV} -O3 --use-native-builtins=sin,cos --cl-fast-relaxed-math
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_view.cl
    COMMENT "clspv: src/smallpt_view.cl -> smallpt_view.spv"
    VERBATIM)
  add_custom_command(
    OUTPUT  ${VIEW_RESOLVE_SPV}
    COMMAND ${CLSPV_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_view_resolve.cl
            -o ${VIEW_RESOLVE_SPV} -O3
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_view_resolve.cl
    COMMENT "clspv: src/smallpt_view_resolve.cl -> smallpt_view_resolve.spv"
    VERBATIM)

  set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui)
  add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_win32.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)
  target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
  target_link_libraries(imgui PUBLIC Vulkan::Vulkan)
  if(MSVC)
    target_compile_options(imgui PRIVATE /W0 /utf-8)
    target_compile_definitions(imgui PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()

  add_executable(vk_viewer src/viewer_main.cpp src/png_writer.cpp src/oidn_denoise.cpp
                 ${VIEW_ACCUM_SPV} ${VIEW_RESOLVE_SPV})
  target_link_libraries(vk_viewer PRIVATE Vulkan::Vulkan imgui user32 gdi32)
  add_dependencies(vk_viewer oidn_shaders)

  file(TO_CMAKE_PATH "${VIEW_ACCUM_SPV}" VIEW_ACCUM_SPV_CMAKE)
  file(TO_CMAKE_PATH "${VIEW_RESOLVE_SPV}" VIEW_RESOLVE_SPV_CMAKE)
  target_compile_definitions(vk_viewer PRIVATE
    SMALLPT_VIEW_ACCUM_SPV="${VIEW_ACCUM_SPV_CMAKE}"
    SMALLPT_VIEW_RESOLVE_SPV="${VIEW_RESOLVE_SPV_CMAKE}"
    OIDN_DENOISER_SPV_DIR="${OIDN_SHADER_DIR_CMAKE}"
    OIDN_DENOISER_WEIGHTS_PATH="${OIDN_WEIGHTS_CMAKE}")

  if(MSVC)
    target_compile_options(vk_viewer PRIVATE /W4 /utf-8)
    target_compile_definitions(vk_viewer PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
  set_property(TARGET vk_viewer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

  target_sources(vk_viewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_view.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_view_resolve.cl)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_view.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/src/smallpt_view_resolve.cl
    PROPERTIES VS_TOOL_OVERRIDE "None")
elseif(NOT WIN32)
  message(STATUS "vk_viewer is Windows-only (Win32 + Vulkan surface), skipped")
endif()

