# clspv -> Vulkan compute: engineering notes

> These are the working notes from the development repository this project was extracted from.
> They are kept verbatim because they carry the measurements, the failed attempts and the
> reasoning behind every decision in the code; the file list, tool paths and some numbers refer
> to that repository, not to this release:
>
> - `tools/clspv/bin/clspv.exe` was a local source build and is not tracked here — build clspv
>   yourself and point `CLSPV_ROOT` at it (see the README).
> - `src/smallpt_vec.cl` (the `vec3` → `float4` control kernel) and the other experiment branches
>   are not part of this repository.
> - The kernel comparison numbers were re-measured on this release; see the README for the
>   current table.

One kernel: `out[i] = a[i] + b[i]` over 1M floats.

- **The kernel is OpenCL C (C99 syntax) in `src/vadd.cl`**, compiled to Vulkan-compatible SPIR-V by **clspv**.
- **The host is plain Vulkan C++ in `src/main.cpp`**, with no third-party dependency (no VMA, no glfw): create buffers, create the pipeline, dispatch, read back and validate.
- **clspv is built from source on this machine** (LLVM + Clang + the libclc builtins), and the result lives in `tools/clspv/bin/`; the whole chain runs locally.

Besides this kernel the repository carries a second target, **`smallpt`**: Kevin Beason's
[smallpt](https://www.kevinbeason.com/smallpt/) path tracer running on the same clspv + plain Vulkan
framework, writing a real image file (PNG, encoded in-process by the tiny self-contained
writer in `src/png_writer.cpp`).
See "smallpt: path tracing on the same framework" below.

## Repository layout (development repository)

```
CMakeLists.txt              build script (compiles the kernel with clspv, then the host)
src/vadd.cl                 kernel, OpenCL C
src/main.cpp                host, plain Vulkan
shaders/vadd.spv            pre-built SPIR-V (produced by the clspv in this repo, see the end of this file)
tools/clspv/bin/clspv.exe            clspv built from source (63.5 MB, statically linked LLVM)
tools/clspv/bin/clspv-reflection.exe descriptor reflection tool
tools/glsl_equiv/vadd.comp  GLSL reference implementation (only to validate the host flow without clspv)
tools/glsl_equiv/make_fallback_spv.ps1  script that builds that equivalent placeholder spv
scripts/build_shaders.ps1   regenerate shaders/vadd.spv with clspv
scripts/build.ps1           configure + build + run (-Validation enables the validation layers)
src/smallpt.cl              smallpt kernel in OpenCL C (scene embedded in __constant arrays); the
                            reference implementation, built as smallpt_c.spv for A/B comparison
src/smallpt_cpp.cl          the kernel smallpt.exe actually renders with: C++ for OpenCL, vector math
                            as operator overloading, built as smallpt.spv
                            (clspv --cl-std=CLC++2021 -inline-entry-points + spirv-opt --freeze-spec-const)
src/smallpt_buf.cl          smallpt control kernel, OpenCL C (scene read from the parameter buffer)
src/smallpt_vec.cl          smallpt control kernel, OpenCL C (vec3 replaced by the native float4 vector type)
src/smallpt_lit.cl          fastest clspv kernel, OpenCL C (macros + literal subscripts, the whole scene table becomes immediates)
src/smallpt_lit.comp         fastest GLSL kernel, same source-level rewrite (macros + literal subscripts)
src/smallpt.comp            GLSL equivalent of smallpt (built by glslc, entry point main)
src/smallpt_main.cpp        smallpt host, plain Vulkan, writes PNG, prints time / throughput / device memory
src/png_writer.cpp          PNG encoder (CRC-32, Adler-32, row filters, dynamic-Huffman deflate), no third-party code
src/oidn_denoise.cpp        OpenImageDenoise U-Net executed as Vulkan compute (30 dispatches, no OIDN library)
src/oidn_denoise.h          denoiser entry point, borrows the renderer's device / queue / command pool
src/oidn_pre.comp           OIDN input reorder: autoexposure + PU transfer function
src/oidn_conv.comp          3x3 conv, padding 1, fused ReLU, one pixel per thread (OIDN_CONV=simple)
src/oidn_conv_tiled.comp    same conv with shared-memory tiling and 2x2 register blocking (default)
src/oidn_pool.comp          2x2 max pooling, stride 2
src/oidn_upsample.comp      2x nearest-neighbour upsampling into a concat buffer
src/oidn_concat_copy.comp    copies a skip connection into a channel slice of a concat buffer
src/oidn_post.comp           OIDN output reorder: inverse PU + undo of the autoexposure
src/oidn_pre.cl .. post.cl   the same seven passes as C++ for OpenCL (clspv, the default backend)
                            (every GLSL pass is also built as <name>_h.spv with -DOIDN_HALF_TENSOR=1, see
                             "FP16 tensor storage"; FP16 is the default, OIDN_STORE=fp32 opts out; the
                             C++ for OpenCL passes are built as <name>_cl.spv / <name>_cl_h.spv, see
                             "The same seven passes as C++ for OpenCL kernels"; oidn_conv_tiled.cl is
                             compiled for a 3x2 pixel block over 4 channels by default, plus the
                             oidn_conv_tiled_{2x2g8,4x2g4,4x3g4}_cl{,_h}.spv alternates, see
                             "Register blocking")
src/oidn_denoise_tool.cpp   standalone driver for the denoiser (raw float32 in/out), used for validation
tools/oidn/export_oidn_weights.py   converts OIDN's rt_hdr.tza into the flat float32 blob below
tools/oidn/ref_forward.py   numpy reference implementation of the same network (validation baseline)
tools/oidn/conv_probe.cpp   runs one convolution pass and compares it against a CPU reference (kernel tuning)
tools/oidn/rt_hdr.bin       denoiser weights, float32 (4.65 MB, generated from OIDN, Apache-2.0)
scripts/render_smallpt.ps1  build + render to PNG in one step
scripts/compare_smallpt.ps1 run seven variants in one command and print a performance / memory table
scripts/ppm_diff.py         standard-library-only byte-level PPM comparison (max diff / mean diff / over-threshold share)
renders/                    render output (one PNG preview per variant)
```

> `tools/clspv/bin/*.exe` is about 65 MB in total. If you do not want to commit binaries, delete them and
> point `CLSPV_ROOT` at your own clspv build directory; both `CMakeLists.txt` and
> `scripts/build_shaders.ps1` support that.

## Environment

| Requirement | Version measured on this machine |
| --- | --- |
| Vulkan SDK | 1.4.357 (`C:\VulkanSDK\1.4.357.0`) |
| Compiler | MSVC 14.44 (VS 2022 Community) |
| CMake | 3.31.12 (`C:\Program Files\CMake\bin\cmake.exe`, not on PATH) |
| clspv | built from source, see the next section; the binaries live in `tools/clspv/bin/` |

## Building clspv from source (steps measured on this machine)

clspv is not part of the standard Vulkan SDK, and it has to be built together with LLVM/Clang. The
dependency revisions are pinned by `deps.json` in the clspv repository:

| Dependency | Commit |
| --- | --- |
| llvm-project | `aab5db6ab0702d190dbf0f3f7407d87f570c44f6` |
| SPIRV-Headers | `b824a462d4256d720bebb40e78b9eb8f78bbb305` |
| SPIRV-Tools | `fb7471844504abb16b732dc4fc0837119a32ec24` |

Source layout (all build output was kept in `D:\clspv_build` so the project directory stays clean):

```
D:\clspv_build\src\clspv\                        clspv sources
D:\clspv_build\src\clspv\third_party\llvm\               llvm-project repository root (llvm/ clang/ libclc/ inside)
D:\clspv_build\src\clspv\third_party\SPIRV-Headers
D:\clspv_build\src\clspv\third_party\SPIRV-Tools
```

The normal route (when git is available) is `python utils/fetch_sources.py` to fetch the dependencies,
then:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release `
      -DCLSPV_BUILD_TESTS=OFF -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_WERROR=OFF
cmake --build build --config Release --target clspv clspv-reflection
```

### Pitfall 1: the Visual Studio generator cannot build the libclc builtins

clspv needs a builtins library called `libclc.bc` (the OpenCL C builtin function implementations).
It is produced by the LLVM runtimes build through a **custom CMake language called `CLC`** that
compiles `libclc/*.cl`. **The Visual Studio generator does not support custom language rules**:
in the generated `libclc-spirv64-unknown-vulkan-clc.vcxproj` all 209 `.cl` files appear as
`<None Include=...>`, no `.o` file is ever produced, and the link then fails with

```
LINK : fatal error LNK1181: cannot open input file "...\clc.dir\Release\clc_mul_hi.o"
MSB8066
```

That is a generator capability problem, not a missing file, and rebuilding does not fix it. The
workaround is to **build libclc separately with Ninja** and let clspv use that pre-built builtins
library through `CLSPV_EXTERNAL_LIBCLC_DIR`:

```powershell
# libclc is an independent LLVM sub-project and builds outside the VS generator (Ninja supports CLC)
cmake -G Ninja -S <libclc source> -B build_libclc_spirv64 `
      -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_C_COMPILER=clang.exe -DCMAKE_CXX_COMPILER=clang.exe `
      -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY `
      -DCMAKE_AR=llvm-ar.exe -DCMAKE_RANLIB=llvm-ranlib.exe `
      -DCMAKE_CLC_COMPILER=clang.exe `
      -DLLVM_CMAKE_DIR=<LLVM build tree>/lib/cmake/llvm `
      -DLLVM_TOOLS_BINARY_DIR=<LLVM build tree>/Release/bin `
      -DLLVM_TOOL_opt=<...>/opt.exe -DLLVM_TOOL_llvm-link=<...>/llvm-link.exe `
      -DLIBCLC_TARGET=spirv64-unknown-vulkan `
      -DCMAKE_INSTALL_PREFIX=D:\clspv_build\libclc_prebuilt
cmake --build build_libclc_spirv64 --target install
# repeat for spirv32-unknown-vulkan
```

A few extra obstacles hit along the way:

- `CMAKE_CLC_COMPILER` defaults to `CMAKE_C_COMPILER` (that is `cl.exe`), so it must be pointed at
  `clang.exe` explicitly, otherwise the `.cl` files cannot be compiled.
- This machine has **no Windows SDK bin directory**, hence no `rc.exe`. The Ninja generator performs
  a compiler self-test through `cmake -E vs_link_exe` and dies at `RC Pass 1 ... no such file or
  directory`. Setting `-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY` makes it build a static
  library instead of an exe, which sidesteps the check (libclc itself compiles no C/C++ sources).
- With clang as host compiler, CMake's `Windows-Clang.cmake` still insists on `CMAKE_RC_COMPILER`;
  point it at any existing exe (clang itself here), since no `.rc` file is ever compiled.
- `find_package(LLVM)` overwrites the plain variable `LLVM_TOOLS_BINARY_DIR`, so `find_program(opt ...)`
  cannot find `opt.exe` and the build fails with
  `libclc toolchain incomplete - missing tool opt!`. Pre-seeding `LLVM_TOOL_opt` /
  `LLVM_TOOL_llvm-link` into the cache solves it.

### Pitfall 2: keep the LLVM Native target after plugging in an external builtins library

Once libclc is external, clspv no longer sets `CLSPV_LLVM_TARGETS_TO_BUILD=Native`, so
`LLVM_TARGETS_TO_BUILD` becomes empty and every LLVM target library is rebuilt from scratch. Pass it
back explicitly when reconfiguring, and clear the runtimes:

```powershell
cmake -G "Visual Studio 17 2022" -A x64 -S D:\clspv_build\src\clspv -B D:\clspv_build\build `
      -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=<python> `
      -DCLSPV_BUILD_TESTS=OFF -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_WERROR=OFF `
      -DCLSPV_EXTERNAL_LIBCLC_DIR=D:/clspv_build/libclc_prebuilt `
      -DCLSPV_LLVM_TARGETS_TO_BUILD=Native -DLLVM_RUNTIME_TARGETS=
cmake --build D:\clspv_build\build --config Release --target clspv clspv-reflection
```

Result: `D:\clspv_build\build\bin\Release\clspv.exe` (63.5 MB) and `clspv-reflection.exe`.
The first full build took roughly 40 minutes on this machine (12 cores / 15.7 GB); a configuration
change that forces an incremental rebuild takes about 35 minutes.

## Build and run

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\Release\vkcompute.exe
```

`CMakeLists.txt` searches `PATH`, then `$env:CLSPV_ROOT\bin`, then `tools/clspv/bin`. When it finds
clspv it adds a custom command that builds `src/vadd.cl` into `build/vadd.spv`; only when nothing is
found does it fall back to the pre-built `shaders/vadd.spv` (which was also produced by clspv, so
behaviour is the same either way).

Measured output (Intel UHD Graphics, Vulkan 1.4.323 vendor driver):

```
[info] device: Intel(R) UHD Graphics (queue family 0)
[info] SPIR-V: E:/code/clspv_test/build/vadd.spv (1308 bytes)
[info] enabled variablePointersStorageBuffer / variablePointers
[info] dispatch: 16384 workgroups x 64 threads = 1048576 work items
[info] elapsed: 0.783 ms
[info] first 4 elements: gpu = 0 2.5 5 7.5
[info] last element: gpu = 101.5 (expected 101.5)

PASS: all 1048576 elements match (max |err| = 0)
```

Notes:

- To enable the validation layers: `$env:CLSPV_EXAMPLE_VALIDATION="1"` before running (requires
  `VK_LAYER_KHRONOS_validation` to be installed). One run with the layers enabled produced
  **no validation errors**; only the elapsed time grew to 1.2-4.9 ms because of layer overhead.
- The SPIR-V path is baked into the executable at compile time (`VADD_SPV_PATH`), so it is found
  from any working directory; `vkcompute.exe <spv path>` overrides it.
- Besides the Intel vendor driver this machine also exposes a `Microsoft Direct3D12 (Dozen)` device.
  Device selection prefers discrete GPU > integrated GPU > other > CPU, then the higher Vulkan
  version, so the Intel vendor driver wins.

## smallpt: path tracing on the same framework

The same `CMakeLists.txt` and the same "clspv compiles the kernel, host is plain Vulkan" flow, for a
second target: Kevin Beason's smallpt (99 lines of C++, unbiased Monte Carlo path tracing, Cornell
box scene).

```powershell
cmake --build build --config Release --target smallpt
.\build\Release\smallpt.exe            # defaults: 640x480 / 64 spp / writes renders\smallpt.png
.\build\Release\smallpt.exe build\smallpt.spv 1024 768 256 renders\smallpt.png
```

Or in one step: `pwsh -File scripts\render_smallpt.ps1 -Width 1024 -Height 768 -Spp 256`
(it builds and renders; there is no conversion step and no Python dependency).

Arguments are `<spv> <width> <height> <spp> [output.png] [entryPoint]`; the host prints elapsed
time and throughput (Mpaths/s). The output format follows the extension: `.png` (the default) or
`.ppm`, which is kept because a raw PPM is the simplest thing to diff byte by byte with
`scripts/ppm_diff.py`. `spp` means "samples per subsample" here, so the actual number of
paths per pixel is 4x spp (2x2 subsamples). The entry point defaults to `smallpt`, which is what a
clspv module uses; a glslc-built GLSL module must pass `main` explicitly.

Measured (Intel UHD Graphics, SPIR-V built with `-O3`):

| Resolution / spp | Total paths | Elapsed | Throughput |
| --- | --- | --- | --- |
| 160x120 / 8 spp | 0.61 M | 67 ms | 9.1 Mpaths/s |
| 480x360 / 32 spp | 22.1 M | 1.46 s | 15.1 Mpaths/s |
| 1024x768 / 256 spp | 805 M | 52.4 s | 15.4 Mpaths/s |
| 1024x768 / 1024 spp | 3.2 G | 397.7 s | 8.1 Mpaths/s |

The last row loses half its throughput to **thermal throttling**: the integrated GPU ran continuously
for more than six minutes and clocked down. It is not a kernel problem. At small resolutions each
pixel traces only a few paths, so startup overhead dominates and the 160x120 throughput is lower.

### Differences from the original smallpt (all needed to run correctly in OpenCL C / float / on a GPU)

1. **Recursion replaced by iteration.** The original `radiance()` is recursive and OpenCL C forbids
   recursion. Here the scheme is "throughput `thr` plus a loop": on every hit, `thr * e` (emission)
   is accumulated into `L`, then the material updates `thr` and the ray continues. Past depth 5
   Russian roulette decides whether to continue (based on `max(f)`, compensating with `f /= p`).
   The returned `L` has the same expectation as the recursive sum of the original.
2. **Branch choice in the glass sphere.** The original adds **both** the reflected and the refracted
   sub-path when depth <= 2 and picks one by Fresnel probability above that. This version always
   picks one by probability (`P = 0.25 + 0.5*Re`, with weights `Re/P` and `Tr/(1-P)` as in the
   original). Unbiased, slightly higher variance.
3. **float plus a private LCG.** The original uses `double` and `erand48`. Here every thread has its
   own LCG (`s = s*1664525u + 1013904223u`, top 24 bits). That choice has a cost, see the next item.
4. **The six walls are analytic planes instead of radius-1e5 spheres.** This is the only place where
   the scene geometry changed, and it comes from a problem measured here: the original saves code by
   using "a radius-1e5 sphere locally looks like a plane", but in float the intersection computes
   `det = b*b - c` by subtracting two numbers of order 1e10, so `t` is off by about 1e-2. Diffuse
   rays then treat the same wall as a new hit, or hit the wrong place, and the image shows **blocky
   patches and striped artifacts**. Raising `EPS` from 1e-4 to 1e-2 only removed part of them (stripes
   remained), so the walls became planes (`PLANES[6*7]`, `n.p = d`): same position, colour and normal
   orientation as the original, only the invisible curvature is gone, and float never self-intersects.
   The three real spheres (mirror r=16.5, glass r=16.5, light r=600) still use sphere intersection;
   their radii are small enough for float.
5. **The camera ray is pushed forward by 140.** The original does
   `radiance(Ray(cam.o + d*140, d.norm()), 0, Xi)` because the camera sits outside the box
   (z=295.6 while the front wall is at z=170), so the origin must start inside; otherwise the first
   ray hits the **black front wall** (`Frnt`, albedo=0). The first version here missed this step and
   produced a **completely black image**, which took a while to find.
6. **Output row order.** The original pixel index is `i = (h-y-1)*w+x`, meaning the first row of the
   file is the **top** of the image; the kernel converts the row index with `(h-1-py)` into the
   camera's "up" direction, otherwise the image is flipped vertically.
7. **Sampling follows the original**: 2x2 subsamples, tent-distributed jitter inside each subsample
   (`dx = r<1 ? sqrt(r)-1 : 1-sqrt(2-r)`) plus spp paths, and the host finally applies
   `pow(v, 1/2.2)` gamma when writing the image.
8. **The per-subsample clamp is 20, not 1.** The original clamps each subsample to [0,1] before
   averaging so the area light cannot blow up a pixel, but that clamp is a *tone map applied before
   the average*, which biases the estimate downwards at low sample counts: at 1 spp almost all of
   the scene's energy arrives in a few paths that hit the light, and clipping those to 1.0 drops
   ~70% of the frame's energy (measured mean 0.062 against a converged 0.326). `CLAMP_MAX 20.0f`
   keeps the fireflies reasonabe while leaving the estimator essentially unbiased: the 1 spp mean
   is 0.325, the 16 spp mean 0.326 and the 256 spp reference 0.326, so spp no longer changes
   brightness. The denoiser is meant to receive an unbiased noisy estimate, which is why this
   matters once a denoiser is in the chain.

### Scene (hard-coded in `src/smallpt.cl`; the host passes no geometry)

| Surface | Position | Colour |
| --- | --- | --- |
| Left wall | x = 1 | (0.75, 0.25, 0.25) red |
| Right wall | x = 99 | (0.25, 0.25, 0.75) blue |
| Back wall | z = 0 | (0.75, 0.75, 0.75) |
| Front wall | z = 170 | (0, 0, 0) pure black |
| Floor | y = 0 | (0.75, 0.75, 0.75) |
| Ceiling | y = 81.6 | (0.75, 0.75, 0.75) |
| Mirror sphere | (27, 16.5, 47), r=16.5 | (0.999, 0.999, 0.999), SPEC |
| Glass sphere | (73, 16.5, 78), r=16.5 | (0.999, 0.999, 0.999), refractive index 1.5 |
| Light sphere | (50, 681.33, 81.6), r=600 | emission (12, 12, 12) |

Camera: `camO = (50, 52, 295.6)`, `camD = normalize((0, -0.042612, -1))`,
`cx = (aspect*0.5135, 0, 0)`, `cy = normalize(cross(cx, camD)) * 0.5135`, ray origin `camO + d*140`.

The **black square** inside the mirror sphere is not a bug: the original notes explain that the
mirror shows the front wall of the box, which is deliberately painted black to stand for emptiness.

### Known trade-offs

- No explicit light sampling (the original does not do it either), so a small area light is noisy;
  256 spp is usable, 1024 spp is cleaner.
- No acceleration structure (again like the original): every intersection tests 9 primitives. This
  scene is made quiet on a GPU mainly by running a huge number of threads in parallel.
- The PNG is encoded in-process (`src/png_writer.cpp`): CRC-32, Adler-32, the per-row filters and
  a dynamic-Huffman deflate block are all hand-written, so the host still links nothing beyond
  Vulkan and the C++ standard library. On the 640x480 / 256 spp reference render the output is
  slightly smaller than what zlib level 9 produces for the same filtered data. A `.ppm` output
  path is still accepted for byte-level diffing with `scripts/ppm_diff.py`.
- **The host must enable `shaderInt64`.** Because storage buffers need 64-bit byte addressing, a
  clspv module declares `OpCapability Int64`. Without `VkPhysicalDeviceFeatures::shaderInt64` the
  Intel driver still runs, but the validation layers report
  `VUID-VkShaderModuleCreateInfo-pCode-08740` (the Int64 capability requires the matching feature).
  `src/smallpt_main.cpp` queries `vkGetPhysicalDeviceFeatures` and enables it when supported.

### The GLSL equivalent, and a performance / device-memory comparison against clspv

`src/smallpt.comp` is the **GLSL version** of the same smallpt: it corresponds line by line to
`src/smallpt.cl` (same 6 analytic planes plus 3 spheres, iterative trace, Russian roulette past depth
5, Fresnel-probability branch selection for the glass sphere, same LCG and same seed, 2x2 subsample
tent jitter, clamp each subsample before averaging). Only the language syntax changes: `vec3` becomes
the builtin vector type, `__constant` arrays become `const vec4[6]` / `const vec3[6]` /
`const vec4[3]`, and `get_global_id` becomes `gl_GlobalInvocationID`.

The variants differ only in "which compiler, and where the scene data lives":

| Variant | Source | Compiler | Scene data | Entry point | SPIR-V size |
| --- | --- | --- | --- | --- | --- |
| A | `src/smallpt.cl` | clspv `-O3` | `__constant` arrays | `smallpt` | 93.5 KiB |
| B | `src/smallpt_buf.cl` | clspv `-O3` | `__global` parameter buffer | `smallpt` | 93.9 KiB |
| C | `src/smallpt.comp` | `glslc -O` | GLSL `const` arrays | `main` | 12.2 KiB |
| D (fastest clspv with the scene in a buffer) | `src/smallpt_buf.cl` | clspv `-O3 --use-native-builtins=sin,cos --cl-fast-relaxed-math` | `__global` parameter buffer | `smallpt` | 25.4 KiB |
| E (fastest clspv overall) | `src/smallpt_lit.cl` | same options as D | compile-time constants (the whole table folds into immediates) | `smallpt` | 24.4 KiB |
| F (fastest overall) | `src/smallpt_lit.comp` | `glslc -O` | compile-time constants | `main` | 14.8 KiB |
| G | `src/smallpt_cpp.cl` | clspv `-O3 --cl-std=CLC++2021 -inline-entry-points` (then `spirv-opt --freeze-spec-const`) | `__constant` arrays | `smallpt` | 94.7 KiB |
| G+ (the default build today) | `src/smallpt_cpp.cl` | G's options plus `--use-native-builtins=sin,cos --cl-fast-relaxed-math`, literal subscripts and hit data carried with the hit | compile-time constants (the whole table folds into immediates) | `smallpt` | 37.7 KiB |

G is the same smallpt as A, written in **C++ for OpenCL**: the vector math is operator overloading
(`a + b`, `a * s`, `dot`, `cross`, `normalize`) instead of `add3` / `scl3` / `dot3` free functions, and
the intersections return through a `float &` reference. It renders a **byte-identical image** to A at
both 480x360/32spp and 1024x768/1024spp, and runs within 2% of A. Details, including the two
C++-mode quirks that have to be handled on the host side, are in "Variant G" below.

G+ is G with the D and E optimizations folded in, so `build\smallpt.spv` is now the fastest clspv
configuration in the repository: 263.3 ms against E's 253.5 ms and A's 1653.6 ms at 480x360/32spp
(interleaved medians, see "Which one `smallpt.exe` renders with today"). G itself stays buildable
as `smallpt_exact` (plain `-O3`), which is still byte-identical to A.

**Which one `smallpt.exe` renders with today:** G, with every kernel-level optimization collected in
this repository, built for speed rather than for bit-exactness. `CMakeLists.txt` compiles
`src/smallpt_cpp.cl` into `build/smallpt.spv` with

```
clspv src/smallpt_cpp.cl -o smallpt_fast_raw.spv -O3 ^
      --cl-std=CLC++2021 -inline-entry-points ^
      --use-native-builtins=sin,cos --cl-fast-relaxed-math
spirv-opt smallpt_fast_raw.spv -o build/smallpt.spv ^
      --set-spec-const-default-value "0:8 1:8 2:1" --freeze-spec-const
```

so the module carries all four optimizations at once:

1. the operator-overloaded vector math of G (`Vec3` + `- * dot cross normalize`);
2. the literal subscripts of E: the six walls and three spheres are written out by the
   `SCENE_TRY_PLANE` / `SCENE_TRY_SPHERE` macros, so every `PLANES[i*7+k]` / `SPHERES[i*16+k]` is a
   compile-time subscript and the scene tables do not exist in the module at all (0 `OpVariable`
   for them, 0 `Private` accesses);
3. the hit information travelling with the hit: the winning primitive's normal or centre, albedo,
   emission and material land in plain locals inside the macro block, so `trace()` never looks
   anything up by a run-time id. Keeping them in scalars instead of a hit struct passed by
   reference matters: a struct is materialized in Function storage (68 `Function` `OpVariable`,
   182 `OpLoad`; with the macro block it is 9 and 44);
4. native `sin`/`cos` and relaxed math from D: `cos`/`sin` become hardware ops instead of the
   inlined libclc polynomials, division loses the denormal/range guards.

Measured on this machine, render only (`SMALLPT_NO_DENOISE=1`), median of three interleaved runs
(the same binary drifts 15% with the GPU clock, so never compare single runs):

| kernel (module) | 320x240/8spp | 480x360/32spp | 640x480/16spp |
| `smallpt_c.spv` -- OpenCL C reference (101212 B) | 178.5 ms | 1653.6 ms | 1344.9 ms |
| `smallpt_exact.spv` -- C++ for OpenCL, plain `-O3` (111016 B) | 163.0 ms | 1170.7 ms | 960.9 ms |
| `smallpt.spv` -- **C++ for OpenCL, default build (38620 B)** | **93.1 ms** | **263.3 ms** | **189.6 ms** |

That is 6.3x faster than the OpenCL C reference at 480x360/32spp (84.0 Mpaths/s vs 13.4 Mpaths/s)
and 7.1x at 640x480/16spp (103.7 Mpaths/s vs 14.6 Mpaths/s); the module shrinks from 128876 B (plain
`-O3`, no literal subscripts, no fast flags) to 38620 B, and the whole scene has become immediates.

**Precision.** `smallpt_exact` renders bit-identical to `src/smallpt.cl`, which is what makes the
rewrite checkable: at 320x240/8spp the PPM payloads are byte-identical (0 of 230400 samples differ)
and so are the PNGs SHA256 `C42D2F46...E3096` (with the denoiser) plus the `SMALLPT_DUMP_AUX`
albedo / normal dumps. The default fast build trades that away, as intended:

| pair | measured difference |
| exact vs default at 320x240/8spp (PPM, 8-bit) | 9.43% of the samples differ, mean 4.3/255, max 255 |
| exact vs default at 640x480/16spp, after the denoiser (PNG) | **47.2 dB PSNR**, mean 0.49/255, max 34 |

So a 1/8 spp frame differs noticeably, a denoised 16 spp frame does not.

```
cmake --build build --config Release --target smallpt
build\Release\smallpt.exe build\smallpt.spv       640 480 16 renders\a.png   # default, fast
build\Release\smallpt.exe build\smallpt_exact.spv 640 480 16 renders\b.png   # bit-identical to C
build\Release\smallpt.exe build\smallpt_c.spv     640 480 16 renders\c.png   # OpenCL C reference
```

D is the configuration that "wins back the 5x" of clspv (see "Why clspv is slow" below); in CMake it
is the target `smallpt_native`, producing `build\smallpt_native.spv`. It uses exactly the source of
B, only the compiler options differ.

E goes one step further: every subscript of the scene tables becomes a compile-time constant (macros
with literal subscripts, expanded by hand), which drops the Private accesses from 52 to 0 and
`OpLoad` from 58 to 6, reaching 197.7 ms at 480x360/32spp -- 36% faster than D, and **the first
clspv build to beat GLSL (220.8 ms)**. CMake target `smallpt_lit`, output `build\smallpt_lit.spv`;
details in "How to make clspv support the constant address space" below.

That 197.7 ms is a clock-dependent number, and the default build above is the same set of tricks
measured again: with the extra auxiliary G-buffer and `CLAMP_MAX` (which add an extra primary ray
per pixel), E's OpenCL C recipe re-measured today lands at 253.5 ms and the C++ build at 263.3 ms
(median of three interleaved runs each, 4% apart, i.e. the language mode costs nothing measurable).
The earlier session recorded 190.9 / 190.5 / 211.7 ms for the same module, and today's first
interleaved round measured 232.6 / 233.2 ms, so the ~40% gap to 197.7 ms is the GPU clock state
rather than a regression in the kernel -- which is why every comparison here is interleaved and
reported as a median.

The host is the same `src/smallpt_main.cpp` in every case, with one extra optional 6th argument,
the entry point name (default `smallpt`; GLSL modules must pass `main`):

```powershell
cmake --build build --config Release --target smallpt   # builds smallpt.spv (C++ for OpenCL, fast) plus
                                                        # smallpt_exact.spv, smallpt_c.spv, smallpt_cpp.spv,
                                                        # smallpt_buf.spv, smallpt_glsl.spv, smallpt_native.spv,
                                                        # smallpt_lit.spv, smallpt_lit_glsl.spv
.\build\Release\smallpt.exe build\smallpt.spv      1024 768 256 out_G.png   # default: C++ for OpenCL, fast
.\build\Release\smallpt.exe build\smallpt_exact.spv 1024 768 256 out_Gx.png  # same source, bit-identical to A
.\build\Release\smallpt.exe build\smallpt_c.spv    1024 768 256 out_A.png   # the OpenCL C reference
.\build\Release\smallpt.exe build\smallpt_buf.spv  1024 768 256 out_B.png smallpt
.\build\Release\smallpt.exe build\smallpt_glsl.spv 1024 768 256 out_C.png main
.\build\Release\smallpt.exe build\smallpt_lit.spv  1024 768 256 out_E.png smallpt
```

Or one command that runs all of them and prints the table:
`pwsh -File scripts\compare_smallpt.ps1 -Width 1024 -Height 768 -Spp 256`
(it also uses `scripts/ppm_diff.py` to compare the C and F images).

The host also reports device memory: it looks for `VK_EXT_memory_budget`, calls
`vkGetPhysicalDeviceMemoryProperties2`, takes the largest device-local heap and prints
"used / budget / size".

#### Measured data (Intel UHD Graphics, Vulkan 1.4.323 vendor driver, all in the same run)

1024x768 / 64 spp (201.33 M paths, two rounds each):

| Variant | Round 1 | Round 2 | Throughput |
| --- | --- | --- | --- |
| A clspv `__constant` | 13591.2 ms | 13146.6 ms | 15.3 Mpaths/s |
| B clspv scene in a buffer | 10066.7 ms | 9902.3 ms | 20.3 Mpaths/s |
| C GLSL `glslc -O` | 2014.1 ms | 2084.9 ms | 96.6-100.0 Mpaths/s |

480x360 / 32 spp (22.12 M paths, A/B/C interleaved three rounds each to avoid clock drift):

| Variant | Three rounds | Average |
| --- | --- | --- |
| A clspv `__constant` | 1454.1 / 1437.2 / 1444.9 ms | 1445 ms (15.3 Mpaths/s) |
| B clspv scene in a buffer | 1105.4 / 1114.4 / 1126.2 ms | 1115 ms (19.8 Mpaths/s) |
| C GLSL `glslc -O` | 218.3 / 226.1 / 211.4 ms | 219 ms (101 Mpaths/s) |

So **the GLSL build is about 6.6x faster than the clspv build (A to C), and about 5.1x faster than
the clspv build with the scene in a buffer**; inside clspv, moving the scene from `__constant` to a
buffer wins back about 1.3x (480x360) to 1.35x (1024x768).

> A single serial 1024x768 / 256 spp run once produced A 79032.7 ms / B 39435.4 ms / C 8334.7 ms;
> the A number was an outlier caused by long-run thermal throttling (the same variant held a stable
> 13.2-13.6 s at 64 spp, roughly 53 s extrapolated, and C about 8.3 s), which is why the table above
> uses interleaved rounds. On an integrated GPU, never read a single measurement as an absolute
> value; read the relative order inside one round.

The production-image setting (1024x768 / 1024 spp, 3.2 G paths) is a long single run on both sides:

| Variant | Elapsed | Throughput | Artifact |
| --- | --- | --- | --- |
| A clspv | 397.7 s (throttled down to 8.1 Mpaths/s) | 8.1 Mpaths/s | `renders/smallpt.png` |
| C GLSL | **33.2 s** | 96.9 Mpaths/s | `renders/smallpt_glsl.png` |

The GLSL build never throttles (it only runs half a minute), which is why the gap grows to 12x at
this setting.

#### Device memory: identical for all three

At 1024x768: the image buffer is 1024x768x4x4 B = 12.00 MiB and the parameter buffer is 316 B (4
uints plus 75 floats of scene data, only B uses them, but all three share the same host),
and `VK_EXT_memory_budget` reports "largest device-local heap used 12.45 MiB" -- A, B and C give the
exact same number (the extra 0.45 MiB is the driver's own allocations).

The conclusion is clear: **device memory usage depends only on resolution, not on the compiler and
not on whether the scene lives in constants or in a buffer.** The SPIR-V module size (93.5 KiB vs
12.2 KiB) only affects host memory and driver compile time, not device memory. At 480x360 all three
agree as well (about 2.6 MiB).

#### Why clspv is slow: `sin`/`cos` fall back to the libclc software implementation (the root cause)

Counting operations with `spirv-dis` and splitting them per basic block (`D:\clspv_build\out\analyze_blocks.py`):

| Metric | clspv (A/B) | glslc (C) |
| --- | --- | --- |
| Total operations | 5258 | 730 |
| Operations inside blocks | 4191 | 476 |
| `OpLabel` | 488 | 90 |
| `OpPhi` | 500 | 45 |
| Merge blocks holding only `phi`+`branch` | 118 | 29 |
| `OpSelectionMerge` (an `if` in the source) | 230 | 26 |
| `OpSelect` | 248 | 0 |
| `OpBitcast` | 247 | 1 |
| `OpBitwiseAnd` / `OpShiftRightLogical` | 298 / 204 | 0 / 0 |
| Largest basic blocks | 196 / 196 / 80 / 64 ops | 32 / 24 / 21 ops |
| `OpAccessChain %_ptr_Private` (dynamic subscripts) | A: 52, B: 0 | 0 (only 10 constant-subscript accesses into `Function` storage) |

**The two largest basic blocks, 196 operations each, are the software `sin` and `cos`.** Line by line
(`D:\clspv_build\out\dump_big_block.py`, `_base.spvasm` lines 1673 / 3362):

```
Bitcast float->uint, IAdd 0xFF800000, ULessThan 0xFF800001   ; denormal/inf test
BitwiseAnd 0x7FFFFF (mantissa mask) + BitwiseOr 0x3F000000 (build 0.5) ; denormal renormalization
ShiftRightLogical 23, And 0x7FFFFF, ExtInst UMin 231, IAdd 24 ;
ShiftLeftLogical 23, ConvertFToU                              ; quadrant index k = floor(x*2/pi)
IMul 4266746795 + UMulExtended->CompositeExtract 1            ; 64-bit multiply, high half
IMul 1011060801 / 3680671129 + IAdd + ULessThan              ; three-constant multiply pipeline
Fma chain: 1.57079625(pi/2), 0.000483751297, 7.54978942e-08, 0.636619747(2/pi)
%1471 = And %uint_3                                           ; select quadrant by k&3
```

In other words clspv **inlines the whole libclc implementation of `sin()`/`cos()`** (polynomial plus
quadrant reduction plus denormal/NaN branches) instead of mapping them to the hardware
`OpExtInst Sin/Cos`. The two call sites in the source (`cos(r1)` and `sin(r1)`) eat 196 operations
each and drag their internal loops and selects along -- that is the main source of the 230
`OpSelectionMerge`, 118 pure merge blocks and 500 `OpPhi`.

A smaller second layer: **division carries denormal/range guards**. `OpFDiv` is wrapped into a
scaling sequence `Select(0.0625, 16777216)` + `FMul` + `FDiv` + `FMul` (the block around line 464
of `t_native.spvasm`), because OpenCL requires correctly-rounded division. The GLSL side has a bare
`OpFDiv` only.

#### Fix: two clspv options, no source change needed

- `--use-native-builtins=sin,cos` -- officially "use a native implementation instead of the one in
  the builtins library", which maps `sin`/`cos` to hardware instructions (equivalent to OpenCL's
  `native_sin`/`native_cos`, no longer guaranteeing the ULP accuracy of the non-native version).
- `--cl-fast-relaxed-math` -- drop the division scaling and denormal guards, allow multiply-add
  contraction (`OpExtInst Fma`).

Measured (480x360 / 32 spp, three interleaved rounds in the same run):

| Variant | Module | ops | blocks | Phi | Average time |
| --- | --- | --- | --- | --- | --- |
| A baseline (`smallpt.cl`, `__constant` scene) | 95716 B | 5217 | 488 | 500 | 1598.9 ms |
| A + `--use-native-builtins=sin,cos` | 32840 B | 1729 | 88 | 184 | 455.5 ms |
| A + native + `--cl-fast-relaxed-math` | 25628 B | 1253 | 88 | 184 | 373.4 ms |
| B scene in a buffer + native | 33256 B | 1767 | 88 | 184 | 372.6 ms |
| **B scene in a buffer + native + fast** (best) | 26044 B | 1291 | 88 | 184 | **302.2 ms** |
| C GLSL (glslc) | 12520 B | 730 | 90 | 45 | 258.1 ms |

**5.5x down to 1.17x**, the module from 93.5 KiB to 25.4 KiB, Phi from 500 to 184, basic blocks from
488 to 88 (GLSL has 90). The earlier conclusion "all those CLI switches do nothing" holds because
they all change clspv's own passes -- while the bloat happens earlier, at the moment the clang
frontend inlines the libclc `sin`/`cos` bitcode.

The production image (1024x768 / 1024 spp) rerun with the best variant: **39.4 s / 81.8 Mpaths/s**
(`renders/smallpt_clspv_native.png`; the baseline at the same 1024 spp took 397.7 s, and extrapolating
the 64 spp steady state of 13.2 s by 16 gives about 211 s, GLSL 33.2 s).

Hypotheses that were ruled out (all measured, do not retry):

| Attempt | Result |
| --- | --- |
| clspv `-O0 / -O1 / -O2 / -O3` | 11.3 / 14.1 / 14.0 / 14.1 Mpaths/s, **the optimization level barely matters** |
| clspv `--cl-fast-relaxed-math` | 14.5 Mpaths/s, only +10% (88920 B) |
| `#pragma clang loop unroll(full)` on the two primitive loops of `intersect()` | module size and the 52 Private accesses **unchanged, not one** |
| `--hack-phis` (scalarize struct-typed Phis early) | ops 5217 -> 5239, still 52 Private accesses, size grew by 536 B |
| `--no-inline-single` (do not inline single-call-site functions) | ops 5217 -> 5215, no change |
| `--hack-inserts` / `--no-vec3-to-vec4` | byte-identical to the baseline (95716 B) |
| `--enable-cse-in-ir-translator`, `--enable-cse-in-legalizer`, `--hack-block-order`, `--no-dra`, `--no-zero-allocas` (alone and combined) | **all byte-identical to the baseline** (95716 B, ops=5217, phi=500, label=488) |
| `#pragma clang loop unroll(disable)` (forbid unrolling the two primitive loops) | 89.0 KiB, ops 5217->4871, Private 52->24, but 1573 ms (baseline 1577-1587 ms), **no speed change** |
| `spirv-opt -O` / `-Os` post-processing the clspv module | Private 52->0, Phi 500->393, ops 5055, 1472 ms (baseline 1587 ms), **only +7%** |
| Scene moved from `__constant` arrays to a `__global` buffer (variant B) | Private accesses 52 -> 0, +30% speed, still 5x behind GLSL |
| `vec3` from `struct{float x,y,z;}` to OpenCL's native `float4` (`src/smallpt_vec.cl`) | 44 `%v4float` float operations emitted (baseline 0), ops 5042, Phi 464, 1551 ms (baseline 1624.8 ms in the same round), **only +5%** |

Conclusion: **none of clspv's own passes can prevent this inlining** (`--hack-phis`, `--no-inline-single`,
`--hack-inserts`, `--no-vec3-to-vec4` and similarly named switches all do nothing; `-O` levels, the
`unroll` pragma, `spirv-opt` post-processing and the native vector type move only 0-7%). The bloat
happens **before** those passes, at the moment the clang frontend inlines the libclc `sin`/`cos`
bitcode. The knob that actually works controls whether the builtin goes through the library:

| Attempt | Result |
| --- | --- |
| `--use-native-builtins=sin,cos` | ops 5217->1729, blocks 488->88, Phi 500->184, **1599->456 ms (3.5x)** |
| plus `--cl-fast-relaxed-math` | ops->1253, **456->373 ms** |
| plus moving the scene to a `__global` buffer (B) | **->302 ms, only 17% behind GLSL's 258 ms** |

#### Do we still need to patch the clspv source?

**No.** The two options above already brought 5.5x down to 1.17x; adding `src/smallpt_lit.cl` (literal
subscripts, Private accesses down to zero) gives another 1.42x, taking clspv from 1586.8 to 222.0 ms,
while GLSL with the same treatment (`src/smallpt_lit.comp`) goes from 265.5 to 215.0 ms -- **after
both rewrites GLSL is still about 5% ahead**, which corresponds to the `vec3` scalarization
residual seen as ops 833 vs 1232. That last gap is not worth patching clspv for.

If the residual ever needs chasing, the candidates ordered by measured leverage are:

| Approach | Measured gain |
| --- | --- |
| Scene table subscripts as compile-time constants (`src/smallpt_lit.cl`, Private 52->0, OpLoad 58->6) | 1.57x (310->198 ms) |
| Scene from `__constant` to a `__global` buffer (`src/smallpt_buf.cl`) | 1.25x (373->302 ms) |
| `spirv-opt -O` post-processing (Private->0, Phi 500->393) | +7% |
| `vec3` as the native `float4` (`src/smallpt_vec.cl`) | +5% |
| source-level `unroll(disable)` | 0% |

If the source were to be touched, the most sensible cut is no longer "reduce ops" but **stop the
clang frontend from unconditionally inlining the libclc trigonometric/exponential bitcode**
(`lib/Builtins.cpp` / `lib/FrontendPlugin.cpp`, or simply making `--use-native-builtins` the default
policy), and second, embedding `spirv-opt -O` after `SPIRVProducerPass`. Neither touches the long and
deep chain of "the clang ABI for structs passed by value" (27 functions, 190 allocas and the
`sret`/`byval` pointer ABI in `--output-format=ll`, with unclear payoff).

So the accurate statement about "clspv is slow" is: **`sin`/`cos` going through the libclc software
implementations (plus the division guards) costs that 5x, and `--use-native-builtins` together with
`--cl-fast-relaxed-math` wins it back**; `__constant` demoting to Private scratch is the second layer
(1.25x), and whether the subscript is a constant decides whether that layer is still there
(`smallpt_lit.cl` clears it for another 1.42x). Three layers together: clspv 1586.8 -> 222.0 ms (7.1x).
Note that **the third cut works for glslc too** (`smallpt_lit.comp`: 265.5 -> 215.0 ms), so after the
rewrite GLSL is still about 5% ahead -- that part is a code-generation difference in whether `vec3`
stays a native vector, not one compiler being "slow".

#### Why the op count stays higher than GLSL after optimizing (1289 vs 730)

Because those two numbers **are not the same kind of measurement**. The instruction count printed by
`spirv-dis` is not "amount of work done":

| Item (`t_nat_buf_fast` vs `smallpt_glsl`) | clspv | GLSL | Note |
| --- | --- | --- | --- |
| Instructions in function bodies | 1019 | 440 | GLSL's 730 includes the types/constants/decorations section; this row counts bodies only |
| Floating-point "scalar slots" (vector instructions expanded by component) | 310 (all scalar) | 232 (including 41 vector instructions) | GLSL's `Normalize` x6 and `Cross` x3 each replace about 9 scalar operations; expanded they are about 313, so both sides do roughly the same arithmetic |
| `OpExtInst` | `Fma` x78, `Sqrt` x17, `Sin/Cos` x2 | `Normalize` x6, `Sqrt` x8, `Cross` x3, `Sin/Cos` x2 | clspv contracts into Fma (fewer instructions), GLSL does not |
| `OpPhi` | 184 | 45 | one of the main differences |
| `OpAccessChain` / `OpLoad` | 65 / 55 | 20 / 16 | 3-3.4x difference |
| `OpBitcast` | 73 | 0 | another main difference |
| `OpSelect` | 51 | 10 | `fabs`/`min`/`max` do not go through `GLSL.std.450` |
| `OpBranchConditional` | 41 | 30 | already close |

Breaking the difference down by source (each item has supporting evidence):

1. **`vec3` is fully scalarized (about 300 instructions).** In clspv every component of
   `struct{float x,y,z;}` is an independent SSA value, so reading or writing one `vec3` becomes 3
   `AccessChain` plus 3 `Load`, and one normalization becomes 3 `FMul` plus 3 `FDiv`. On the GLSL
   side it is 1 `Load vec4` plus 1 `VectorTimesScalar`/`Dot`/`Normalize`. Evidence: `AccessChain`
   65 vs 20, `Load` 55 vs 16, and clspv has **not a single vector floating-point instruction**
   (`OpVectorTimesScalar`/`OpDot` are both 0).
2. **Floats in the `__global` buffer are loaded as `uint` and then `OpBitcast` (about +55).**
   Evidence: 55 `OpLoad %uint ... Aligned 4` immediately followed by `OpBitcast %float`.
   `smallpt_vec.cl`, which uses a `__constant` array, does not show this (`OpBitcast` 73 -> 14), so
   it is a fixed cost of the buffer path. What it buys is Private 52->0 and 1.25x speed, still a
   good trade.
3. **Loop-carried values x3 (`OpPhi` 184 vs 45).** The trace loop header is a block with 19 `OpPhi`
   (`t_nat_buf_fast.spvasm` line 1204), because `L`, `o` and `d` each split into 3 scalars, while on
   the GLSL side one `vec3` is one `OpPhi`.
4. **`fabs`/`min`/`max` go through bit operations and selects (`OpSelect` 51 vs 10, part of the
   `OpBitcast` count).** clspv lowers `fabs(x)` to `Bitcast -> BitwiseAnd 0x7FFFFFFF -> Bitcast`
   (7 sites) and `min`/`max` to a compare plus `OpSelect`; GLSL has one `FAbs`/`FMax`.

**But this does not explain the speed difference (302 ms vs 258 ms, only 17%).** Of those extra
about 580 instructions, the vast majority are `OpPhi` (register renaming, which the driver removes),
`OpBitcast` (a free reinterpretation) and `OpSelect`/integer work (1 cycle each). The genuinely
expensive operations -- floating-point multiply-add (310 vs 313 scalar slots), `sin`/`cos`, `sqrt`,
and the amount of scene data read -- are essentially equal on both sides, and clspv's `Fma`
contraction even saves a batch of instructions. That is why "1.77x the instruction count" turns into
only 1.17x at run time.

Item 1 was verified in passing: replacing `vec3` with the native `float4` (`smallpt_vec.cl`) and
compiling with the same options gives function-body instructions 1019->875, `OpPhi` 184->148,
`OpBitcast` 73->14 (because it uses a `__constant` array), but adds 30 `OpVectorShuffle` and loses
the 1.25x of the buffer -- not worth it overall, so the best configuration remains `smallpt_buf.cl`
(struct `vec3` + `__global` buffer + native + fast).

#### Why switching `__constant` to a `__global` buffer is 1.25x faster

Vulkan has no OpenCL `__constant` address space, so clspv can only demote it to a **Private
(per-thread) array**. Lines 538-539 of `_base.spvasm` are the evidence:

```
%18 = OpVariable %_ptr_Private__arr_float_uint_42 Private %16   ; 42 floats (6 walls x 7)
%35 = OpVariable %_ptr_Private__arr_float_uint_48 Private %33   ; 48 floats (3 spheres x 16)
%16 = OpConstantComposite %_arr_float_uint_42 ...               ; initializer
```

Three consequences follow:

1. **Every thread gets a private array of 90 floats (360 B).** Vulkan's Private storage is "the stack
   of one invocation"; a GPU has no hardware stack, so the driver puts it in scratch (spilling to
   local/global memory) and every thread lays it out from the initializer. Compared with "one shared
   read-only buffer for all threads" this is pure extra work, and it directly hurts L1/L2 hit rates
   (the same table is copied thousands of times).
2. **A dynamic subscript cannot be constant-folded.** The hit id is computed at run time (it decides
   which plane or sphere to look up), so every lookup is a genuine indexed access through
   `OpAccessChain %_ptr_Private` plus `OpLoad` (52 such Private accesses in A, 0 in B). With a
   constant subscript clspv could inline the value as an `OpConstant` for free.
3. **A Private array cannot use the read-only buffer optimizations** (uniform propagation, L1
   read-only caching, and so on).

With a `__global` buffer the table exists once, in device memory, shared read-only by all threads,
and the indexed access becomes a normal `OpLoad` (hitting the same data in L1/L2). Measured A
1598.9 -> B 1105 (1.3x with baseline options), and A+native+fast 373.4 -> B+native+fast 302.2
(1.24x) -- both rounds sit around 1.25x, and it is orthogonal to the 5x of `sin`/`cos` (fix either
one first and they stack).

**That 1.25x is a clspv-specific consequence of the demotion, not a general GPU rule.** CUDA has no
such problem:

- CUDA's `__constant__` is a real 64 KB constant region served by the constant cache (8 KB per SM):
  **one copy per module, never copied per thread, no initialization unless written**, and dynamic
  subscripting `c[i]` is supported.
- The constant cache broadcasts accesses that are **uniform within a warp** (one transaction for the
  whole warp, faster than global), and only serializes per address when the addresses **diverge
  inside a warp**. This kernel sits on the diverging side (different pixels hit different primitives
  and take different material branches), so in CUDA `__constant__` -> `__global__` usually also
  gives a small win, but never 1.25x, because there is no "copy the table per thread" cost to remove.
- The CUDA construct that really corresponds to "the clspv pitfall" is **declaring and initializing a
  local array inside the kernel** (`float planes[42] = {...};`), which also lands in local memory
  with per-thread copies and spills -- a well-known anti-pattern in CUDA. Likewise, the most
  efficient home for an 88-float (352 B) table in CUDA is **kernel arguments** (they land in the
  constant bank, addressed with immediate offsets, copied once per launch) or `__shared__` copied
  once per block, not `__constant__` and certainly not a local array.

So the CUDA takeaway is: **"move a constant table from per-thread private storage to shared or
read-only storage" is valid and universal, but the specific move "from `__constant__` to
`__global__`" buys little or can even lose in CUDA** (when the indices are uniform, the constant
cache broadcast beats global). On Vulkan the move is mandatory simply because clspv has no constant
address space available and settles for the worst option, Private.

#### How to make clspv "support" the constant address space (three routes, the third is fastest)

clspv does not implement OpenCL's `__constant` address space (and Vulkan's descriptor types have no
"constant memory" either), so a module-scope `__constant` array is always demoted to Private. To
store that table exactly once there are three routes:

| Route | How | Output | Host change | Measured |
| --- | --- | --- | --- | --- |
| 1. Official option | `--module-constants-in-storage-buffer` | adds `DescriptorSet 1 / Binding 0`, `%_struct_10 = OpTypeStruct %_arr_float_uint_42 %_arr_float_uint_48`; 96156 B / ops 5208 / Private accesses 0 | **Yes**: need a set 1 layout, binding, pool size, `vkUpdateDescriptorSets`, and you must write the initial data yourself | Only the module shape was verified, not run end to end |
| 2. Push constant | `--pod-pushconstant` (with `--max-pushconstant-size`) | the scene table lands in a push constant block | Yes (`VkPushConstantRange` + `vkCmdPushConstants`, and it must fit under the common 128 B minimum) | Not measured; 90 floats = 360 B already exceeds the push constant limit of most implementations |
| 3. Literal subscripts | unroll by hand in the source so every subscript is a compile-time constant | **the whole scene table disappears**, folded into `OpConstant` immediates; 24944 B / ops 1232 / Private accesses **0** / `OpLoad` 6 | **No** (no host change, no extra descriptor) | 480x360/32spp **197.7 ms**, faster than GLSL's 220.8 ms |

The key insight: **whether clspv goes through Private depends on whether the subscript is a
constant, not on what the address space is.** As long as the `i` in `PLANES[i*7+k]` is a
compile-time constant, clspv folds the value into an immediate and not even an `OpVariable` remains.
The original `smallpt.cl` in this repository tried to get the loop unrolled with
`#pragma clang loop unroll(full)` and **that has no effect on clspv** -- the loop survives and `i`
stays a run-time value, hence 6 walls x 5 accesses + 3 spheres x 7 accesses = 52
`OpAccessChain %_ptr_Private`. `private_owner.py` counted them per variable: all 52 land on `%18`
(PLANES, 30) and `%35` (SPHERES, 22), nothing to do with the RNG state.

`src/smallpt_lit.cl` therefore uses macros with literal subscripts:

```c
#define TRY_PLANE(I)  /* PLANES[(I)*7+0] ... written out once per wall */
#define TRY_SPHERE(I) /* SPHERES[(I)*16+0] ... written out once per sphere */
TRY_PLANE(0); TRY_PLANE(1); ... TRY_PLANE(5);
TRY_SPHERE(0); TRY_SPHERE(1); TRY_SPHERE(2);
```

The cost is 6+3 repeated blocks in the source (readability preserved by the macros), and the reward
is that the whole scene table disappears. The hit information (albedo / emission / material / sphere
centre) is selected at the moment of the hit and carried up through `trace()`, so `trace` no longer
does a dynamic lookup by hit id -- which also removes the need for an output struct.

Measured (480x360 / 32 spp, three interleaved rounds in the same run):

| Variant | Module | ops | blocks | Phi | Private accesses | OpLoad | Three rounds | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A baseline (`smallpt.cl`, `__constant`) | 95716 B | 5217 | 488 | 500 | 52 | 58 | 1586.8 / 1574.9 / 1558.9 | 1573.5 ms |
| D `smallpt_buf.cl` + native + fast | 26044 B | 1291 | 88 | 184 | 0 | 55 | 293.3 / 331.1 / 306.6 | 310.3 ms |
| **E `smallpt_lit.cl` + native + fast** | **24944 B** | **1232** | **109** | **270** | **0** | **6** | **211.7 / 190.9 / 190.5** | **197.7 ms** |
| C GLSL (`smallpt.comp`) | 12520 B | 730 | 90 | 45 | 0 | 16 | 210.5 / 231.7 / 220.1 | 220.8 ms |

E is another 36% faster than D (310.3 -> 197.7 ms) and **beats GLSL (220.8 ms) for the first time**.
The reason is that `OpLoad` drops from 55 to 6: variant D has to `OpLoad %uint` + `OpBitcast %float`
for every scene table access (floats in the buffer are loaded as uint), while in E all those values
are immediates and there is no memory access at all. Block and Phi counts are slightly higher than in
D (109 / 270 vs 88 / 184, because the nine hand-written intersection blocks each carry branches),
but that cost is far smaller than the memory traffic removed.

Production image (1024x768 / 1024 spp): **29548.3 ms / 109.02 Mpaths/s** (variant D: 39386.3 ms /
81.79 Mpaths/s), artifact `renders/smallpt_lit.png`. It is **SHA256-identical** to the D image
(`16543658...A3CE9E`) -- same RNG and same arithmetic in the same floating-point order, so the output
is byte-for-byte equal; against the GLSL build the mean difference is 0.016/255.

**Conclusion: no clspv source change and no host cooperation are needed.** To make a "constant table"
actually constant under clspv, the answer is not to find a constant address space but to make the
subscripts constant.

#### GLSL takes the same hit: `const` arrays are materialized into Function storage

The same trick works for GLSL. `glslc` also does not treat `const vec4 PLANE_GEO[6]` as a pure
constant; it materializes an array in Function storage plus dynamic `OpAccessChain`:

```
%1475 = OpVariable %_ptr_Function__arr_v4float_uint_6 Function %107   ; copy of PLANE_GEO
%1499 = OpVariable %_ptr_Function__arr_v3float_uint_6 Function %290   ; copy of PLANE_COL
%1517 = OpVariable %_ptr_Function__arr_v3float_uint_3 Function %318   ; copy of SPH_COL
(in `smallpt_glsl.spvasm` there are 20 `OpAccessChain` total, 10 of them into in-function arrays)
```

So "a constant table smeared into per-thread storage plus dynamic indexing" is not a clspv-only
problem: **glslc has it too, only the storage class is called Function instead of Private**. After
the same macro plus literal-subscript rewrite in `src/smallpt_lit.comp`: Function-storage variables
10 -> **0**, `OpLoad` 16 -> **6**, module 12520 -> 15196 B (slightly larger, because unrolling
creates more branches and Phis: Phi 45 -> 93).

Measured (480x360 / 32 spp, three variants interleaved four rounds each, taking the minimum to avoid
thermal throttling):

| Variant | Module | ops | Phi | Function arrays | OpLoad | Four rounds | Minimum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C GLSL original (`smallpt.comp`) | 12520 B | 730 | 45 | 10 | 16 | 268.2 / 265.5 / 274.4 / 266.9 | 265.5 ms |
| **F GLSL literal (`smallpt_lit.comp`)** | 15196 B | 833 | 93 | **0** | **6** | 217.1 / 247.5 / 215.0 / 221.5 | **215.0 ms** |
| E clspv literal (`smallpt_lit.cl`) | 24944 B | 1232 | 270 | 0 | 6 | 252.5 / 237.4 / 229.6 / 222.0 | 222.0 ms |

F is **1.23x** faster than C (265.5 -> 215.0 ms), the same direction of gain as the clspv cut (1.42x),
but F's absolute number is lower -- **with the same source rewrite, the GLSL build is another 5%
faster than the clspv build**.

The reason is the remaining op difference: F 833 / E 1232. GLSL's `vec3` is a native vector type, so
`dot`/`cross`/`normalize` stay on `%v3float` (F has 41 vector instructions, each replacing several
scalar ones); clspv scalarizes `struct{float x,y,z;}` completely, flattening into `AccessChain` plus
per-component `Load`/`FMul`, so the arithmetic is the same but the instruction count is 50% higher.

Production image (1024x768 / 1024 spp): F **26989.9 ms / 119.35 Mpaths/s**, artifact
`renders/smallpt_lit_glsl.png`; against E the mean difference is 0.016/255 (noise-level, caused by
the two compilers associating floating-point operations differently).

#### Are the images equivalent?

Both run the same host, the same seed and the same sampling strategy, but the compilers reassociate
floating-point expressions and the RNG consumption order shifts, so **pixel values are not
bit-identical**. Comparison at 480x360 / 32 spp:

- max difference 121/255, RMS 9.34/255, 28.8% of bytes differ by more than 1 (most of the difference
  is noise from a single path hitting a different primitive);
- but the per-channel means are nearly identical: clspv 125.27 / 102.18 / 125.35, GLSL 125.22 /
  102.15 / 125.31;
- the top-left pixel's linear RGB is exactly the same (0.1129 0.1079 0.1129).

1024x768 / 256 spp: max difference 60/255, mean difference 2.15/255, 49.7% of bytes differ by more
than 1, and the top-left pixel is A/B `0.1101 0.0729 0.0838` vs C `0.1025 0.0662 0.0766` -- again
two different implementations of the same unbiased estimator, not an algorithmic difference (the
higher the resolution and spp, the closer they get).

After `--use-native-builtins=sin,cos --cl-fast-relaxed-math` the accuracy does not degrade in any way
that affects the image:

| Comparison (1024x768 / 1024 spp) | Per-channel mean | Mean byte diff | Max diff | Share >1 |
| --- | --- | --- | --- | --- |
| Baseline clspv: 128.848 / 105.785 / 129.080 | - | - | - | - |
| Best variant: 128.785 / 105.737 / 129.017 | 0.063 / 0.048 / 0.063 vs baseline | 1.12/255 | 35 | 26.4% |
| GLSL: 128.769 / 105.724 / 129.000 | 0.016 / 0.013 / 0.016 vs best variant | 0.82/255 | 19 | 16.2% |

So the best variant converges to an image that differs from GLSL by **only 0.016/255 (about
0.01%)**, less than it differs from the clspv baseline itself; the difference is entirely per-pixel
noise. The 512 spp convergence test shows the same: baseline 128.826 / 105.768 / 129.060 vs best
128.764 / 105.721 / 128.997 (difference 0.062 / 0.048 / 0.064). The price is that
`native_sin`/`native_cos` no longer guarantee the ULP of the non-native OpenCL versions and that
denormals are flushed -- entirely acceptable for a Monte Carlo renderer.

#### Semantic cross-check list (`src/smallpt.cl` vs `src/smallpt.comp`)

Compared line by line; every item below matches (the corresponding form in the two files is given in
parentheses):

| Item | clspv version `smallpt.cl` | GLSL version `smallpt.comp` |
| --- | --- | --- |
| Constants | `PI / EPS=1e-4 / BIG=1e20 / MAX_DEPTH=10 / CAM_PUSH=140` | same values (`const`, same names) |
| Six walls | `PLANES[6*7]`: n(x,y,z), d, albedo | `PLANE_GEO[6]` (vec4, w holds d) + `PLANE_COL[6]` |
| Wall plane equation | `n.p = d`, normals into the box; left x=1, right x=99, back z=0, front z=170 (black), floor y=0, ceiling y=81.6 | same |
| Three spheres | `SPHERES[3*16]`: center, r, albedo, emit, material | `SPH_GEO/SPH_COL/SPH_EMIT/SPH_MAT[3]` |
| Sphere parameters | mirror (27,16.5,47) r16.5 SPEC; glass (73,16.5,78) r16.5 GLASS; light (50,681.6-0.27,81.6) r600 emit12 DIFF | same |
| Plane intersection | `denom` jitter threshold +-1e-8, `t > EPS && t < best` | same |
| Sphere intersection | `det = b*b - c`, a hit at `t0` does `continue`, then try `t1` | same (including the `continue` position) |
| Hit id | `<NPLANE` is a wall, `NPLANE+i` is a sphere | same |
| Normal | walls use the plane normal; spheres `normalize(x - center)` | same |
| Iterative trace | `L += thr*e` -> break when `thr` is exhausted -> Russian roulette past depth 5 -> material dispatch | same (same order: roulette **before** material dispatch) |
| Roulette | `pmax = max(f.xyz)`, continue only when `rnd < pmax`, then `f /= pmax` | same (`max(f.x, max(f.y, f.z))`) |
| Diffuse | `r1=2*pi*rnd`, `r2=rnd`, `r2s=sqrt(r2)`, `u=normalize(cross(w.x>0.1\|\|w.x<-0.1 ? (0,1,0):(1,0,0), w))`, `v=cross(w,u)`, `nd` from the three components | same (`abs(w.x) > 0.1` is equivalent to `w.x>0.1\|\|w.x<-0.1`) |
| Mirror | `d - n*2*dot(n,d)` with the geometric normal `n` (not flipped) | same |
| Glass | `into = dot(n,nl)>0`; `nnt=nc/nt` or `nt/nc`; total internal reflection branch `thr *= f`; `R0=(nt-nc)^2/(nt+nc)^2`; `cc`, `c5`, `Re`, `Tr`, `P=0.25+0.5Re`, `RP=Re/P`, `TP=Tr/(1-P)`; `rnd < P` picks reflection | same (identical formulas) |
| RNG | LCG `s*1664525u+1013904223u`, takes `s>>8`, scaled by `1/16777216` | same |
| Seed | `(py*w+px)*1973u + frame*9277u + 1u` | same |
| Camera | `camO=(50,52,295.6)`, `camD=normalize(0,-0.042612,-1)`, `cx=(aspect*0.5135,0,0)`, `cy=normalize(cross(cx,camD))*0.5135` | same |
| Sampling | 2x2 subsamples; `r=2*rnd`, `d = r<1 ? sqrt(r)-1 : 1-sqrt(2-r)`; `dRaw = cx*(...)+cy*(...)+camD`; `start=camO+d*140` | same |
| Accumulation | per subsample `/spp` -> `clamp` (cap 20) -> sum the 4 subsamples -> `*0.25` -> write at `idx=(py*w+px)*4`, 4th channel = 1 | same, but the cap is 1 |
| Row order | `pyUp = h-1-py` | same |
| Workgroup | `reqd_work_group_size(8,8,1)` | `local_size_x=8, local_size_y=8` |
| Descriptors | `img` = binding 0, `params` = binding 1 | `Img` = 0, `Params` = 1 (readonly) |

**The only two differences are not semantic**:

1. the representation of `vec3`: `struct{float x,y,z;}` in the clspv version, the native `vec3` in
   GLSL;
2. the compiler's floating-point reassociation and RNG consumption order (glslc reassociates), which
   makes the per-pixel sample sequence differ.

The third implementation (`src/smallpt_buf.cl`) and the host's `kScene[75]` were cross-checked as
well: offsets `SC_BASE=4`, plane geometry 0, plane albedo 24, sphere geometry 42, sphere albedo 54,
sphere emit 63, sphere material 72, 75 floats in total, with values identical to the
`PLANE_GEOM/SPH_GEO` tables (including `681.6 - 0.27`), and the entry point is `smallpt` everywhere.

Convergence cross-check (both run 1024x768 / 1024 spp through the same host):

| Metric | clspv (A) | GLSL (C) |
| --- | --- | --- |
| Per-channel mean (0-255) | 128.848 / 105.785 / 129.080 | 128.769 / 105.724 / 129.000 |
| Mean difference | - | 0.078 / 0.061 / 0.080 (about 0.03%) |
| Share of dark pixels (<10) | 0.15% | 0.15% |
| Share of bright pixels (>200) | 1.80% | 1.79% |
| Mean byte difference | - | 1.15/255 (max 34) |

At 1024 spp both have converged onto the same image, which shows **this is not an algorithmic
difference**, just two different sampling implementations of the same estimator.

#### Conclusion

- For **performance**: on the Intel driver of this machine the GLSL (glslc) route is 5-6.6x faster
  than clspv and produces SPIR-V more than 7x smaller (so the driver compiles it faster too). This
  scene -- lots of scalar control flow with short vectors -- is exactly the shape that suffers most
  from the frontend over-inlining of clspv.
- For **reusing OpenCL C code**: clspv remains the right route, but expect a 3-5x performance cost
  for kernels like this, and move `__constant` arrays to a parameter buffer where possible (worth
  about 1.3x).
- **Device memory**: identical across all variants and determined only by resolution (1024x768 ->
  12.00 MiB image buffer).
- After the literal-subscript rewrite on both sides, the numbers are 215.0 ms (GLSL) vs 222.0 ms
  (clspv): the remaining 5% is `vec3` scalarization, not a compiler defect.

## Denoiser: OpenImageDenoise's U-Net weights, executed as Vulkan compute

The renderer can denoise its own output without linking OpenImageDenoise. OIDN ships its trained
weights as a plain data blob (`weights/rt_hdr.tza`), and the RT filter behind them is an ordinary
U-Net; both are reproduced here:

- `tools/oidn/export_oidn_weights.py` parses the `.tza` container (u16 magic `0x41D7`, version
  2.0, tensor directory, float32 payloads - see `core/tza.cpp`) and concatenates the 16 layers in
  network order. Each 3x3 kernel row is padded from 9 to 12 floats so every row stays 16-byte
  aligned.
- `src/oidn_conv_tiled.comp` (convolution) and `src/oidn_{pre,pool,upsample,concat_copy,post}.comp`
  implement the passes; `src/oidn_denoise.cpp` records the whole chain into one command buffer on
  the renderer's own queue.

### Network reproduced

The topology comes from `core/unet.cpp` (`buildNet`) and the operator descriptions from
`core/conv.h`, `core/pool.h` and `core/upsample.cpp`:

| Layer | Operator | In -> Out | Note |
| --- | --- | --- | --- |
| enc_conv0/1 | 3x3, pad 1, ReLU | C -> 32 -> 32 | full resolution, C = input channels (3, 6 or 9) |
| pool1 | 2x2 max, stride 2 | 32 | H/2 |
| enc_conv2 | 3x3, pad 1, ReLU | 32 -> 48 | H/2 |
| pool2 | 2x2 max | 48 | H/4 |
| enc_conv3 | 3x3, pad 1, ReLU | 48 -> 64 | H/4 |
| pool3 | 2x2 max | 64 | H/8 |
| enc_conv4 | 3x3, pad 1, ReLU | 64 -> 80 | H/8 |
| pool4 | 2x2 max | 80 | H/16 |
| enc_conv5a/5b | 3x3, pad 1, ReLU | 80 -> 96 -> 96 | H/16 |
| upsample4 + concat(pool3) | 2x nearest, channel concat | 96 + 64 = 160 | H/8 |
| dec_conv4a/4b | 3x3, pad 1, ReLU | 160 -> 112 -> 112 | H/8 |
| upsample3 + concat(pool2) | 2x nearest, channel concat | 112 + 48 = 160 | H/4 |
| dec_conv3a/3b | 3x3, pad 1, ReLU | 160 -> 96 -> 96 | H/4 |
| upsample2 + concat(pool1) | 2x nearest, channel concat | 96 + 32 = 128 | H/2 |
| dec_conv2a/2b | 3x3, pad 1, ReLU | 128 -> 64 -> 64 | H/2 |
| upsample1 + concat(input) | 2x nearest, channel concat | 64 + C | full resolution |
| dec_conv1a/1b | 3x3, pad 1, ReLU | (64+C) -> 64 -> 32 | |
| dec_conv0 | 3x3, pad 1, **no ReLU** | 32 -> 3 | |

`dec_conv1a` really is `64 + C` input channels (67 with colour only): OIDN pads tensors to
`Chw16c` internally, but the padding channels are zero on both sides (weights padded with zeros,
input reorder zero-fills), so keeping the compact channel counts gives bit-identical results with
less work.

### Pre- and post-processing

Taken from `core/input_reorder.ispc`, `core/output_reorder.ispc` and `core/color.ispc`:

1. `inputScale = 0.18 / exp2(mean(log2 L))` over 16x-downsampled block averages (autoexposure,
   `core/color.cpp`), so exposure is normalised before the network sees the image.
2. colour: `y = clamp(rgb * inputScale, 0, 65504)`, then the PU curve
   (`puForward`, three branches, `normScale = 1/puForward(65504)`).
3. output: `clamp`, inverse PU, then `* (1/inputScale)`, which is how OIDN restores the original
   exposure (`TransferFunction::setInputScale` sets `outputScale = 1/inputScale`).

### Layout and dispatch

- Tensors are pixel-major (HWC) float32 in SSBOs; weights are one more SSBO.
- `oidn_conv_tiled.comp` (the default) is a shared-memory tiled, register-blocked convolution:
  a work group covers 16x16 pixels and 8 output channels, every thread computes a 2x2 pixel
  block, and four input channels are staged in shared memory at a time (details below).
- `oidn_conv.comp` is the original one-thread-per-pixel version. It is still built and still
  selectable with `OIDN_CONV=simple`, so the two kernels can be compared on the same input; its
  output is bit-identical to the tiled one (see Validation), which makes the A/B measurement a
  pure performance question.
- Upsampling writes directly into the concatenation buffer at a channel offset, so each decoder
  stage costs two dispatches instead of three.
- One `vkCmdPipelineBarrier` per pass (compute -> compute), 30 dispatches per tile, one submit.

### Convolution kernel: shared-memory tiling and register blocking

The simple kernel is load-bound: one thread handles one pixel and 8 output channels, so for every
input channel it issues 9 voxel loads plus 9*8 weight loads for 8*9 = 72 multiply-adds, i.e. roughly
1.1 load instructions per FMA. At 640x480 that is 37 G MAC running at ~35 GFLOP/s on the integrated
GPU, which is far below its arithmetic peak, so the goal was to raise the work per load. After the
rewrite the same frame takes 0.57 s, i.e. ~130 GFLOP/s, at unchanged arithmetic:

```
work group   8x8 = 64 threads                 (local_size 8, 8, 1)
block        16x16 output pixels + 8 channels (dispatch x/y = ceil(w/16), ceil(h/16))
per thread   2x2 pixels x 8 channels = 32 accumulators
shared       tileBuf[1296]: CHUNK=4 channels of an 18x18 patch (5.1 KB)
```

- Load phase: the 18x18 patch of each 4-channel chunk is staged cooperatively, one blocked chunk per
  patch pixel per thread, so consecutive threads read consecutive pixels of the same chunk out of
  the channel-blocked tensor (see the layout section below). Out-of-image positions are written as
  0 instead of being skipped, because the compute phase reads the patch without bounds checks.
- Because the chunk width is now tied to the tensor block, `CHUNK` is fixed at 4 and the sweep
  below no longer applies: 8 or 16 channels would mean two or four vector loads per thread and a
  larger staging buffer for no arithmetic benefit.
- Compute phase: a thread's 2x2 pixel block needs a 4x4 voxel window (16 shared loads) instead of
  four separate 3x3 neighbourhoods (36 loads), and that window is reused by all 8 output channels
  and all 9 taps. Per chunk channel the cost is 16 shared loads + 72 weight
  loads (identical across the work group, so they broadcast) for 288 FMAs, about 0.31 loads per
  FMA - 3.5x better than the simple kernel.
- The tap accumulation order per output pixel is unchanged (input channel ascending, taps row
  major), so the result is bit-identical rather than merely close.
- Pitfalls found while writing it: `patch` is a reserved GLSL tessellation keyword (the shared
  array had to be renamed); glslc rejects a computed array length such as
  `shared float a[CHUNK * PATCH * PATCH]` (a literal is required); and the block origin must come
  from `gl_WorkGroupID`, not `gl_GlobalInvocationID`, which already includes the local size.

Tuning was measured on one layer (dec_conv4a, 160 -> 112 channels, 640x480) with the single-pass
probe described below, kernel time only, three of the four variants checked against the CPU
reference on the same run:

| Variant | Time | Note |
| --- | --- | --- |
| simple kernel (1 pixel/thread, no shared memory) | 2644 ms | baseline |
| tiled, 2x2 register blocking, no shared memory | 1082 ms | register blocking alone |
| tiled, CHUNK=4 shared memory | 592 ms | adopted |
| tiled, CHUNK=2 shared memory | 591 ms | same speed, twice the barriers |
| tiled, CHUNK=8 shared memory | 890 ms | 10.1 KB of shared memory, fewer resident work groups |
| tiled, CHUNK=16 shared memory | 1746 ms | 20.7 KB, one work group resident per SM |

So the 2x2 blocking is worth 2.4x on its own and the shared-memory staging another 1.8x, but only
while the staging buffer stays small enough that several work groups fit in the 32 KB of shared
memory this device reports. CHUNK=8 was the original choice and it was 1.5x slower than CHUNK=4.

### Tiling (identical to OIDN's)

Large images do not fit in one tensor set, so they are split exactly the way `UNetFilter` in
`core/unet.cpp` does it, with the constants from `core/unet.h`:

```
alignment      = 16
receptiveField = 174
overlap        = round_up(receptiveField / 2, alignment) = 96
```

- **Tile size** (`computeTileSize`): start with `tileW/H = round_up(W/H, 16)`, then while the
  tensor set exceeds the memory budget, bump the tile count along the longer axis and recompute
  `tileH = max(round_up(ceil_div(H - 2*overlap, countH), 16) + 2*overlap, 3*overlap)` (same for the
  width). The final counts come from `ceil_div(H - 2*overlap, tileH - 2*overlap)`. The budget is
  OIDN's default 3000 MB, overridable with `OIDN_DENOISE_MAX_MB`.
- **Tile mapping** (`execute`): tile `i` starts at `h = i*(tileH - 2*overlap)`; `overlapBeginH` is
  96 for every tile except the first, `overlapEndH` 96 for every tile except the last;
  `tileH1 = min(H - h, tileH)`; `tileH2 = tileH1 - overlapBeginH - overlapEndH` is the part this
  tile writes back; `alignOffsetH = tileH - round_up(tileH1, alignment)` places the valid content
  at the bottom of the tile buffer (the 16-pixel slack plus the area outside the image is zero).
- `oidn_pre.comp` implements the input half of that mapping (source strides stay the full image
  width) and `oidn_post.comp` the output half: only tile rows
  `[alignOffset + overlapBegin, alignOffset + overlapBegin + tileH2)` are written to the image.
- Because `overlap = 96` is larger than half of the 174-pixel receptive field, every written pixel
  still sees its full neighbourhood, so tiling is not an approximation: a 640x480 image denoised
  as 15 tiles of 288x288 matches the single-tile result **bit for bit** (below).

### Validation

`tools/oidn/ref_forward.py` is an independent numpy implementation of the same network (same
weights, same pre/post-processing). Feeding the same image to both:

| Image | numpy vs Vulkan (mean abs) | max abs | pixels with rel > 1% |
| --- | --- | --- | --- |
| 128x96 synthetic | 1.4e-6 | 1.1e-4 | 0.000% |
| 640x480 synthetic | 1.3e-6 | 1.2e-4 | 0.000% |

Tiling is checked separately, against the single-tile result of the same build:

| Comparison | mean abs | max abs |
| --- | --- | --- |
| 640x480, 15 tiles (288x288, 100 MB) vs 1 tile | 0.000e+00 | 0.000e+00 |
| 641x481, 20 tiles vs 1 tile (odd image size) | 0.000e+00 | 0.000e+00 |
| 3840x2160, 6 tiles vs 24 tiles | 0.000e+00 | 0.000e+00 |
| 641x481 GPU vs numpy on the same padded content | 3.8e-6 | 4.4e-4 |

The tiled convolution is checked against the simple kernel, which was already validated against
numpy. The two kernels decompose the work differently but accumulate the taps in the same order,
so the agreement is exact:

| Comparison (`OIDN_CONV=simple` vs default) | mean abs | max abs |
| --- | --- | --- |
| 640x480, full 30-pass chain | 0.000e+00 | 0.000e+00 |
| 641x481 (odd size), full chain | 0.000e+00 | 0.000e+00 |
| 3840x2160, 6 tiles, full chain | 0.000e+00 | 0.000e+00 |
| dec_conv4a alone, 65x49 (per-layer CPU reference) | 2.3e-8 | 7.2e-7 |

`tools/oidn/conv_probe.cpp` is the per-layer tool used for those rows: it runs one convolution
dispatch of either shader on synthetic input and prints the difference against a CPU reference
computed in the same weight layout, which is how the kernel was iterated without paying for the
whole 30-pass chain:

```
cmake --build build --config Release --target conv_probe
conv_probe.exe <conv.spv> <rt_hdr.bin> <w> <h> <cin> <cout> <weightOfs> <block> [group] [repeat] [--clc] [blockY]
```

`group`, `repeat` and `blockY` were added for the register-blocking sweep: a blocked kernel needs a
rectangular block (24x16 for 3x2 pixels per thread), and `--clc` binds the parameter buffer plus the
12-byte push constant range that the clspv builds expect. `repeat` dispatches the pass N times inside
one command buffer and divides by N, which is what makes a single-layer measurement stable on a
throttling part.

The last row checks the odd-size path against the reference: the 641x481 input is padded to the
656x496 tensor the shader actually builds, run through `ref_forward.py` with the same exposure
scale (7th argument), and cropped back. 0.000% of the pixels differ by more than 1%.

The residual against numpy is float32 rounding, so the Vulkan path reproduces OIDN's arithmetic
rather than approximating it.

### Auxiliary inputs: albedo and normal (`rt_hdr_alb` / `rt_hdr_alb_nrm`)

OIDN's RT filter is trained for three feature sets and picks the matching weights at run time
(`core/unet.cpp:293-310`): colour only (`rt_hdr`), colour + albedo (`rt_hdr_alb`),
colour + albedo + normal (`rt_hdr_alb_nrm`). Colour + normal without albedo is rejected by OIDN
(`throw`), and this implementation does the same. The same rule is applied here, in
`aux_weights_path()` inside `src/oidn_denoise.cpp`.

**Preprocessing** differs per channel group (`core/input_reorder.ispc`, the three `store*`
functions):

| Channels | Transform |
| --- | --- |
| colour | `clamp(rgb * inputScale, 0, 65504)` -> PU -> `* normScale` |
| albedo | `clamp(a, 0, 1)`; no exposure scale, no transfer function |
| normal | `clamp(n, -1, 1)` -> `* 0.5 + 0.5` |

`src/oidn_pre.comp` writes the three groups into one `C`-channel tensor and zero-fills the
padding region, so the rest of the chain does not care how many features are present.

**Two layers change shape with the feature set**, which is why the weight offsets cannot be a
single constant shift:

- `enc_conv0` has `C` input channels, i.e. `32 * (C-3) * 12` more floats.
- `concat1 = getConcatDesc({upsample1Desc, inputReorderDesc})` has `64 + C` channels
  (`core/unet.cpp:405-407`), so `dec_conv1a` has `64 + C` input channels, i.e.
  `64 * (C-3) * 12` more floats.

Adding both gives the measured blob sizes: `rt_hdr.bin` 1,219,140 floats,
`rt_hdr_alb.bin` 1,222,596 (+3456 = 1152 + 2304), `rt_hdr_alb_nrm.bin` 1,226,052 (+6912).
`export_oidn_weights.py` reads `cin`/`cout` from each tensor's own dims, so exporting the 6 and 9
channel blobs needed no code change. The host walks the 16 layers at run time
(`layerWeightOfs` in `oidn_denoise_image`) instead of hardcoding the offsets, and every other
channel count is taken from the source tensor of each pass, so no pass table entry depends on the
feature set any more.

**Renderer side.** `src/smallpt.cl` writes a first-hit G-buffer next to the beauty buffer: one
extra primary ray through the pixel centre gives the albedo (`hit_color`) and the shading normal
(`hit_normal`) of the first surface, written to two more SSBOs (bindings 1 and 2) as 3 floats
per pixel. `src/smallpt_main.cpp` hands both to `oidn_denoise_image` through `OidnAuxImages`.
First-hit is what OIDN's examples use and what the `cleanAux` variants would require.

**Validation.** `tools/oidn/ref_forward.py` now takes the feature set too:

```powershell
python tools\oidn\ref_forward.py tools\oidn\rt_hdr.bin         in.f32 640 480 out3 -
python tools\oidn\ref_forward.py tools\oidn\rt_hdr_alb.bin     in.f32 640 480 out6 - alb.f32
python tools\oidn\ref_forward.py tools\oidn\rt_hdr_alb_nrm.bin in.f32 640 480 out9 - alb.f32 nrm.f32
```

With synthetic albedo (including out-of-range values, so the clamp is exercised) and synthetic
unit-length normals, 640x480:

| Features | numpy vs Vulkan (mean abs) | max abs | rel > 1% |
| --- | --- | --- | --- |
| colour only (3ch) | 1.3e-6 | 1.2e-4 | 0.000% |
| + albedo (6ch) | 1.4e-6 | 1.2e-4 | 0.000% |
| + albedo + normal (9ch) | 1.6e-6 | 1.5e-4 | 0.000% |

Every intermediate tensor was also compared stage by stage (25 tensors for 3ch and for 9ch, 21
for 6ch), max error 1.3e-5, i.e. float32 accumulation noise. The 6ch blob consumes exactly
1,222,596 floats and the 9ch blob 1,226,052, so the offsets line up to the last bias padding float.

**Effect on the render.** 640x480, 1 spp, denoise time ~0.6 s:

| Features | Sphere silhouettes | Mirror reflection |
| --- | --- | --- |
| colour only | smeared away, spheres nearly vanish | flattened into the wall colour |
| albedo + normal | preserved, plus the contact shadow | reflection region keeps its contrast |

At 4 spp the two paths differ by mean 4.3e-3 (max 0.28) and the largest differences sit exactly on
the sphere rims, which is where the normal map carries information the colour alone does not.

**Cost.** Only two layers grow: `enc_conv0` gets 3 more input channels and `dec_conv1a` gets 3
more (both at full resolution), i.e. 5184 extra MAC per output pixel out of ~120222, so the extra
arithmetic is ~4%. The measured overhead at the default 3000 MB tile budget, same build, same input
(median of 3 runs, denoise time only, Intel UHD Graphics / Vulkan 1.4.323):

| Image (MP) | Tiling | 3ch `rt_hdr` | 9ch `rt_hdr_alb_nrm` | Overhead | Tensors 3ch / 9ch |
| --- | --- | --- | --- | --- | --- |
| 128x96 (0.012) | 1 tile | 39 ms | 70 ms | JIT dominated (~15 ms of shader compile per run) | 16.7 / 17.2 MB |
| 640x480 (0.307) | 1 tile | 615 ms | 656 ms | +6.7% | 416 / 430 MB |
| 641x481 (0.308) | 1 tile of 656x496 | 656 ms | 676 ms | +3.0% | 441 / 456 MB |
| 1024x768 (0.786) | 1 tile | 1516 ms | 1577 ms | +4.0% | 1066 / 1102 MB |
| 1600x900 (1.44) | 1 tile of 1600x912 | 2825 ms | 3013 ms | +6.7% | 1978 / 2045 MB |
| 1920x1080 (2.07) | 1 tile of 1920x1088 | 4103 ms | 4329 ms | +5.5% | 2831 / 2927 MB |
| 3840x2160 (8.29) | 6 tiles of 1408x1184 | 20112 ms | 21218 ms | +5.5% | 2260 / 2336 MB |

The 9 channel path runs at a nearly constant cost per output megapixel, ~2.1 s/MP, from 640x480 to
4K (0.47 MP/s of processed pixels), so the aux features do not change how the cost scales. Tiling
remains the only thing that makes the numbers worse: 640x480 in 4 tiles (200 MB budget) takes 1067 ms
instead of 656 ms, in 12 tiles (120 MB) 1798 ms, and 4K in 24 tiles (800 MB) 27000 ms instead of
21218 ms, because the 96 pixel overlap is recomputed on both sides of every seam.

Tensor memory grows by a flat +3.4% (the input tensor goes from 3 to 9 channels at full resolution
and `concat1` from 67 to 73), i.e. +14 MB at 640x480 and +96 MB at 1080p. At 1080p the 9 channel
tensor set is 2927 MB, only 73 MB under OIDN's default 3000 MB budget, so anything larger tiles.



`tiled` is the default shared-memory kernel, `simple` the `OIDN_CONV=simple` kernel, same build and
same input, denoise time only (the path tracer is not included):

| Image | Tiling | tiled | simple | Speedup | Tensor memory |
| --- | --- | --- | --- | --- | --- |
| 128x96 | 1 tile | 52 ms | 234 ms | 4.5x | 16.7 MB |
| 640x480 | 1 tile | 0.57 s | 2.14 s | 3.7x | 416 MB |
| 640x480 | 8 tiles of 304x336 | 1.33 s | 8.1 s (15 tiles of 288x288) | 6.1x | 138 MB |
| 641x481 (odd size, padded to 656x496) | 1 tile | 0.68 s | 2.20 s | 3.2x | 441 MB |
| 1024x768 | 1 tile | 1.53 s | 5.40 s | 3.5x | 1066 MB |
| 1024x768 | 8 tiles of 400x480 | 2.66 s | 11.7 s | 4.4x | 260 MB |
| 1920x1080 | 1 tile of 1920x1088 | 4.24 s | 14.74 s | 3.5x | 2831 MB |
| 3840x2160 | 6 tiles of 1408x1184 | 19.0 s | 70.7 s | 3.7x | 2260 MB |

The multi-tile rows scale better than the single-tile ones because a smaller tile means more
independently scheduled work groups per dispatch, which the new kernel needs in order to hide
latency. A 4K frame used to need ~11.8 GB of tensors and would also have hit the device's 1 GB
`maxStorageBufferRange` (the 67-channel `concat1` alone is 2.2 GB at that size); with tiling it
runs in a 2.3 GB budget. Tiling is still not free: 8 tiles of 304x336 cover ~2.2x the pixel area of
the single 640x480 tile (the overlap is recomputed on both sides of every seam), so the same frame
takes 1.33 s instead of 0.57 s. The tile size should be as large as the budget allows.

For comparison the path tracer itself takes 1.3 s at 640x480 / 16 spp and 1.5 s at 1024x768 / 8 spp,
so at 640x480 the denoiser is now cheaper than the render that feeds it. The remaining gap to
arithmetic peak is still instruction issue and load latency rather than bandwidth, so the next
levers are fewer output channels per accumulator set (fp16 storage, measured in the next section),
int8 dot products, or a wider per-thread pixel block (2x4 instead of 2x2, at the cost of 64
accumulators and likely register spills). Cooperative matrices are not available on this device
(`cap_probe` reports no `VK_KHR_cooperative_matrix`).

### FP16 tensor storage (the default; `OIDN_STORE=fp32` opts out)

OIDN's CUDA backend stores every tensor and weight as FP16 and accumulates in FP32
(`devices/cuda/cuda_device.cpp` sets `tensorDataType = weightDataType = DataType::Float16`, and
`cuda_conv.cu` picks `Float32` accumulation unless `fastMath` is set). This tree does the same for
the tensors only: FP16 halves the element size of every intermediate tensor while the weights, the
shared-memory patch, the transfer functions and every accumulator stay FP32, so the rounding is
applied once per tensor store and nowhere else. The shaders are built twice
(`<pass>.spv` and `<pass>_h.spv`, the latter with `-DOIDN_HALF_TENSOR=1`) and the host picks the set
at run time, which means the two variants can be compared on the same input and machine.

**This is now the default**, because it is the fastest configuration: `oidn_denoise_image` starts
from `ctx.storage16Bit`, which `src/smallpt_main.cpp` and `src/oidn_denoise_tool.cpp` set from
`VkPhysicalDevice16BitStorageFeatures::storageBuffer16BitAccess` after enabling the feature at
device creation. A device without 16-bit storage buffers therefore silently gets FP32;
`OIDN_STORE=fp32` forces FP32 on a capable device (the configuration the FP32 reference outputs
come from) and `OIDN_STORE=fp16` forces FP16, which then fails with an explicit message if the
feature is missing. The tool prints which set it loaded:

```
[oidn] weights 4.68 MB, tensors 219.32 MB FP16 (640x480 tile, 30 dispatches per tile)
```

Effect on the image, measured on a fixed 640x480 color + albedo + normal input: FP16 against FP32
gives mean absolute error 3.7e-4, max 8.0e-2 on values up to 15.7 (a 0.1% relative peak), mean
relative error 7.6e-4, and 0.011% of the pixels above 1% relative error; after the 8-bit PNG encode
the two differ by at most 1 LSB on 4.34% of the samples (61.75 dB PSNR). Historical per-resolution
figures (74.3 dB at 640x480, 71.8 dB at 1080p, 74.1 dB at 4K on float output) are unchanged.

Measured denoise time, median of 3 interleaved runs on a fixed input (today's build, 640x480 with
color + albedo + normal):

| Configuration | Denoise time | Tensor memory |
| --- | --- | --- |
| `OIDN_STORE=fp32` (the old default) | 467.4 ms | 438.65 MB |
| **default (FP16 tensors)** | **412.6 ms** | **219.32 MB** |

and the tile count reacts to the smaller elements, which is the bigger effect at high resolution:
at 1920x1080 the 3000 MB budget now fits the whole frame in **one** tile (1491 MB of FP16 tensors,
2673.7 ms), while forcing `OIDN_DENOISE_MAX_MB=800` splits it into 4 tiles and costs 3279.5 ms
(+23%) in overlap rework.

The per-resolution table from the round that introduced FP16 (median of 3 runs, each pair on the
same input, measured before the channel-blocked layout landed, so the absolute times are about twice
today's - the *ratios* and the tile counts are what it was recorded for):

| Image | Feature set | FP32 tensors | FP16 tensors | Change | Tensor memory |
| --- | --- | --- | --- | --- | --- |
| 640x480 | 3ch | 608 ms | 535 ms | -12% | 416 -> 208 MB |
| 640x480 | 9ch | 630 ms | 578 ms | -8% | 430 -> 215 MB |
| 1024x768 | 3ch | 1515 ms | 1413 ms | -7% | 1066 -> 533 MB |
| 1024x768 | 9ch | 1570 ms | 1469 ms | -6% | 1102 -> 551 MB |
| 1920x1080 | 3ch | 3843 ms | 3433 ms | -11% | 2831 -> 1416 MB |
| 1920x1080 | 9ch | 4101 ms | 3653 ms | -11% | 2927 -> 1464 MB |
| 3840x2160 | 3ch | 18002 ms | 14556 ms | -19% | 6 tiles -> 2 tiles |
| 3840x2160 | 9ch | 19312 ms | 16601 ms | -14% | 6 tiles -> 4 tiles |

Two things happen at once. The direct saving is modest, 6-12%, because the pipeline is limited by
instruction issue and sector traffic rather than by bytes (see the profile below). The larger win
is at 4K, where folding 4-byte elements into 2-byte ones lets the tiling planner pick a bigger tile
for the same 3000 MB budget (2016x2160 instead of 1408x1184), which removes most of the 96-pixel
overlap rework. That is also why the tile planner now takes the element size and the auxiliary
channel count into account instead of assuming float32 and the color-only channel sums.

### Channel-blocked tensor layout

The profile below shows the real ceiling of the interleaved (HWC) layout: a thread owns one channel
of one pixel, a warp writes 32 four-byte words that are `channels*4` bytes apart, and every word
pays for a whole 64-byte sector. The pure data-movement passes ran at about 2 GB/s that way, and
that ceiling applied to the convolution staging loops too.

Every tensor is therefore stored channel-blocked: the channel stride is `c` rounded up to a
multiple of 4, so a pixel is a run of whole 4-channel chunks and every chunk is 16-byte (FP32) /
8-byte (FP16) aligned. Each shader then processes one pixel and one 4-channel chunk per thread
instead of one channel, and consecutive threads (consecutive pixels) touch consecutive chunks.

- `oidn_conv_tiled.comp`: the staging loop loads one chunk per patch pixel instead of one channel
  per thread (324 vector loads for a 16x16 block instead of 1296 scalar loads, ~5 iterations per
  thread), and the store side writes two chunks per pixel instead of eight scalars. `og` is a
  multiple of 8 and the destination stride a multiple of 4, so both stores stay 16-byte aligned;
  `dec_conv0` (3 output channels, stride 4) writes one chunk, and the padding channel is
  initialized to zero (or the bias) so it is stored deterministically.
- `oidn_pool.comp`, `oidn_upsample.comp`, `oidn_concat_copy.comp`: one 4-channel chunk per thread,
  one or four vector loads and one vector store, no per-channel loop.
- `oidn_post.comp`: the 3 output channels come from one aligned chunk, one load.
- `oidn_pre.comp` keeps scalar stores (it computes each channel with a different transfer
  function) but honours the padded stride and zero-fills the padding channels, so no consumer ever
  reads uninitialized memory. `oidn_conv.comp` (the `OIDN_CONV=simple` reference kernel) keeps its
  scalar accesses and only follows the padded stride.
- Host: tensor bytes are `w*h*pad4(c)*elemBytes`, `OIDN_DUMP_TENSOR` compacts the padding back to
  `w*h*c` floats (the contract the numpy references expect), the per-pixel passes dispatch
  `pad4(c)/4` chunks in z, and the tile-budget channel sums count padded strides (236 full-res
  channels, +8 per auxiliary image).

The layout changes addressing only, never the arithmetic order: the shared-memory patch is still
stored as channel planes and the accumulation order per output channel is unchanged, so the FP32
result stays **bit-identical** to the pre-change baseline (`compare_f32.py` reports 0.000e+00 at
640x480), and all 25 dumped tensors still match the numpy reference to 1e-8 mean / 1e-6 max.

Per-pass time at 640x480, color only, FP32 tensors, same machine:

| Pass | Before | After | Pass | Before | After |
| --- | --- | --- | --- | --- | --- |
| dec_conv1a | 154.5 ms | 104.0 ms | up3 | 30.7 ms | 6.4 ms |
| enc_conv1 | 111.2 ms | 42.0 ms | pool1 | 27.8 ms | 4.4 ms |
| up4 | 80.2 ms | 16.4 ms | dec3b | 20.4 ms | 13.0 ms |
| dec_conv1b | 75.4 ms | 45.6 ms | cat2 | 10.3 ms | 2.4 ms |
| dec_conv2a | 74.6 ms | 49.6 ms | enc_conv0 | 39.6 ms | 12.0 ms |
| dec3a | 35.5 ms | 24.6 ms | **total** | **759.7 ms** | **382.4 ms** |

The three movement passes drop 4-5x (`up4` 80.2 -> 16.4, `pool1` 27.8 -> 4.4, `cat2` 10.3 -> 2.4),
which confirms the sector diagnosis, and the convolutions gain 1.5-2.6x from the vectorized staging
and stores. `dec_conv1a` is now the single biggest pass at 27% -- it is the one with 67 input
channels at full resolution, i.e. 11.8 GFLOP of pure FMA, and it runs at ~114 GFLOPS.

Whole-image timings, median of alternating FP32/FP16 runs:

| Image | Feature set | FP32 (before) | FP32 (blocked) | FP16 (blocked) | Tiles |
| --- | --- | --- | --- | --- | --- |
| 640x480 | 3ch | 608 ms | 449 ms | 411 ms | 1 |
| 640x480 | 9ch | 630 ms | 442 ms | 418 ms | 1 |
| 1024x768 | 3ch | 1515 ms | 1043 ms | 967 ms | 1 |
| 1024x768 | 9ch | 1570 ms | 1111 ms | 1150 ms | 1 |
| 1920x1080 | 3ch | 3843 ms | 2576 ms | 2456 ms | 1 |
| 1920x1080 | 9ch | 4101 ms | 3056 ms | 2690 ms | 1 |
| 3840x2160 | 3ch | 18002 ms | 11586 ms | 11288 ms | 6 / 2 (FP16) |
| 3840x2160 | 9ch | 19312 ms | 12121 ms | 12719 ms | 6 / 4 (FP16) |

The improvement is 25-40% depending on how warm the device is: the first measurements after a build
gave 608 -> 358 ms at 640x480 (-41%), sustained runs on this fanless Tiger Lake part settle 20%
lower and report 449 ms, so the table above is the conservative reading.

Note what happened to FP16: it is now worth only about 5-8% (and nothing at 4K, `1024 9ch` is
noise). With the sector amplification gone the convolutions are bound by FMA throughput rather than
by bytes, so the half/float conversions on every store become visible work. It stays the right
choice for the memory budget (half the tensor footprint, bigger tiles at 4K), not for speed.

### Where the time actually goes (`OIDN_PROFILE=1`)

`OIDN_PROFILE=1` inserts one timestamp query before each dispatch and one at the end of each tile
chain (it changes no dispatch, so the total stays comparable). At 640x480 with color only, FP32
tensors:

| Pass | ms | Share | Pass | ms | Share |
| --- | --- | --- | --- | --- | --- |
| dec_conv1a | 154.5 | 20.3% | dec3a | 35.5 | 4.7% |
| enc_conv1 | 111.2 | 14.6% | dec2b | 34.7 | 4.6% |
| up4 (nearest 2x into concat1) | 80.2 | 10.6% | up3 | 30.7 | 4.0% |
| dec_conv1b | 75.4 | 9.9% | pool1 | 27.8 | 3.7% |
| dec_conv2a | 74.6 | 9.8% | dec3b | 20.4 | 2.7% |
| enc_conv0 | 39.6 | 5.2% | cat2 | 10.3 | 1.4% |

The three pure data-movement passes (`up4`, `up3`, `cat2`) together cost 121 ms, 16% of the total,
for 161 MB, 69 MB and 20 MB of traffic respectively -- that is 2 GB/s where the device should do
twenty times that. The cause is the HWC layout combined with one channel per thread: a warp writes
32 four-byte words that are `channels*4` bytes apart, so each word lands in its own 64-byte sector
and the 4-byte useful payload pays for 16 bytes of sector traffic (a 16x amplification, exactly the
observed 2 GB/s versus ~32 GB/s). The convolution kernels have the same problem in their staging
loop (4 threads cover the 4 channels of one pixel = 16 useful bytes of a 64-byte sector), which is
why shrinking the element to FP16 saves far less than half the bytes: the sector count stays the
same. This is what the channel-blocked tensor layout above fixes (channels padded to a multiple of
4 and one chunk per thread); it takes the same 640x480 case from 759.7 ms to 382.4 ms, with `up4`
going 80.2 -> 16.4 ms and `pool1` 27.8 -> 4.4 ms.


### The same seven passes as C++ for OpenCL kernels (clspv)

Every pass exists twice: as GLSL (`src/oidn_*.comp`, compiler: glslc) and as C++ for OpenCL
(`src/oidn_*.cl`, compiler: clspv). Both sets are built into `smallpt` and
`oidn_denoise_tool`; `OIDN_BACKEND=glsl` picks the GLSL set, anything else (including no
variable) picks the clc set, which is the default because it measured faster on the same
input and reproduces the same image.

| option | kernels | entry point | parameters arrive |
| --- | --- | --- | --- |
| `OIDN_BACKEND=clc` (default) | `build/oidn/oidn_*_cl.spv`, `*_cl_h.spv` | kernel name (`oidn_pool` ...) | 64-byte buffer per pass |
| `OIDN_BACKEND=glsl` | `build/oidn/oidn_*.spv`, `*_h.spv` | `main` | 64-byte push constant |

Build command per pass and variant (CMake does this, shown here because none of it is optional):

```
clspv src/oidn_pool.cl -o build/oidn/oidn_pool_cl_raw.spv -O3 \
       --cl-std=CLC++2021 -inline-entry-points [-DOIDN_HALF_TENSOR=1]
spirv-opt build/oidn/oidn_pool_cl_raw.spv -o build/oidn/oidn_pool_cl.spv \
       --set-spec-const-default-value "0:8 1:8 2:1" --freeze-spec-const
```

Four things about clspv's output shaped the host side, all of them observed in the module
(`spirv-dis`) rather than assumed:

1. **No POD push constants in C++ mode.** `--pod-pushconstant` is rejected with
   `POD arguments as push constants are not compatible with module scope push constants`,
   because CLC++/CL2.0 always emits a module-scope 12-byte `PushConstantRegionOffset` block
   (OpenCL 2.0 global offsets). The same flags fail with `--global-offset`,
   `--global-offset-push-constant` and `--cl-std=CL2.0`. So the clc passes take their
   parameters from a `__global const uint *params` buffer instead: one 64-byte buffer per
   pass, uploaded with `vkCmdUpdateBuffer` right before the dispatch plus a
   TRANSFER_WRITE -> SHADER_READ barrier. The 12-byte global-offset block is still pushed
   once per command buffer, all zero.
2. **Bindings are numbered by the pointer arguments the kernel keeps, starting at 0**, so
   the numbering depends on which buffers a pass actually uses. The clc passes therefore
   list their arguments in the order they need them and the host writes a per-pass table
   (`clcBinding[][]` in `src/oidn_denoise.cpp`):
   `pre` = 0 src, 1 dst, 2 albedo, 3 normal, 4 params; `conv`/`conv_tiled` = 0 src, 1 dst,
   2 weights, 3 params; `pool`/`upsample`/`concat_copy`/`post` = 0 src, 1 dst, 2 params.
   Note the difference from GLSL, where albedo and normal sit at 3 and 4.
3. **The entry point keeps the kernel name.** `__kernel void main` is rejected
   (`kernel cannot be called 'main'`), and GLSL modules are built with `main`, so the
   pipeline creation takes the name from a per-backend table.
4. **`reqd_work_group_size` is ignored in CLC++ mode.** The workgroup size comes out as
   spec constants 0/1/2 with default 1/1/1, hence the `spirv-opt --freeze-spec-const`
   step that pins them to 8/8/1 (the same treatment the smallpt kernel gets).

The kernels themselves mirror the GLSL passes statement for statement. Two differences are
OpenCL's rather than stylistic: `__local` memory cannot be declared at module scope, so the
convolution's 18x18 patch lives inside the kernel and is passed to the helpers by pointer,
and the C++ sources are plain C++ (`struct`, `inline`, overload-free) with the OpenCL
address-space qualifiers on every pointer.

**Unrolling is what makes the clc convolution competitive.** The first clc build of
`oidn_conv_tiled` left the 2x2 pixel loops rolled, which kept the
`float acc[2][2][GROUP]` array in function memory: 164 `OpAccessChain` against the GLSL
module's 53, and a profiled total of 646.1 ms against GLSL's 426.9 ms (0.66x). Adding
`#pragma unroll` to the two 2-iteration pixel loops and the `GROUP` loop keeps the 32
accumulators in registers; the profiled total drops to 394.1 ms, i.e. below the GLSL
build, and every convolution pass gets faster.

Measured on the fixed 640x480 / 9-channel input used for the FP16 comparison
(`D:\clspv_build\out\oidn\ab_in_640.f32` plus its albedo / normal), 6 alternating rounds,
median of the board:

| setting | GLSL | clc (C++ for OpenCL) | change |
| --- | --- | --- | --- |
| 640x480, FP16 tensors (default) | 379.6 ms | **350.8 ms** | -7.6% |
| 640x480, FP32 tensors (`OIDN_STORE=fp32`) | 466.8 ms | **416.0 ms** | -10.9% |
| 1920x1080, FP16, single tile | 2648.4 ms | **2454.9 ms** | -7.3% |

Per pass (`OIDN_PROFILE=1`, 640x480, FP16, one tile, single run each -- the totals in that
mode include the timestamp queries, so only the per-pass column is meant to be compared):

| Pass | GLSL ms | clc ms | Pass | GLSL ms | clc ms |
| --- | --- | --- | --- | --- | --- |
| enc_conv0 | 40.53 | 17.81 | dec_conv3a | 34.04 | 37.62 |
| enc_conv1 | 108.28 | 59.98 | dec_conv2a | 51.27 | 50.26 |
| enc_conv2 | 24.84 | 18.04 | dec_conv1a | 132.17 | 119.52 |
| pool1 | 6.07 | 4.32 | dec_conv1b | 54.57 | 49.75 |
| pre | 2.11 | 2.35 | post | 0.20 | 0.23 |

Numerics, same input, against the GLSL output:

| tensors | max abs diff | mean rel diff | p99 abs diff | samples with rel > 1% |
| --- | --- | --- | --- | --- |
| FP32 | 4.959e-05 (range 15.69) | 4.885e-07 | 1.788e-06 | 0.000% |
| FP16 | 9.587e-02 | 3.369e-04 | 1.640e-03 | 0.009% |

The FP16 row is the same size as the FP16-versus-FP32 difference measured earlier
(8.0e-2 max, 7.6e-4 mean relative, 0.011% of samples above 1% relative error), so the
language switch hides inside the storage-precision noise rather than adding to it. On the
8-bit side, `smallpt` at 640x480/16spp with both backends differs in 16684 of 921600 samples
(1.81%), by at most 1 LSB, PSNR 65.55 dB.

Module sizes (`spirv-val`-clean, frozen workgroup size):

| Pass | GLSL FP32 | clc FP32 | GLSL FP16 | clc FP16 |
| --- | --- | --- | --- | --- |
| pre | 5740 | 39576 | 5848 | 39912 |
| conv | 5764 | 15684 | 5832 | 14712 |
| conv_tiled (clc: 3x2g4) | 8880 | 26860 | 9336 | 29264 |
| pool | 2432 | 4216 | 3752 | 5896 |
| upsample | 2224 | 2872 | 2524 | 3220 |
| concat_copy | 2220 | 2840 | 2520 | 3188 |
| post | 5124 | 72988 | 5216 | 73204 |

The clc modules are larger because the PU curve calls go through libclc rather than being
lowered to native instructions; `pre` and `post` are 52% of the total size here but 0.5% of
the runtime. `--use-native-builtins=pow,log --cl-fast-relaxed-math` would shrink them to
8072 B and 7028 B, at the cost of changing the transfer function's rounding, so it is not
enabled. The `conv_tiled` row is the shipped 3x2 blocking; the 2x2 build that the table above was
originally measured with is 37352 / 42204 B and still ships as
`oidn_conv_tiled_2x2g8_cl{,_h}.spv` for the A/B runs described in the next section.

### Register blocking: what ships, and why 4x4 lost

`src/oidn_conv_tiled.cl` is parameterised by three macros, so the blocking is a compile-time
choice rather than three hand-written kernels:

```
OIDN_PX / OIDN_PY   pixels one thread computes, which sets the block: 8*PX x 8*PY
OIDN_GROUP          output channels one work group covers
accumulators        PX*PY*GROUP floats per thread
staged patch        (8*PX+2) x (8*PY+2) floats per input channel, CHUNK = 4
```

The arithmetic argument for wider blocking is only about how many times a weight is fetched. One
(o, i) weight row is 12 floats (9 taps padded to 12) and covers 9*PX*PY multiply-adds, so the load
per MAC ratio is 12 / (9*PX*PY): 0.33 at 2x2, 0.22 at 3x2, 0.17 at 4x2, 0.083 at 4x4. `GROUP` does
not enter that ratio at all - it only amortises the staged voxel window - so a wider block with a
smaller group still fetches fewer weights per MAC.

On this GPU the wider blockings lose anyway, and the SPIR-V says why. `nvdisasm`-free evidence,
straight out of `spirv-dis`:

| Variant | accumulators | `OpVariable %_ptr_Function` | `OpStore` | 640x480 FP16, median of 4 |
| --- | --- | --- | --- | --- |
| 2x2g8 (baseline) | 16 | 0 | 16 | **372.55 ms** |
| 4x2g8 | 64 | 1 | 208 | 1422.4 ms (3.8x) |
| 4x4g8 | 128 | 1 | 408 | 3850.6 ms (10.3x) |
| 4x4g4 | 64 | 1 | 240 | 1032.9 ms (2.8x) |

One `OpVariable` in function storage plus 200-400 `OpStore` instructions means the accumulator
array stopped being promoted to registers and became per-thread scratch memory; the kernel then
spends its time writing and re-reading its own accumulators. The threshold is not simply "how many
floats": 3x2g8 (48 accumulators) spills while 4x3g4 (48 accumulators, plus a 5x6 voxel window)
does not, so whether the array is promoted also depends on the shape of the surrounding code.

Spill-free configurations were therefore the interesting ones, and none of the 4-wide blocks could
beat 2x2:

| Variant | accumulators | single-pass fp32 vs 2x2 (see below) | 640x480 fp16 end-to-end |
| --- | --- | --- | --- |
| 2x2g8 | 16 | reference | reference |
| 3x2g4 (block 24x16) | 24 | -4% .. +5%, shape dependent | **0.937 (10 alternating rounds)** |
| 4x2g4 (block 32x16) | 32 | +52% .. +74% | 1.103 (4 rounds) |
| 4x3g4 (block 32x24) | 48 | not measured | 1.004 (4 rounds) |
| 3x3g4 | 36 | not measured | 1.409 (4 rounds) |

The single-pass numbers come from `conv_probe` (built as a CMake target now), kernel time only,
real `rt_hdr.bin` weights, 3 rounds, median, `repeat` 8-12 dispatches inside one command buffer:

| Shape (w x h, cin, cout) | 2x2g8 | 3x2g4 | 4x2g4 |
| --- | --- | --- | --- |
| 640x480, 32 -> 64 | 45.81 ms | 48.30 ms | 69.74 ms |
| 320x240, 64 -> 64 | 22.93 ms | 21.96 ms | 39.94 ms |
| 160x120, 128 -> 128 | 25.84 ms | 23.06 ms | 39.12 ms |

So the 4-wide blocks are clearly out, and 3x2 is a wash on the largest layer while it wins on the
channel-heavy mid layers - which is where the network spends its time. End to end, on the fixed
640x480 / 9-channel input, 10 alternating rounds (the board throttles 15-40% under sustained load,
so only paired ratios mean anything):

```
per-round ratio 3x2g4/2x2g8  0.90 0.93 0.94 0.96 0.92 0.96 0.89 1.15 0.95 0.93
median 0.937, mean 0.954, 9 of 10 rounds faster
```

and with the FP32 tensor path 0.97 / 1.00 / 1.00 / 0.94 (median 0.985), at 1920x1080 FP16 2251.1 ms
against 2408.1 ms (-6.5%). The output is bit-identical to 2x2g8 - the per-pixel accumulation order
does not change, only the assignment of pixels to threads - which is why the switch was safe to make
the default: `smallpt` at 1024x768/256spp still produces the same PNG hash.

`OIDN_TILE=PXxPYgGROUP` runs one of the other modules built alongside the default
(`2x2g8`, `4x2g4`, `4x3g4`), and `conv_probe` times a single pass:

```powershell
$env:OIDN_TILE="2x2g8"; build\Release\oidn_denoise_tool.exe in.f32 640 480 out.f32
build\Release\conv_probe.exe build\oidn\oidn_conv_tiled_cl.spv tools\oidn\rt_hdr.bin 640 480 32 64 0 24 4 8 --clc 16
```

(`conv_probe` arguments: module, weights, width, height, cin, cout, weight offset, blockX,
group, repeat, `--clc` for the clspv module, blockY as the last argument. A blockY is needed because
the blocked kernels have rectangular blocks, and `--clc` switches the probe to the parameter buffer
plus the 12-byte push constant range the clspv builds use.)

### Limits

- `rt_hdr`, `rt_hdr_alb` and `rt_hdr_alb_nrm` are supported; the `cleanAux` variants
  (`rt_hdr_calb_cnrm` and friends) are not, because the renderer writes a first-hit G-buffer but
  never a prefiltered one. `SMALLPT_NO_AUX=1` forces the colour-only path for A/B comparison.
- Mirror and glass surfaces still lose their reflections, even with the normal map: they are delta
  BSDFs, so there is no radiance the network could reconstruct from neighbouring pixels. OIDN
  solves that by denoising the diffuse and specular parts separately and recombining them, which
  this example does not do.
- Any resolution works: the tile plan rounds the image up to 16 pixels and zero-fills the slack,
  exactly like OIDN's `input_reorder`. `SMALLPT_NO_DENOISE=1` skips the pass.
- Autoexposure is computed over the image you pass in (its zero padding included), the same way
  OIDN's `getAutoexposure(*color)` works, so the same content at 640x480 and 641x481 gets a
  slightly different exposure scale (0.4846 vs 0.4863 measured). That is semantics, not drift.
- A single tensor is still limited by `maxStorageBufferRange` (1 GB on this device, and
  `concat1` needs 272 bytes per pixel with FP32 tensors, 136 with FP16), so a tile budget above
  roughly 5 GB would fail to allocate; keep `OIDN_DENOISE_MAX_MB` below that.
- Tiling follows `core/unet.cpp`, but the tile size is derived from a plain budget over the
  tensor set rather than from OIDN's `buildNet(true)` scratch measurement, and the tensors are
  one big allocation per tensor instead of OIDN's single scratch block. The geometry (tile
  count, overlap, alignment offsets) is the same.
- Tensors live in host-visible, coherent memory, which is free on this integrated GPU but would be
  slow on a discrete card; there, allocating device-local memory and staging the input/output
  through a small host-visible buffer is the right setup.

Run it:

```powershell
build\Release\smallpt.exe build\smallpt.spv 640 480 16 renders\smallpt_denoised.png
$env:SMALLPT_NO_DENOISE=1; build\Release\smallpt.exe build\smallpt.spv 640 480 16 renders\smallpt_raw.png
# colour-only denoiser for A/B, and the albedo / normal G-buffer as raw float32 for inspection
$env:SMALLPT_NO_AUX=1; build\Release\smallpt.exe build\smallpt.spv 640 480 16 renders\smallpt_aux_off.png
$env:SMALLPT_DUMP_AUX="D:\tmp\g"; build\Release\smallpt.exe build\smallpt.spv 640 480 16 renders\a.png
$env:OIDN_DENOISE_MAX_MB=800; build\Release\smallpt.exe build\smallpt.spv 1920 1080 32 renders\big.png
$env:OIDN_CONV=simple; build\Release\smallpt.exe build\smallpt.spv 640 480 16 renders\smallpt_simple.png
```

For the standalone path (raw float32 in and out, no rendering):

```powershell
build\Release\oidn_denoise_tool.exe in.f32 640 480 out.f32 preview.ppm
# with auxiliary images (raw float32 RGB, width*height*3 each); --alb alone selects rt_hdr_alb
build\Release\oidn_denoise_tool.exe in.f32 640 480 out.f32 preview.ppm --alb alb.f32 --nrm nrm.f32
# FP16 tensor storage is the default (about 10% faster and half the tensor memory, see
# "FP16 tensor storage"); force the FP32 reference path with:
$env:OIDN_STORE="fp32"; build\Release\oidn_denoise_tool.exe in.f32 640 480 out.f32
# per-pass GPU timing breakdown
$env:OIDN_PROFILE="1"; build\Release\oidn_denoise_tool.exe in.f32 640 480 out.f32
# run the original GLSL pass set instead of the default C++ for OpenCL set
$env:OIDN_BACKEND="glsl"; build\Release\oidn_denoise_tool.exe in.f32 640 480 out.f32
# run a different register blocking of the tiled convolution (see "Register blocking")
$env:OIDN_TILE="2x2g8"; build\Release\oidn_denoise_tool.exe in.f32 640 480 out.f32
```

## Kernel languages: OpenCL C and C++ for OpenCL

clspv is not limited to C. `clspv --cl-std=...` accepts `CL1.0 / CL1.1 / CL1.2 / CLC++ / CLC++2021`,
so a kernel can be written in **C++ for OpenCL** (the Khronos subset described in the
"C++ for OpenCL 2021" spec), not in general-purpose C++. Compiled and measured here with the clspv
built from source in this repository:

```powershell
tools\clspv\bin\clspv.exe src\k.cl -o build\k.spv -O3 --cl-std=CLC++2021 -inline-entry-points
```

What compiled (a single test file using all of it at once, module 1812 B, `spirv-val` against
vulkan1.2 passes, reflection reports `k` with `a`/`b` at set 0 / binding 0,1):

- namespaces, `struct`/`class` with constructors and `const` member functions;
- operator overloading (`operator+`);
- function templates (`template <typename T>`);
- references (`float &r = b[i];`).

What does not compile (the frontend rejects them, so this is a hard limit rather than a bad build):

```
error: virtual functions are not supported in C++ for OpenCL
error: cannot use 'throw' with exceptions disabled
```

So no exceptions, no RTTI, no virtual dispatch or `override`, and no device-side standard library
(`new`/`delete`, `std::vector` and friends are unavailable; use the OpenCL builtins such as
`sqrt`, plus your own allocations through the kernel's buffer arguments).

Two gotchas specific to the C++ mode:

1. **`-inline-entry-points` is mandatory.** Without it clspv stops with
   `cannot compile languages that use the generic address space (e.g. CLC++, CL2.0) without
   -inline-entry-points`. C++ for OpenCL uses the generic address space by default, and the entry
   point has to be inlined before the address spaces can be resolved.
2. **`reqd_work_group_size` does not take effect.** Both spellings were tried:
   `__attribute__((reqd_work_group_size(64,1,1)))` is accepted silently, and
   `[[cl::reqd_work_group_size(64, 1, 1)]]` is reported as `unknown attribute ... ignored`; in both
   cases the module has **no `OpExecutionMode LocalSize`** and instead declares the workgroup size as
   spec constants (`OpDecorate ... SpecId 0/1/2`) whose **defaults are 1/1/1**. With
   `--cl-std=CL1.2` the same attribute does produce `LocalSize 64 1 1`, so this is specific to the
   C++ path.
   Supplying 8/8/1 through `VkSpecializationInfo` is not enough on this driver: the pipeline ran,
   but only one thread per workgroup executed (a 480x360 dispatch wrote exactly 60x45 pixels, and it
   finished in 0.6 ms). The reliable fix is to turn the spec constants into ordinary constants before
   loading the module, which is exactly the shape glslc emits for a GLSL workgroup size:

   ```powershell
   spirv-opt build\smallpt_cpp_raw.spv -o build\smallpt_cpp.spv ^
     --set-spec-const-default-value "0:8 1:8 2:1" --freeze-spec-const
   ```

   This step is part of the `smallpt` CMake target, which is the one that produces `smallpt.spv`.

A third quirk is not about size but about **global offsets**, and it bit this repository first: in
C++ for OpenCL (and OpenCL 2.0) semantics `get_global_id` is `global_offset + local id`, and clspv
reads that offset from a **12-byte push constant block**:

```
%5  = OpVariable %_ptr_PushConstant__struct_3 PushConstant
%86 = OpAccessChain %_ptr_PushConstant_uint %5 %uint_0 %uint_0
%87 = OpLoad %uint %86
%88 = OpIAdd %uint %87 %84          ; push constant offset + gl_GlobalInvocationID.x
```

`clspv-reflection` reports it as `pushconstant,name,region_offset,offset,0,size,12`. A host that
never pushes those bytes leaves the value undefined, every thread computes a garbage pixel index and
returns at the bounds check, and the whole dispatch finishes in a fraction of a millisecond with a
black image. `src/smallpt_main.cpp` therefore always declares a 12-byte push constant range and
pushes three zeros before the dispatch. Modules built with `--cl-std=CL1.2` simply ignore it.

#### Variant G: the operator-overloaded kernel (measured)

`src/smallpt_cpp.cl` keeps the same algorithm and the same arithmetic order as `src/smallpt.cl`, and
only rewrites the math as a class:

| A (OpenCL C) | G (C++ for OpenCL) |
| --- | --- |
| `typedef struct { float x, y, z; } Vec3;` | `struct Vec3 { float x, y, z; Vec3(); Vec3(float, float, float); ... };` |
| `add3(a, b)` | `a + b` (`operator+`) |
| `sub3(a, b)` | `a - b` (`operator-`) |
| `scl3(a, s)` | `a * s` and `s * a` (`operator*(float)` plus a `friend`) |
| `mul3(a, b)` (component-wise) | `a * b` (`operator*(Vec3)`) |
| `dot3`, `cross3`, `norm3` | `dot`, `cross`, `normalize` free functions |
| `float t; intersect(o, d, &t)` | `float t; intersect(o, d, t)` with `float &tOut` |
| `rng_next(&s)` | `rng.next()` |
| `trace(o, d, &s)` | `trace(const Vec3 &o, const Vec3 &d, Rng &rng)` |

The `intersect` and `rng_next` rows describe the original G; the current source replaced
`intersect()` (which returned the hit through references) with the `SCENE_HIT_TEST` macro block
described in "Which one `smallpt.exe` renders with today", because a hit struct passed by reference
is materialized in Function storage and reloaded (68 `Function` `OpVariable` and 182 `OpLoad` in the
module, against 9 and 44 for the macro block). The values are unchanged, which the `smallpt_exact`
build proves bit for bit.

Order of operations is deliberately identical, including `normalize` multiplying by `1/len` rather
than dividing, so the two variants produce the same bits:

| Run | A (OpenCL C) | G (C++ for OpenCL) | Ratio |
| --- | --- | --- | --- |
| 480x360, 32 spp (min of 3 interleaved rounds) | 1474.3 ms (14.99 Mpaths/s) | 1501.1 ms (14.73 Mpaths/s) | 1.02x slower |
| 1024x768, 1024 spp | 407.3 s (7.91 Mpaths/s) | 408.9 s (7.88 Mpaths/s) | 1.004x slower |
| SPIR-V size | 95716 B (93.5 KiB) | 97004 B (94.7 KiB) | +1288 B |
| Device memory | 3.14 MiB used | 3.14 MiB used | identical |
| Output (480x360/32spp) SHA256 | `5bb853b9...4158b5` | `5bb853b9...4158b5` | **byte identical** |
| Output (1024x768/1024spp) SHA256 | `4c888569...cb01f1b` | `4c888569...cb01f1b` | **byte identical** |

The 1024 spp PNG of G is byte-for-byte the same file as `renders/smallpt.png`, so no extra render is
committed for this variant. The 0.4-2% difference is the usual run-to-run noise on this iGPU; the
interesting point is that `-O3` inlines every overloaded operator back into the same instructions, so
abstraction costs nothing measurable here.

Two C++-only compile errors are worth recording, because both come from the generic address space
rather than from the C++ syntax:

```
error: incompatible operand types ('__private Vec3' and 'Vec3')
```

This hits `a ? b : c` whenever the two arms do not live in the same address space -- `dot(n,d) < 0 ?
n : -n` fails because `n` is generic while `-n` is a private temporary. The fix is to write the
selection as an `if` statement. The same applies to `?` between a temporary and a named object in
`hit_color` / `hit_emit` / `hit_mat`, which are written with `if` for that reason.


## Kernel code to Vulkan mapping

```c
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void vadd(__global const float *a, __global const float *b, __global float *out)
{
    const unsigned int gid = (unsigned int)get_global_id(0);
    out[gid] = a[gid] + b[gid];
}
```

| In the kernel | In Vulkan | Corresponding host code |
| --- | --- | --- |
| `__kernel void vadd(...)` | compute entry point named `vadd` | `pName = "vadd"` (**not** `"main"`) |
| n-th `__global` pointer argument | storage buffer at descriptor **set 0**, **binding n** | 3 `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` at bindings 0/1/2 |
| `get_global_id(0)` | `gl_GlobalInvocationID.x` | total work items of `vkCmdDispatch` = element count |
| `reqd_work_group_size(64,1,1)` | `OpExecutionMode LocalSize 64 1 1` | one work item per invocation, `16384 x 64 = 1048576` |

clspv's rule "`__global` pointer -> set=0, binding=argument index" is fixed. Verify it with the
bundled reflection tool (actual output from this repository):

```
$ tools\clspv\bin\clspv-reflection.exe shaders\vadd.spv
kernel_decl,vadd
kernel,vadd,arg,a,argOrdinal,0,descriptorSet,0,binding,0,offset,0,argKind,buffer
kernel,vadd,arg,b,argOrdinal,1,descriptorSet,0,binding,1,offset,0,argKind,buffer
kernel,vadd,arg,out,argOrdinal,2,descriptorSet,0,binding,2,offset,0,argKind,buffer
```

A few easy pitfalls, all handled in this example:

- **Entry point name**: clspv keeps the OpenCL kernel function name, while GLSL is always `main`.
  Mixing both kinds of module with the wrong `pName` produces
  `VUID-VkPipelineShaderStageCreateInfo-pName-00707` at `vkCreateComputePipelines`. On the Intel
  driver this was measured to crash the process outright, so do not get it wrong.
- **clspv command line order**: input file first, options after it; and `-O` is not a switch, it
  needs a level (`-O3` / `-O0`), otherwise the message is
  `clspv.exe: for the -O option: requires a value!`, which looks like a missing file.
- **Workgroup size is a trap**: without `reqd_work_group_size`, clspv emits `WorkgroupSize` as
  **spec constants** (SpecId 0/1/2, default `1,1,1`) and the module has no
  `OpExecutionMode LocalSize`. If the host then dispatches 64 threads per group, each group actually
  runs a single thread, so "only the first `dispatchGroupCount` elements are correct and the rest are
  zero". The first version here did exactly that and got 1031913 of 1M elements wrong. Three fixes:
  (a) add `reqd_work_group_size(64,1,1)` to the kernel (used here; the module then holds
  `LocalSize 64 1 1`); (b) set SpecId 0/1/2 to 64/1/1 through `VkSpecializationInfo` on the host
  (works for OpenCL C; the C++ for OpenCL path on this driver ignored it, see "Variant G");
  (c) freeze the spec constants into plain constants with
  `spirv-opt --set-spec-const-default-value "0:64 1:1 2:1" --freeze-spec-const`, which works
  regardless of driver support and is what the `smallpt_cpp` target does.
- **Global offsets in push constants**: C++ for OpenCL and OpenCL 2.0 kernels read the OpenCL global
  offset from a 12-byte push constant block, so a host that never pushes it gets undefined pixel
  indices. Declare the range and push zeros (done in `src/smallpt_main.cpp`).
- **variablePointers in Vulkan 1.1**: clspv modules emulate OpenCL pointer semantics with pointer
  arithmetic internally, and kernels with pointer arithmetic need `variablePointersStorageBuffer` /
  `variablePointers`. `main.cpp` queries support and enables them when available.
- **Scalar arguments**: this example deliberately uses pointer arguments only. Added scalar
  arguments such as `int n` are packed by clspv into a uniform buffer by default (binding after the
  pointer arguments); for a push constant use `--pod-pushconstant` and create a
  `VkPushConstantRange` on the host. For changes like this, export a mapping with `--descriptormap`
  rather than writing the host code from memory.
- **Loop bounds**: clspv's structured control flow pass needs to prove a loop terminates, and a
  dynamic bound read from a buffer makes it generate code that can hang the GPU. Loop bounds in the
  kernel must be compile-time constants (`#define` / specialization constant).

## Host-side steps (matching the numbered comments in `src/main.cpp`)

1. `vkCreateInstance` (no surface extension needed; `CLSPV_EXAMPLE_VALIDATION=1` enables the
   validation layers and the debug messenger)
2. Enumerate physical devices and pick one with a `VK_QUEUE_COMPUTE_BIT` queue family; query Vulkan
   1.1 features and enable `variablePointers*`
3. `vkCreateDevice` and fetch the compute queue
4. Create 3 `HOST_VISIBLE | HOST_COHERENT` storage buffers (the example skips staging buffers for
   simplicity; a real project would upload inputs into device-local memory through staging and read
   the output back through staging, which is much faster)
5. Descriptor set layout (3 storage buffers) + pool + set, write the descriptors
6. `vkCreateShaderModule` (reading `vadd.spv`) -> `vkCreateComputePipelines`, after which the shader
   module can be destroyed
7. Record the command buffer: `bindPipeline` -> `bindDescriptorSets` -> `dispatch(16384,1,1)` ->
   one `SHADER_WRITE -> HOST_READ` `vkCmdPipelineBarrier` (without this barrier the values the host
   reads back are not guaranteed to be visible by the spec)
8. `vkQueueSubmit` + `vkWaitForFences`
9. Compare the result element by element with `a[i] + b[i]` computed on the CPU and print PASS / FAIL
   (process exit code 0 / 1)
10. Destroy everything layer by layer

## About `shaders/vadd.spv`

**This SPIR-V was produced by a source-built clspv, with the same flags the CMake targets use**:
1308 bytes, entry point `vadd`, `LocalSize 64 1 1`, and bindings 0/1/2 in descriptor set 0:

```
clspv src\vadd.cl -o shaders\vadd.spv -O3
```

It is the same kernel as `build\vadd.spv`. Its purpose: on a machine without clspv the host flow
still works (`CMakeLists.txt` copies it with `configure_file` when clspv is not found). It is
deliberately kept in version control, which is why `.gitignore` has an explicit exception for it.

The GLSL reference implementation in `tools/glsl_equiv/` together with `make_fallback_spv.ps1` is
legacy: at the very beginning, before clspv existed on this machine, an equivalent module was built
through `glslc` -> `spirv-dis` -> (rename the `main` entry point to `vadd`) -> `spirv-as` to get the
host flow working. Now that the real clspv is available that path is kept only as a reference and is
no longer needed.

## Verified / not verified

Verified (Intel UHD on this machine, Vulkan 1.4.323 vendor driver):

- clspv builds from source: LLVM/Clang 24.0.0git + libclc builtins (spirv32/spirv64 vulkan) +
  `clspv.exe`, `clspv-reflection.exe`.
- `clspv src/vadd.cl -o vadd.spv -O3` passes `spirv-val --target-env vulkan1.2`, has entry point
  `vadd` and only the `Shader` capability (this kernel uses no builtins).
- `clspv-reflection` confirms the bindings: `a`/`b`/`out` = set 0 / binding 0,1,2.
- The whole host-side Vulkan compute flow works (device selection, descriptors, pipeline, dispatch,
  barrier, readback validation) with all 1M elements correct (`max |err| = 0`) and
  `VK_LAYER_KHRONOS_validation` enabled without any error.

Not verified:

- The Windows SDK bin directory is missing on this machine (`rc.exe` / `mt.exe` absent), which is why
  the clspv build sidesteps the check with Ninja plus a static-library self-test. On a machine with
  a complete SDK the official `-G Ninja` configure should work in one go, but that was not measured.
- The `spirv32-unknown-vulkan` builtins were built but no 32-bit SPIR-V artifact was ever dispatched.

## License

MIT, see [LICENSE](LICENSE).
