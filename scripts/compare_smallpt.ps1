# compare_smallpt.ps1 -- compare several implementations of the same kernel under identical
# parameters: clspv (__constant scene) / clspv (scene in a buffer) / clspv (scene in a buffer with
# native sin/cos) / clspv (literal subscripts) / clspv (C++ for OpenCL, operator overloading) /
# GLSL (glslc) / GLSL (literal subscripts).
#
# Usage:
#   pwsh -File scripts\compare_smallpt.ps1
#   pwsh -File scripts\compare_smallpt.ps1 -Width 1024 -Height 768 -Spp 256
#
# Every case runs the same host (smallpt.exe); only the SPIR-V module and the entry point name differ:
#   A  build\smallpt_c.spv          entry smallpt (src/smallpt.cl, OpenCL C, scene embedded as __constant)
#   B  build\smallpt_buf.spv        entry smallpt (src/smallpt_buf.cl, scene read from the params buffer)
#   C  build\smallpt_glsl.spv       entry main    (src/smallpt.comp, GLSL built by glslc)
#   D  build\smallpt_native.spv     entry smallpt (src/smallpt_buf.cl, native sin/cos + fast math)
#   E  build\smallpt_lit.spv        entry smallpt (src/smallpt_lit.cl, literal subscripts)
#   F  build\smallpt_lit_glsl.spv   entry main    (src/smallpt_lit.comp, GLSL with literal subscripts)
#   G  build\smallpt_cpp.spv        entry smallpt (src/smallpt_cpp.cl, C++ for OpenCL, plain -O3)
#   H  build\smallpt.spv            entry smallpt (src/smallpt_cpp.cl, C++ for OpenCL, native sin/cos +
#                                                fast math; the module the smallpt target renders with)
#
# The denoiser is switched off for these runs (SMALLPT_NO_DENOISE=1), so the comparison is about the
# path-traced estimate only. The cases are timings and image equivalence, not a benchmark suite.

param(
  [int]$Width = 480,
  [int]$Height = 360,
  [int]$Spp = 32,
  [string]$OutDir = "",
  [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root "build"
$exe = Join-Path $build "Release\smallpt.exe"
if ($OutDir -eq "") { $OutDir = Join-Path $root "renders\compare" }
if (-not (Test-Path -LiteralPath $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake -and (Test-Path -LiteralPath "C:\Program Files\CMake\bin\cmake.exe")) {
  $cmake = "C:\Program Files\CMake\bin\cmake.exe"
}

if (-not $SkipBuild) {
  if (-not $cmake) { throw "cmake not found" }
  & $cmake -S $root -B $build -G "Visual Studio 17 2022" -A x64 | Out-Null
  & $cmake --build $build --config Release --target smallpt | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

$cases = @(
  @{ Name = "A: clspv  OpenCL C, __constant scene"; Spv = (Join-Path $build "smallpt_c.spv");        Entry = "smallpt"; Ppm = (Join-Path $OutDir "A.ppm") },
  @{ Name = "B: clspv  scene in buffer";           Spv = (Join-Path $build "smallpt_buf.spv");      Entry = "smallpt"; Ppm = (Join-Path $OutDir "B.ppm") },
  @{ Name = "C: GLSL   (glslc)";                   Spv = (Join-Path $build "smallpt_glsl.spv");     Entry = "main";    Ppm = (Join-Path $OutDir "C.ppm") },
  @{ Name = "D: clspv  buffer + native sin/cos";   Spv = (Join-Path $build "smallpt_native.spv");   Entry = "smallpt"; Ppm = (Join-Path $OutDir "D.ppm") },
  @{ Name = "E: clspv  literal subscripts";        Spv = (Join-Path $build "smallpt_lit.spv");      Entry = "smallpt"; Ppm = (Join-Path $OutDir "E.ppm") },
  @{ Name = "F: GLSL   literal subscripts";        Spv = (Join-Path $build "smallpt_lit_glsl.spv"); Entry = "main";    Ppm = (Join-Path $OutDir "F.ppm") },
  @{ Name = "G: clspv  C++ for OpenCL";            Spv = (Join-Path $build "smallpt_cpp.spv");      Entry = "smallpt"; Ppm = (Join-Path $OutDir "G.ppm") },
  @{ Name = "H: clspv  C++ + native/fast (default)"; Spv = (Join-Path $build "smallpt.spv");        Entry = "smallpt"; Ppm = (Join-Path $OutDir "H.ppm") }
)

# The comparison is about the kernel, so skip the denoiser: the PPM bytes are the path-traced estimate.
$env:SMALLPT_NO_DENOISE = "1"

$rows = @()
foreach ($c in $cases) {
  if (-not (Test-Path -LiteralPath $c.Spv)) {
    Write-Warning "missing $($c.Spv), skipping $($c.Name)"
    continue
  }
  $out = & $exe $c.Spv $Width $Height $Spp $c.Ppm $c.Entry 2>&1
  $text = ($out | Out-String)
  function Grab([string]$pattern) {
    $m = [regex]::Match($text, $pattern)
    if ($m.Success) { return [double]$m.Groups[1].Value } else { return [double]::NaN }
  }
  $rows += [pscustomobject]@{
    Implementation = $c.Name
    ElapsedMs      = [math]::Round((Grab "elapsed: ([\d\.]+) ms"), 1)
    MpathsPerSec   = [math]::Round((Grab "([\d\.]+) Mpaths/s"), 2)
    ModuleKiB      = [math]::Round((Grab "SPIR-V module: ([\d\.]+) KiB"), 1)
    DeviceMemMiB   = [math]::Round((Grab "used ([\d\.]+) MiB"), 2)
  }
}

$rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

# Are the images equivalent (statistically)?
# ppm_diff.py only needs the standard library; any python 3 on PATH will do.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue).Source }
if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
$ppmDiff = Join-Path $PSScriptRoot "ppm_diff.py"
$pc = Join-Path $OutDir "C.ppm"
$pf = Join-Path $OutDir "F.ppm"
if ($py -and (Test-Path -LiteralPath $pc) -and (Test-Path -LiteralPath $pf) -and (Test-Path -LiteralPath $ppmDiff)) {
  Write-Host "C (GLSL, original) vs F (GLSL, literal subscripts) image difference:"
  & $py $ppmDiff $pc $pf
} else {
  Write-Warning "no python or no images, skipping the C/F comparison"
}

# A and G are expected to be byte identical: G only rewrites the math as operator overloads and keeps
# the arithmetic order, so any difference here means the rewrite changed something.
$pa = Join-Path $OutDir "A.ppm"
$pg = Join-Path $OutDir "G.ppm"
if ($py -and (Test-Path -LiteralPath $pa) -and (Test-Path -LiteralPath $pg) -and (Test-Path -LiteralPath $ppmDiff)) {
  Write-Host "A (OpenCL C) vs G (C++ for OpenCL) image difference:"
  & $py $ppmDiff $pa $pg
}
