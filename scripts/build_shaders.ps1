# Compile src\vadd.cl into shaders\vadd.spv with clspv.
#
# Usage:
#   pwsh -File scripts\build_shaders.ps1
#
# When clspv is not on PATH, point CLSPV_ROOT at its install/build directory:
#   $env:CLSPV_ROOT = "C:\dev\clspv\build"; pwsh -File scripts\build_shaders.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$src  = Join-Path $root "src\vadd.cl"
$out  = Join-Path $root "shaders\vadd.spv"

$clspv = (Get-Command clspv -ErrorAction SilentlyContinue).Source
if (-not $clspv) {
  # Lookup order: CLSPV_ROOT\bin, then the in-tree tools\clspv\bin
  foreach ($dir in @($env:CLSPV_ROOT, (Join-Path $root "tools\clspv"))) {
    if (-not $dir) { continue }
    $candidate = Join-Path $dir "bin\clspv.exe"
    if (Test-Path -LiteralPath $candidate) { $clspv = $candidate; break }
  }
}

if (-not $clspv) {
  Write-Host "clspv not found."
  Write-Host "clspv does not ship with the Vulkan SDK; you have to build it yourself"
  Write-Host "(together with LLVM/Clang/libclc). The full build notes are in the README section"
  Write-Host "'Building clspv from source'."
  Write-Host ""
  Write-Host "This repository may contain a source build under tools\clspv\bin. If it is gone,"
  Write-Host "set CLSPV_ROOT to your own clspv build directory (it looks for <CLSPV_ROOT>\bin\clspv.exe)."
  exit 1
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $out) | Out-Null
# Note: clspv takes the input file first and the options after it,
# and -O is not a flag: it needs a level (-O3 / -O0), otherwise it reports "requires a value".
& $clspv $src -o $out -O3
if ($LASTEXITCODE -ne 0) { throw "clspv failed" }

Write-Host "wrote $out"
Write-Host "To verify the bindings: clspv-reflection $out"
