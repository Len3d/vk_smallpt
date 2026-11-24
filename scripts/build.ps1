# Configure, build, then run.
#
# Usage:
#   pwsh -File scripts\build.ps1
#   pwsh -File scripts\build.ps1 -Validation     # run once with the Vulkan validation layers

param(
  [switch]$Validation
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot

$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake -and (Test-Path -LiteralPath "C:\Program Files\CMake\bin\cmake.exe")) {
  $cmake = "C:\Program Files\CMake\bin\cmake.exe"
}
if (-not $cmake) { throw "cmake not found" }

& $cmake -S $root -B (Join-Path $root "build") -G "Visual Studio 17 2022" -A x64
& $cmake --build (Join-Path $root "build") --config Release

$exe = Join-Path $root "build\Release\vkcompute.exe"
if ($Validation) { $env:CLSPV_EXAMPLE_VALIDATION = "1" }
& $exe
$code = $LASTEXITCODE
Remove-Item Env:\CLSPV_EXAMPLE_VALIDATION -ErrorAction SilentlyContinue
exit $code
