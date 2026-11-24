# render_smallpt.ps1 -- build smallpt and render it straight to a PNG.
#
# Usage:
#   pwsh -File scripts\render_smallpt.ps1
#   pwsh -File scripts\render_smallpt.ps1 -Width 1024 -Height 768 -Spp 256
#
# Output: renders\smallpt.png. The host encodes the PNG itself, so there is no
# conversion step and no Python dependency.

param(
  [int]$Width = 640,
  [int]$Height = 480,
  [int]$Spp = 64,
  [string]$OutPng = "",
  [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$renders = Join-Path $root "renders"
if ($OutPng -eq "") { $OutPng = Join-Path $renders "smallpt.png" }

if (-not (Test-Path -LiteralPath $renders)) {
  New-Item -ItemType Directory -Path $renders | Out-Null
}

$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake -and (Test-Path -LiteralPath "C:\Program Files\CMake\bin\cmake.exe")) {
  $cmake = "C:\Program Files\CMake\bin\cmake.exe"
}
if (-not $cmake) { throw "cmake not found, add it to PATH or run cmake --build manually" }

if (-not $SkipBuild) {
  & $cmake -S $root -B (Join-Path $root "build") -G "Visual Studio 17 2022" -A x64 | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
  & $cmake --build (Join-Path $root "build") --config Release --target smallpt | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

$exe = Join-Path $root "build\Release\smallpt.exe"
$spv = Join-Path $root "build\smallpt.spv"
if (-not (Test-Path -LiteralPath $exe)) { throw "missing $exe (build the smallpt target first)" }
if (-not (Test-Path -LiteralPath $spv)) { throw "missing $spv (did the clspv step fail?)" }

& $exe $spv $Width $Height $Spp $OutPng
if ($LASTEXITCODE -ne 0) { throw "render failed" }
Write-Host "done: $OutPng"
