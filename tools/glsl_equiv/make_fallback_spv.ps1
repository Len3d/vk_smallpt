# Build the placeholder SPIR-V used to validate the host flow when clspv is not available.
#
# Why the extra step: glslc / glslangValidator always name the GLSL entry point main, while clspv
# keeps the OpenCL C kernel function name (vadd). The host code passes "vadd" as pName (because that
# is what a clspv module looks like), so this script renames the entry point of the GLSL module to
# vadd, making both modules look identical to the host.
#
# Usage:  pwsh -File tools\glsl_equiv\make_fallback_spv.ps1

$ErrorActionPreference = "Stop"

$root    = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # repository root
$sdkBin  = Join-Path $env:VULKAN_SDK "Bin"
$glslc   = Join-Path $sdkBin "glslc.exe"
$dis     = Join-Path $sdkBin "spirv-dis.exe"
$as      = Join-Path $sdkBin "spirv-as.exe"

foreach ($t in @($glslc, $dis, $as)) {
  if (-not (Test-Path -LiteralPath $t)) { throw "$t not found, set VULKAN_SDK first" }
}

$src      = Join-Path $root "tools\glsl_equiv\vadd.comp"
$out      = Join-Path $root "shaders\vadd.spv"
$tmpSpv   = Join-Path $env:TEMP "vadd_glsl.spv"
$tmpAsm   = Join-Path $env:TEMP "vadd_glsl.spvasm"

& $glslc -fshader-stage=compute $src -o $tmpSpv
if ($LASTEXITCODE -ne 0) { throw "glslc failed" }

& $dis $tmpSpv -o $tmpAsm
if ($LASTEXITCODE -ne 0) { throw "spirv-dis failed" }

$text = [IO.File]::ReadAllText($tmpAsm)
$text = $text.Replace('OpEntryPoint GLCompute %main "main"', 'OpEntryPoint GLCompute %main "vadd"')
$text = $text.Replace('OpName %main "main"', 'OpName %main "vadd"')
[IO.File]::WriteAllText($tmpAsm, $text, (New-Object Text.UTF8Encoding $false))

& $as --target-env vulkan1.1 $tmpAsm -o $out
if ($LASTEXITCODE -ne 0) { throw "spirv-as failed" }

& $dis $out | Select-String -Pattern "OpEntryPoint"
Write-Host "wrote $out"
