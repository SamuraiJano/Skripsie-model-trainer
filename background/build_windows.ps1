param(
    [string]$BuildDir = "build-msvc",
    [string]$OpenCvDir = "",
    [string]$OnnxRuntimeRoot = ""
)

$ErrorActionPreference = "Stop"

$backgroundRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $backgroundRoot

if (-not $OpenCvDir) {
    $OpenCvDir = Join-Path $repoRoot "third_party\opencv\build"
}
if (-not $OnnxRuntimeRoot) {
    $OnnxRuntimeRoot = Join-Path $repoRoot "third_party\onnxruntime-win-x64-1.27.0"
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "Could not find vswhere.exe. Install Visual Studio Build Tools with the C++ workload."
}

$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
    throw "Could not find Visual Studio C++ build tools."
}

$vsdev = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path -LiteralPath $vsdev)) {
    throw "Could not find VsDevCmd.bat under $vsInstall."
}

$buildPath = Join-Path $backgroundRoot $BuildDir
$cmd =
    '"' + $vsdev + '" -arch=x64 >nul && ' +
    'cmake -S "' + $backgroundRoot + '" -B "' + $buildPath + '" -G Ninja ' +
    '-DCMAKE_BUILD_TYPE=Release ' +
    '-DCMAKE_CXX_COMPILER=cl ' +
    '-DOpenCV_DIR="' + $OpenCvDir + '" ' +
    '-DONNXRUNTIME_ROOT="' + $OnnxRuntimeRoot + '" && ' +
    'cmake --build "' + $buildPath + '" --config Release --parallel'

cmd /c $cmd
