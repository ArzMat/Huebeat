# Compila hb_loopback.exe con i Visual Studio 2022 Build Tools (cl.exe x64).
# Uso: powershell -ExecutionPolicy Bypass -File build.ps1

$ErrorActionPreference = "Stop"

$vsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $vsDevCmd)) {
    Write-Error "VsDevCmd.bat non trovato: $vsDevCmd"
}

$here = $PSScriptRoot
$src = Join-Path $here "main.cpp"
$exe = Join-Path $here "hb_loopback.exe"
$obj = Join-Path $here "main.obj"

# cmd /c: carica l'ambiente dei Build Tools e compila nella stessa shell
$cmdLine = "`"$vsDevCmd`" -arch=x64 -no_logo && cl /nologo /std:c++17 /W4 /O2 /EHsc `"$src`" /Fe:`"$exe`" /Fo:`"$obj`" /link ole32.lib mmdevapi.lib"
cmd /c $cmdLine

if ($LASTEXITCODE -ne 0) {
    Write-Error "Compilazione fallita con exit code $LASTEXITCODE"
}

Write-Host "OK: $exe"
