# Build dell'exe di huebeat (app desktop) con PyInstaller.
# Uso: .\build_exe.ps1  (dalla root del repo, col venv creato)
# Output: dist\huebeat\huebeat.exe (cartella onedir, da distribuire intera).
#
# Note di packaging (imparate empiricamente):
# - pywebview carica il backend a runtime e ha dati propri -> --collect-all webview.
# - pystray sceglie il backend win32 a runtime -> --collect-submodules pystray.
# - la UI (webui/index.html) e' un data file del pacchetto.
# - --noconsole: niente finestra terminale; i log vanno su file nella home
#   (%APPDATA%\huebeat\huebeat.log) via huebeat.paths quando frozen.

# L'helper nativo di cattura per-processo va compilato prima (una tantum).
if (-not (Test-Path "native\hb_loopback\hb_loopback.exe")) {
    Write-Output "hb_loopback.exe mancante: lo compilo con native\hb_loopback\build.ps1..."
    & .\native\hb_loopback\build.ps1
    if (-not (Test-Path "native\hb_loopback\hb_loopback.exe")) {
        Write-Output "BUILD FALLITA: helper nativo non compilato."
        exit 1
    }
}

# Icona dell'app (facoltativa): assets\huebeat.ico, usata per l'exe e
# inclusa nel bundle per la tray. Senza file la build resta identica.
$iconArgs = @()
if (Test-Path "assets\huebeat.ico") {
    $iconArgs = @("--icon", "assets\huebeat.ico", "--add-data", "assets\huebeat.ico;assets")
}

# NB: niente $ErrorActionPreference = "Stop": PyInstaller logga su stderr e
# PowerShell 5.1 lo tratterebbe come errore fatale. Si controlla l'exit code.
& .\.venv\Scripts\python.exe -m PyInstaller `
    --noconfirm `
    --noconsole `
    --name huebeat `
    --add-data "huebeat\webui\index.html;huebeat\webui" `
    --add-binary "native\hb_loopback\hb_loopback.exe;." `
    --collect-all webview `
    --collect-submodules pystray `
    @iconArgs `
    launcher_app.py

if ($LASTEXITCODE -ne 0) {
    Write-Output "BUILD FALLITA (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}
# Il README viaggia col bundle (PyInstaller rigenera dist\huebeat a ogni build).
Copy-Item README.md dist\huebeat\README.md -Force
Write-Output ""
Write-Output "Build completata: dist\huebeat\huebeat.exe"
Write-Output "Distribuire l'INTERA cartella dist\huebeat (l'exe da solo non basta)."
