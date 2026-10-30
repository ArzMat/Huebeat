"""Percorsi dei file di huebeat: sviluppo (repo) contro installazione (exe).

Tutti i file di stato (config.toml, config.local.toml, .env, token Spotify,
credenziali dei bridge, log) vivono in un'unica cartella "home" risolta qui:

- override esplicito con la variabile d'ambiente HUEBEAT_HOME (utile nei test);
- in SVILUPPO (processo non congelato e config.toml presente nella cwd, cioe'
  la root del repo) la home e' la cwd: comportamento storico invariato;
- altrimenti (exe PyInstaller, o avvio da una cartella qualsiasi) la home e'
  %APPDATA%\\huebeat, creata al primo accesso: un exe puo' stare ovunque e i
  suoi dati restano in un posto solo.
"""

import os
import sys
from pathlib import Path

# Nome della cartella dati dentro %APPDATA%.
APP_DIR_NAME = "huebeat"
# Variabile d'ambiente che forza la home (test, setup particolari).
HOME_ENV_VAR = "HUEBEAT_HOME"
# File il cui essere presente nella cwd identifica la root del repo (sviluppo).
DEV_MARKER_FILENAME = "config.toml"

# Nomi dei file di stato dentro la home.
CONFIG_FILENAME = "config.toml"
LOCAL_CONFIG_FILENAME = "config.local.toml"
ENV_FILENAME = ".env"
SPOTIFY_TOKENS_FILENAME = ".spotify_tokens.json"
BRIDGES_FILENAME = ".bridges.json"
LOG_FILENAME = "huebeat.log"
# Lockfile di istanza singola del motore: un solo processo alla volta puo'
# pilotare le luci (evita il ping-pong di due run sullo stesso slot DTLS del
# bridge). Il lock e' un byte-range lock Windows (msvcrt), rilasciato alla
# chiusura del processo: non resta orfano dopo un crash.
ENGINE_LOCK_FILENAME = ".engine.lock"


def is_frozen() -> bool:
    """Indica se il processo gira da un eseguibile PyInstaller.

    Returns:
        True se congelato (attributo sys.frozen impostato dal bootloader).
    """
    return bool(getattr(sys, "frozen", False))


def data_dir() -> Path:
    """Risolve (e crea se serve) la cartella home dei file di huebeat.

    Returns:
        Il percorso della home secondo le regole del modulo.
    """
    override = os.environ.get(HOME_ENV_VAR)
    if override:
        home = Path(override)
    elif not is_frozen() and (Path.cwd() / DEV_MARKER_FILENAME).exists():
        return Path.cwd()
    else:
        home = Path(os.environ.get("APPDATA", str(Path.home()))) / APP_DIR_NAME
    home.mkdir(parents=True, exist_ok=True)
    return home
