"""Entrypoint dell'exe PyInstaller: avvia l'app desktop di huebeat.

PyInstaller non supporta `-m huebeat`, quindi l'exe parte da questo script,
che delega alla CLI col sottocomando `app` inoltrando gli argomenti (es.
`--tray` per l'avvio automatico nascosto).
"""

import sys

from huebeat.cli import main

if __name__ == "__main__":
    sys.exit(main(["app", *sys.argv[1:]]))
