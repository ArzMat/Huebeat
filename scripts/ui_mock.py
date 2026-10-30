"""Server mock per collaudare la UI di huebeat senza motore ne' hardware.

Avvia il ControlServer reale (index.html del repo) su 127.0.0.1:8799 con uno
snapshot finto realistico: card IN RIPRODUZIONE popolata e brano "in play",
cosi' i controlli playback sono attivi e ispezionabili da un browser.
"""

import sys
import time
from pathlib import Path

# Eseguibile da qualsiasi cwd: la root del repo entra nel path dei moduli.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huebeat.control import ControlBus, ControlServer  # noqa: E402

PORT = 8799

SNAPSHOT = {
    "area": "Area di prova",
    "channel_count": 11,
    "fps": 60,
    "uptime_seconds": 4242,
    "effect": "adaptive",
    "adaptive_mode": "spectrum",
    "levels": {"bass": 0.62, "mid": 0.35, "high": 0.18, "voice": 0.4},
    "onset": 0.3,
    "bpm": 122.0,
    "bpm_locked": True,
    "beat_confidence": 0.8,
    "section_energy": 0.55,
    "warmth": 0.1,
    "palette": [
        (0.9, 0.2, 0.5),
        (0.2, 0.8, 0.9),
        (0.5, 0.3, 0.9),
        (0.9, 0.6, 0.2),
        (0.2, 0.9, 0.4),
        (0.8, 0.2, 0.2),
        (0.3, 0.4, 0.9),
    ],
    "track": {
        "title": "Brano di prova",
        "artists": "Artista di prova",
        "is_playing": True,
        "volume": 65,
    },
    "device": "Cuffie (Bluetooth)",
    "audio_source": "device",
    "source_target": "",
    "reconnecting": False,
    "lights": [],
    "bridges": [
        {
            "id": "mock",
            "name": "Hue Bridge",
            "area": "Area di prova",
            "channel_count": 11,
            "active": True,
            "reconnecting": False,
            "error": "",
        }
    ],
    "hotkey": "ctrl+shift+a",
}


def main() -> None:
    bus = ControlBus("adaptive")
    bus.snapshot = SNAPSHOT
    bus.engine_running = True
    server = ControlServer(bus, PORT)
    server.start()
    print(f"UI mock su http://127.0.0.1:{PORT}/ (Ctrl+C per uscire)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
