"""Configurazione centralizzata di huebeat.

Unico punto del codebase che legge config.toml e .env: tutto il resto
importa `settings` da qui.
"""

import logging
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from huebeat import paths

logger = logging.getLogger(__name__)

# Home dei file di stato, risolta una volta all'import (repo in sviluppo,
# %APPDATA%\huebeat da exe: vedi huebeat.paths). I file possono anche non
# esistere ancora (primo avvio): pydantic-settings li tollera.
_HOME = paths.data_dir()

# Suffisso con cui si mette da parte un config.local.toml illeggibile (file
# GENERATO dal frontend): non deve mai impedire l'avvio.
CORRUPT_SUFFIX = ".corrupt"
# Marker leggibile scritto nella home quando config.local.toml viene messo da
# parte: da exe --noconsole lo stderr non si vede, l'errore d'avvio deve
# restare comunque su file.
STARTUP_ERROR_FILENAME = "startup-error.log"


class TuningSettings(BaseModel):
    """Knob di tuning persistiti (sezione [tuning], scritta dal frontend).

    Stessi nomi e default di `huebeat.tuning.LiveTuning`: all'avvio i valori
    persistiti vengono travasati nell'istanza runtime `tuning.live`.
    """

    punch_gain: float = 0.6
    level_gain: float = 0.85
    bass_contrast_gamma: float = 2.0
    punch_half_life_seconds: float = 0.25
    calm_energy_on: float = 0.35
    burst_density_on: float = 7.0
    min_dwell_seconds: float = 2.0
    drift_period_seconds: float = 30.0
    crossfade_seconds: float = 2.0
    hue_rotation_seconds_per_turn: float = 720.0
    drop_intensity: float = 1.0
    stereo_gain: float = 0.35
    beat_predict: float = 0.5
    # Knob (contratto con huebeat.tuning): dynamics_gain e hue_offset in
    # 0-1, gamma in 0.4-2.5. I range e il clamp vivono in tuning.TUNING_SPECS.
    dynamics_gain: float = 0.0
    hue_offset: float = 0.0
    gamma: float = 1.0
    # Knob ondata 3 (contratto con huebeat.tuning): harmony_gain e mirror in
    # 0-1. harmony_gain attivo di fabbrica (0.5) come drop_intensity; mirror 0.
    harmony_gain: float = 0.5
    mirror: float = 0.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_HOME / paths.ENV_FILENAME),
        env_file_encoding="utf-8",
        # Il secondo fa override del primo (config.local.toml e' generato).
        toml_file=(
            str(_HOME / paths.CONFIG_FILENAME),
            str(_HOME / paths.LOCAL_CONFIG_FILENAME),
        ),
        extra="ignore",
        case_sensitive=False,
    )

    # --- da config.toml: parametri non sensibili ---
    bridge_host: str = ""  # vuoto = auto-discovery (percorso legacy a bridge singolo)
    entertainment_area: str = ""  # vuoto = prima area disponibile (legacy)
    # Multi-bridge: mappa bridge_id -> area da sincronizzare.
    # Valori: "" = quel bridge NON sincronizza; "auto" = prima area; altrimenti
    # l'id dell'area. Mappa vuota = default compat: con un solo bridge si
    # sincronizza in "auto", con piu' bridge si parte tutti spenti (si sceglie
    # dalla UI, sezione BRIDGE). Scritta dal frontend in config.local.toml.
    bridge_sync: dict[str, str] = {}
    stream_fps: int = Field(60, ge=10, le=60)
    audio_device: str = ""  # vuoto = loopback dell'output di default
    # Sorgente audio: "device" = loopback di un'uscita intera (audio_device);
    # "process" = solo l'audio dell'app indicata in audio_process (helper
    # nativo Process Loopback; l'app puo' anche non essere ancora avviata).
    audio_source: Literal["device", "process"] = "device"
    audio_process: str = ""  # nome processo (es. "Spotify.exe") per audio_source="process"
    audio_block_size: int = Field(1024, ge=128, le=8192)
    invert_x: bool = False  # specchia sinistra/destra dell'effetto spaziale
    # Effetto luce: adaptive (default: tre regimi - pulse in calma, spectrum sul
    # ritmo normale, waterfall sulle raffiche), spectrum (spettro spaziale
    # storico), chase (flash che corre), pulse (l'area respira sul beat),
    # auto (pulse quieto / spectrum carico), waterfall (i colpi scorrono).
    # Il default vale solo per installazioni fresche: chi ha gia' un
    # config.local.toml conserva l'effetto salvato.
    effect: Literal["spectrum", "chase", "pulse", "auto", "waterfall", "adaptive"] = "adaptive"
    # Ruoli forzati per canale (sezione TOML [channel_roles]: channel_id = ruolo).
    # "rest" = tutto tranne i bassi (medi+alti); "voice" = presenza vocale
    # stimata (euristica); "drums" = componente percussiva (HPSS); "wash" =
    # componente armonica (tappeto). I canali non elencati seguono la
    # distribuzione spaziale automatica. Onorati dall'effetto spectrum; gli
    # altri li ignorano.
    channel_roles: dict[int, Literal["bass", "mid", "high", "rest", "voice", "drums", "wash"]] = {}
    # Ruoli per-bridge: bridge_id -> {channel_id -> ruolo}. Per il
    # bridge indicato ha precedenza sul flat channel_roles; i bridge senza
    # mappa propria usano il flat (compat a bridge singolo). Sezioni TOML
    # [channel_roles_by_bridge.<bridge_id>], scritte dal frontend.
    channel_roles_by_bridge: dict[
        str, dict[int, Literal["bass", "mid", "high", "rest", "voice", "drums", "wash"]]
    ] = {}
    # Hotkey globale per sospendere/riattivare le luci durante run (formato
    # della libreria keyboard, es. "ctrl+shift+a"); vuota = disattivata.
    toggle_hotkey: str = ""
    # Porta TCP del server di controllo del frontend (bind su 127.0.0.1);
    # 0 = server disattivato (run torna al comportamento senza frontend).
    control_port: int = Field(8765, ge=0, le=65535)
    # Knob di tuning persistiti (sezione [tuning] in config.local.toml,
    # scritta dal frontend; default = costanti storiche dei moduli).
    tuning: TuningSettings = TuningSettings()
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- da .env: secrets del Bridge ---
    # Opzionali a livello di modello: al primo avvio non esistono ancora
    # (si ottengono col pairing). Il check di presenza lo fa l'entrypoint.
    hue_username: SecretStr | None = None
    hue_clientkey: SecretStr | None = None
    # Client id Spotify (chiave SPOTIFY_CLIENT_ID nel .env). Non e' un vero
    # segreto per un public client PKCE, ma vive nel .env con le altre
    # credenziali per tenere tutte le config sensibili in un unico posto.
    spotify_client_id: SecretStr | None = None

    def bridge_paired(self) -> bool:
        """
        Indica se il pairing col Bridge risulta gia' fatto.

        Returns:
            True se username e clientkey sono presenti e non vuoti.
        """
        return bool(
            self.hue_username
            and self.hue_username.get_secret_value()
            and self.hue_clientkey
            and self.hue_clientkey.get_secret_value()
        )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def _quarantine_path(local_path: Path) -> Path:
    """Sceglie un nome .corrupt libero per il config.local.toml da mettere da parte.

    Args:
        local_path: Percorso del config.local.toml corrotto.

    Returns:
        Un percorso .corrupt non ancora esistente (con timestamp, ed
        eventuale contatore, se il nome base e' gia' occupato).
    """
    candidate = local_path.with_name(local_path.name + CORRUPT_SUFFIX)
    if not candidate.exists():
        return candidate
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = local_path.with_name(f"{local_path.name}{CORRUPT_SUFFIX}.{stamp}")
    counter = 1
    while candidate.exists():
        candidate = local_path.with_name(f"{local_path.name}{CORRUPT_SUFFIX}.{stamp}.{counter}")
        counter += 1
    return candidate


def _write_startup_marker(quarantined: Path, error: Exception) -> None:
    """Scrive un marker leggibile con l'errore d'avvio e dove sta il file corrotto.

    Self-contained: scrive direttamente su file senza dipendere dal logging, che
    all'import di questo modulo non e' ancora configurato (vedi cli.main). Cosi'
    l'errore resta visibile anche da exe --noconsole, dove lo stderr non c'e'.

    Args:
        quarantined: Percorso in cui e' stato spostato il config.local.toml.
        error: L'eccezione che ha reso illeggibile il file generato.
    """
    marker = _HOME / STARTUP_ERROR_FILENAME
    text = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - config.local.toml non valido, "
        "messo da parte.\n"
        f"Originale spostato in: {quarantined}\n"
        f"Errore: {error}\n"
        "huebeat e' ripartito con i soli valori di config.toml.\n"
    )
    try:
        marker.write_text(text, encoding="utf-8")
    except OSError:
        logger.warning("Marker d'avvio non scrivibile in %s.", marker, exc_info=True)


def _load_settings() -> Settings:
    """Carica le settings proteggendo il solo config.local.toml generato.

    Se la prima costruzione fallisce e config.local.toml esiste, quel file (che
    e' GENERATO dal frontend, non scritto a mano) viene messo da parte con
    suffisso .corrupt e la costruzione ritentata senza di esso: un TOML troncato
    dal frontend non deve impedire l'avvio. Se il problema NON e' il file
    generato (config.toml / .env), il file generato viene ripristinato e
    l'errore originale rilanciato (fail-fast sulla config scritta a mano).

    Returns:
        L'istanza Settings caricata.
    """
    try:
        return Settings()
    except Exception as first_error:  # noqa: BLE001 - riclassificato sotto
        local_path = _HOME / paths.LOCAL_CONFIG_FILENAME
        if not local_path.exists():
            raise  # non e' colpa del file generato: fail-fast su config.toml/.env
        quarantined = _quarantine_path(local_path)
        local_path.rename(quarantined)
        try:
            loaded = Settings()
        except Exception:  # noqa: BLE001 - non era il file generato
            quarantined.rename(local_path)  # ripristina il file generato
            raise first_error from None
        _write_startup_marker(quarantined, first_error)
        logger.warning(
            "config.local.toml non valido messo da parte in %s; avvio con config.toml.",
            quarantined,
        )
        return loaded


settings = _load_settings()  # fail-fast solo sulla config scritta a mano


def reload_settings() -> None:
    """Ricarica `settings` da disco IN PLACE (stessa istanza condivisa).

    Serve dopo il pairing o il salvataggio del client id Spotify dalla UI: i
    moduli che hanno gia' fatto `from huebeat.config import settings` tengono
    un riferimento all'istanza, quindi rimpiazzarla non basterebbe; qui si
    aggiorna lo stato interno dell'istanza esistente con quello di una
    rilettura fresca dei file. Passa dal caricamento protetto, cosi' un
    config.local.toml corrotto a runtime non fa crashare chi ricarica.
    """
    fresh = _load_settings()
    settings.__dict__.update(fresh.__dict__)
