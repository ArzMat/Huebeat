"""Persistenza delle regolazioni del frontend in config.local.toml.

config.toml resta il file scritto A MANO (commenti inclusi) e non viene mai
toccato; le modifiche fatte dal frontend finiscono in config.local.toml, che
pydantic-settings gia' carica come override (vedi Settings.model_config).
Il file e' interamente GENERATO: riscriverlo da zero a ogni salvataggio e'
sicuro e mantiene un formato stabile e leggibile.

Le scritture passano da un lock di modulo (piu' richieste HTTP concorrenti)
e avvengono su file temporaneo + replace atomico, cosi' un crash a meta'
scrittura non lascia mai un TOML troncato.
"""

import logging
import threading
from pathlib import Path

from huebeat import paths

logger = logging.getLogger(__name__)

LOCAL_CONFIG_FILENAME = paths.LOCAL_CONFIG_FILENAME

_HEADER = (
    "# File GENERATO dal frontend di huebeat: non modificare a mano.\n"
    "# Fa override di config.toml (vedi huebeat/config.py). Eliminarlo\n"
    "# riporta tutto ai valori di config.toml.\n"
)

_write_lock = threading.Lock()


def _format_value(value: object) -> str:
    """Formatta un valore scalare Python come TOML.

    Args:
        value: str, bool, int o float.

    Returns:
        La rappresentazione TOML del valore.

    Raises:
        TypeError: Per tipi non supportati.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (int, float)):
        return repr(value)
    raise TypeError(f"Tipo non supportato in config.local.toml: {type(value)!r}")


def render(overrides: dict) -> str:
    """Genera il testo TOML degli override.

    Le chiavi con valore dict diventano sezioni (dopo le chiavi top-level,
    come richiede TOML); le altre righe `chiave = valore`.

    Args:
        overrides: Dict di override; i valori dict sono sezioni.

    Returns:
        Il contenuto TOML completo, header incluso.
    """
    top_lines: list[str] = []
    sections: list[str] = []
    for key, value in overrides.items():
        if isinstance(value, dict):
            rows = "\n".join(f"{k} = {_format_value(v)}" for k, v in value.items())
            sections.append(f"[{key}]\n{rows}\n")
        else:
            top_lines.append(f"{key} = {_format_value(value)}")
    parts = [_HEADER]
    if top_lines:
        parts.append("\n".join(top_lines) + "\n")
    parts.extend(sections)
    return "\n".join(parts)


def save(overrides: dict, directory: Path | None = None) -> Path:
    """Scrive gli override in config.local.toml (replace atomico).

    Args:
        overrides: Dict di override da persistere (vedi `render`).
        directory: Cartella di destinazione; None per la home di huebeat
            (accanto a config.toml, come si aspetta pydantic-settings).

    Returns:
        Il percorso del file scritto.
    """
    target = (directory or paths.data_dir()) / LOCAL_CONFIG_FILENAME
    text = render(overrides)
    with _write_lock:
        tmp = target.with_suffix(".toml.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)
    logger.info("Regolazioni salvate in %s.", target)
    return target
