"""Aggiornamento in-place del file .env preservando commenti e righe esistenti.

Un solo punto del codebase che riscrive il file .env chiave per chiave: il
pairing del Bridge e il collegamento Spotify (dal terminale o dal frontend)
passano tutti da `update_env_values`. La logica preserva commenti, righe vuote
e ogni chiave non toccata; aggiorna quelle presenti e appende quelle nuove in
fondo. Se il file non esiste, lo crea con le sole chiavi date.
"""

from pathlib import Path


def update_env_values(path: Path, values: dict[str, str]) -> None:
    """Aggiorna (o appende) le chiavi date nel file .env preservando il resto.

    Riscrive solo le righe delle chiavi in `values`, lasciando intatti commenti,
    righe vuote e ogni altra chiave. Le chiavi non ancora presenti vengono
    aggiunte in fondo. Se il file non esiste, lo crea con le sole righe date.

    Args:
        path: Percorso del file .env.
        values: Mappa chiave -> valore da scrivere/aggiornare.
    """
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    seen: set[str] = set()
    out_lines: list[str] = []
    for line in lines:
        key = _env_line_key(line)
        if key in values:
            out_lines.append(f"{key}={values[key]}")
            seen.add(key)
        else:
            out_lines.append(line)

    # Aggiungi in fondo le chiavi non ancora presenti nel file.
    for key, value in values.items():
        if key not in seen:
            out_lines.append(f"{key}={value}")

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _env_line_key(line: str) -> str | None:
    """Estrae il nome di chiave da una riga .env, se e' un assegnamento.

    Args:
        line: Riga grezza del file .env.

    Returns:
        Il nome della chiave (senza spazi) se la riga e' del tipo `KEY=...` e
        non e' un commento; altrimenti None.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    return stripped.split("=", 1)[0].strip()
