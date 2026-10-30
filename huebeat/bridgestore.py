"""Storage delle credenziali di PIU' bridge Hue in un file JSON.

Fondamento del multi-bridge. Oggi il progetto conosce un solo bridge
via secrets HUE_USERNAME/HUE_CLIENTKEY nel .env; qui le credenziali vivono in
`<home>/.bridges.json`, con una voce per bridge indicizzata dal `bridgeid`.

Il file contiene SECRETS (application key e PSK DTLS di ogni bridge): non
finisce mai in git (vedi .gitignore) e i due campi sensibili non compaiono in
log, messaggi d'errore o repr.

Le scritture passano da un lock di modulo e avvengono su file temporaneo +
replace atomico (stesso pattern di huebeat.localconfig), cosi' un crash a meta'
scrittura non lascia mai un JSON troncato.

`migrate_legacy` porta una tantum il bridge del .env dentro lo store: e'
idempotente (se lo store non e' vuoto non fa nulla) e non solleva mai, cosi' il
chiamante puo' sempre ricadere sul percorso legacy se la migrazione fallisce.
"""

import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from huebeat import paths
from huebeat.bridge import (
    BRIDGE_TIMEOUT_SECONDS,
    BRIDGE_VERIFY_TLS,
    BridgeError,
    discover_bridge,
)
from huebeat.config import settings

logger = logging.getLogger(__name__)

BRIDGES_FILENAME = paths.BRIDGES_FILENAME

# Endpoint legacy NON autenticato del bridge con l'anagrafica (bridgeid, name,
# modelid, ...). Innocuo, non richiede la application key.
CONFIG_PATH = "/api/0/config"

# Chiavi della struttura del file .bridges.json.
_BRIDGES_KEY = "bridges"
# Campi persistiti per ogni bridge (ordine stabile nel JSON).
_FIELDS = ("id", "host", "username", "clientkey", "name")

# RLock (non Lock): add/remove tengono il lock sull'INTERO ciclo load-modifica-
# save e chiamano save_bridges, che lo riacquisisce. Con un Lock semplice
# sarebbe deadlock; l'RLock permette la riacquisizione dallo stesso thread e
# continua a serializzare thread diversi.
_write_lock = threading.RLock()


class BridgeStoreError(Exception):
    """Errore di lettura/scrittura dello store dei bridge."""


def _atomic_write_text(path: Path, text: str) -> None:
    """Scrive `text` in `path` in modo atomico (tmp unico nella stessa dir + replace).

    Crea un temporaneo con nome unico (tempfile.mkstemp) nella stessa directory,
    lo scrive e lo rinomina con os.replace: robusto anche cross-process (niente
    tmp a nome fisso condiviso) e senza mai lasciare un JSON troncato. Il
    temporaneo viene rimosso se qualcosa fallisce prima del replace.

    Args:
        path: Percorso di destinazione.
        text: Contenuto da scrivere (UTF-8).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass
class BridgeCredentials:
    """Credenziali di un singolo bridge Hue.

    Attributes:
        id: bridgeid del bridge in minuscolo (chiave stabile nello store).
        host: IP LAN del bridge.
        username: application key CLIP v2 (SECRET).
        clientkey: PSK DTLS per lo streaming Entertainment (SECRET).
        name: nome leggibile del bridge (facoltativo).
    """

    id: str
    host: str
    username: str
    clientkey: str
    name: str = ""

    def __repr__(self) -> str:
        """Repr che maschera i due campi sensibili (username e clientkey)."""
        return (
            f"BridgeCredentials(id={self.id!r}, host={self.host!r}, "
            f"username='***', clientkey='***', name={self.name!r})"
        )


def _store_path(directory: Path | None) -> Path:
    """Percorso del file .bridges.json nella cartella indicata o nella home.

    Args:
        directory: Cartella di destinazione; None per la home di huebeat.

    Returns:
        Il percorso del file dello store.
    """
    return (directory or paths.data_dir()) / BRIDGES_FILENAME


def load_bridges(directory: Path | None = None) -> list[BridgeCredentials]:
    """Carica le credenziali dei bridge dallo store.

    Args:
        directory: Cartella dello store; None per la home di huebeat.

    Returns:
        La lista dei bridge; vuota se il file non esiste ancora.

    Raises:
        BridgeStoreError: Se il JSON e' corrotto o la struttura e' inattesa.
            Il messaggio non contiene MAI il contenuto del file (secrets).
    """
    path = _store_path(directory)
    if not path.exists():
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise BridgeStoreError(f"Store dei bridge illeggibile ({path.name}): {err}") from err

    if not isinstance(raw, dict) or not isinstance(raw.get(_BRIDGES_KEY), list):
        raise BridgeStoreError(f"Struttura inattesa nello store dei bridge ({path.name}).")

    bridges: list[BridgeCredentials] = []
    for entry in raw[_BRIDGES_KEY]:
        if not isinstance(entry, dict):
            raise BridgeStoreError(f"Voce non valida nello store dei bridge ({path.name}).")
        try:
            bridges.append(
                BridgeCredentials(
                    id=str(entry["id"]),
                    host=str(entry["host"]),
                    username=str(entry["username"]),
                    clientkey=str(entry["clientkey"]),
                    name=str(entry.get("name", "")),
                )
            )
        except KeyError as err:
            raise BridgeStoreError(
                f"Campo mancante in una voce dello store dei bridge ({path.name}): {err}"
            ) from err
    return bridges


def save_bridges(bridges: list[BridgeCredentials], directory: Path | None = None) -> Path:
    """Scrive le credenziali dei bridge nello store (replace atomico).

    Args:
        bridges: La lista completa dei bridge da persistere.
        directory: Cartella dello store; None per la home di huebeat.

    Returns:
        Il percorso del file scritto.
    """
    target = _store_path(directory)
    # Ordine dei campi stabile per ogni voce (leggibilita' del file).
    payload = {_BRIDGES_KEY: [{field: getattr(b, field) for field in _FIELDS} for b in bridges]}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with _write_lock:
        _atomic_write_text(target, text)
    # Solo il conteggio nel log: mai le credenziali.
    logger.info("Salvati %d bridge in %s.", len(bridges), target)
    return target


def add_bridge(creds: BridgeCredentials, directory: Path | None = None) -> None:
    """Aggiunge o sostituisce un bridge nello store (per `id`).

    Carica lo store, rimpiazza l'eventuale voce con lo stesso `id`, altrimenti
    appende in coda, e salva.

    Args:
        creds: Le credenziali del bridge da inserire.
        directory: Cartella dello store; None per la home di huebeat.
    """
    # Lock sull'INTERO ciclo load-modifica-save: due richieste concorrenti non
    # devono partire dallo stesso stato e perdere l'una la scrittura dell'altra.
    with _write_lock:
        bridges = [b for b in load_bridges(directory) if b.id != creds.id]
        bridges.append(creds)
        save_bridges(bridges, directory)


def remove_bridge(bridge_id: str, directory: Path | None = None) -> bool:
    """Rimuove un bridge dallo store per `id`.

    Args:
        bridge_id: bridgeid del bridge da rimuovere.
        directory: Cartella dello store; None per la home di huebeat.

    Returns:
        True se il bridge esisteva ed e' stato rimosso, False altrimenti.
    """
    # Lock sull'INTERO ciclo load-modifica-save (vedi add_bridge).
    with _write_lock:
        bridges = load_bridges(directory)
        kept = [b for b in bridges if b.id != bridge_id]
        if len(kept) == len(bridges):
            return False
        save_bridges(kept, directory)
        return True


def update_bridge_host(bridge_id: str, host: str, directory: Path | None = None) -> bool:
    """Aggiorna l'host di un bridge nello store (per `id`).

    Usato dalla ri-discovery al cambio IP (DHCP): un bridge che riprende un
    indirizzo diverso viene ritrovato per bridgeid e il nuovo host va persistito
    perche' regga al prossimo avvio. Ciclo load-modifica-save sotto il lock (come
    add_bridge/remove_bridge), cosi' due scritture concorrenti si serializzano.

    Args:
        bridge_id: bridgeid del bridge da aggiornare.
        host: Nuovo host (IP LAN) da persistere.
        directory: Cartella dello store; None per la home di huebeat.

    Returns:
        True se il bridge esisteva (host aggiornato, o gia' uguale e quindi
        nessuna scrittura), False se l'id non e' nello store.
    """
    with _write_lock:
        bridges = load_bridges(directory)
        for creds in bridges:
            if creds.id == bridge_id:
                if creds.host == host:
                    return True
                creds.host = host
                save_bridges(bridges, directory)
                return True
        return False


def fetch_bridge_info(host: str) -> dict:
    """Legge l'anagrafica del bridge dall'endpoint legacy non autenticato.

    GET ``https://<host>/api/0/config`` (self-signed, quindi `verify=False`):
    l'endpoint non richiede la application key e restituisce almeno
    ``bridgeid`` e ``name``.

    Args:
        host: IP o hostname del bridge.

    Returns:
        Il dict JSON dell'anagrafica del bridge.

    Raises:
        BridgeStoreError: Su errore di rete, JSON non valido o campo
            ``bridgeid`` mancante.
    """
    url = f"https://{host}{CONFIG_PATH}"
    try:
        response = httpx.get(url, verify=BRIDGE_VERIFY_TLS, timeout=BRIDGE_TIMEOUT_SECONDS)
        response.raise_for_status()
        info = response.json()
    except (httpx.HTTPError, ValueError) as err:
        raise BridgeStoreError(f"Anagrafica del bridge {host} non leggibile: {err}") from err

    if not isinstance(info, dict) or not info.get("bridgeid"):
        raise BridgeStoreError(f"Risposta del bridge {host} priva del campo 'bridgeid'.")
    return info


def any_bridge_configured(directory: Path | None = None) -> bool:
    """Indica se esiste almeno un bridge utilizzabile (store o .env legacy).

    Usato come gate d'avvio: non solleva mai. Se lo store e' corrotto lo
    ignora (log warning) e considera solo il .env legacy.

    Args:
        directory: Cartella dello store; None per la home di huebeat.

    Returns:
        True se lo store ha almeno un bridge oppure il .env legacy e' completo.
    """
    try:
        if load_bridges(directory):
            return True
    except BridgeStoreError as err:
        logger.warning("Store dei bridge non valido, uso solo il .env legacy: %s", err)
    return settings.bridge_paired()


def migrate_legacy(
    fetch_info: Callable[[str], dict] | None = None,
    directory: Path | None = None,
) -> BridgeCredentials | None:
    """Migra una tantum il bridge del .env legacy dentro lo store.

    Idempotente e best-effort: se lo store non e' vuoto o il .env non e'
    completo non fa nulla; su qualsiasi errore (rete, JSON, store) logga un
    warning e ritorna None, cosi' il chiamante puo' ricadere sul percorso
    legacy e la migrazione verra' ritentata al prossimo avvio.

    Args:
        fetch_info: Funzione host -> anagrafica del bridge; None usa
            `fetch_bridge_info`.
        directory: Cartella dello store; None per la home di huebeat.

    Returns:
        Le credenziali migrate, oppure None se non c'e' nulla da migrare o la
        migrazione fallisce.
    """
    try:
        if load_bridges(directory):
            return None
    except BridgeStoreError as err:
        logger.warning("Store dei bridge non valido, migrazione legacy saltata: %s", err)
        return None

    if not settings.bridge_paired():
        return None

    fetch = fetch_info or fetch_bridge_info
    try:
        host = settings.bridge_host or discover_bridge()
        info = fetch(host)
        creds = BridgeCredentials(
            id=str(info["bridgeid"]).lower(),
            host=host,
            username=settings.hue_username.get_secret_value(),
            clientkey=settings.hue_clientkey.get_secret_value(),
            name=info.get("name", ""),
        )
        add_bridge(creds, directory)
    except (BridgeError, BridgeStoreError, httpx.HTTPError, KeyError) as err:
        logger.warning("Migrazione del bridge legacy fallita, riprovo al prossimo avvio: %s", err)
        return None

    logger.info(
        "Migrato il bridge legacy %s (%s) nello store.", creds.id, creds.name or "senza nome"
    )
    return creds
