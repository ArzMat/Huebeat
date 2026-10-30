"""Client REST del Bridge Hue (discovery, pairing, API CLIP v2).

Copre tutto il dialogo HTTP col Bridge Hue che precede lo streaming DTLS:
scoperta dell'IP sulla LAN, pairing col pulsante fisico, e le chiamate CLIP v2
per elencare/avviare/fermare le entertainment configuration. Lo streaming vero e
proprio dei colori (UDP/DTLS) vive altrove: qui si arriva fino a "start".

Tutte le chiamate verso il Bridge usano HTTPS con certificato self-signed
(`verify=False`) e timeout corto, come da hue-entertainment-pykit e dal blog
iotech sull'Entertainment API.
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

# --- Discovery cloud ---
# Endpoint N-UPnP ufficiale: restituisce i bridge visti dall'IP pubblico del
# chiamante. Campo con l'IP LAN del bridge nel JSON di risposta.
DISCOVERY_URL = "https://discovery.meethue.com/"
DISCOVERY_IP_FIELD = "internalipaddress"
DISCOVERY_TIMEOUT_SECONDS = 10.0

# --- Pairing ---
# Tipo device inviato al bridge e intervallo di ritentativo mentre si attende la
# pressione del pulsante fisico.
PAIR_DEVICETYPE = "huebeat#pc"
PAIR_RETRY_SECONDS = 2.0

# --- Chiamate al bridge (CLIP v2 e legacy /api) ---
# Il bridge espone un certificato self-signed: la verifica va disabilitata.
BRIDGE_TIMEOUT_SECONDS = 5.0
BRIDGE_VERIFY_TLS = False

# Header di autenticazione CLIP v2: la "username" del pairing va qui.
AUTH_HEADER = "hue-application-key"

# Endpoint CLIP v2.
APPLICATION_ID_PATH = "/auth/v1"
APPLICATION_ID_HEADER = "hue-application-id"
ENTERTAINMENT_CONFIG_PATH = "/clip/v2/resource/entertainment_configuration"

# Azioni per avviare/fermare lo streaming di una entertainment configuration.
ACTION_START = "start"
ACTION_STOP = "stop"

# Status HTTP che consideriamo "successo" per le risposte del bridge.
HTTP_OK = 200


class BridgeError(Exception):
    """Errore di comunicazione col Bridge."""


def _bridge_base_url(host: str) -> str:
    """Costruisce l'URL base HTTPS per un host del bridge.

    Args:
        host: IP o hostname del Bridge Hue.

    Returns:
        URL base nella forma ``https://{host}``.
    """
    return f"https://{host}"


def discover_bridges() -> list[dict]:
    """Scopre TUTTI i Bridge Hue sulla LAN via discovery cloud.

    Interroga l'endpoint N-UPnP di Philips, che risponde con l'elenco dei bridge
    visibili dall'IP pubblico del chiamante, e ne restituisce uno per voce con
    id e host. Le voci prive di ``internalipaddress`` vengono saltate.

    Returns:
        Lista di dict ``{"id": <campo "id" minuscolo, "" se assente>,
        "host": <internalipaddress>}``, una per bridge con IP valido.

    Raises:
        BridgeError: Se la richiesta fallisce, la risposta non e' valida o
            l'elenco dei bridge (con IP valido) e' vuoto.
    """
    logger.info("Discovery bridge via %s", DISCOVERY_URL)
    try:
        response = httpx.get(DISCOVERY_URL, timeout=DISCOVERY_TIMEOUT_SECONDS)
        response.raise_for_status()
        bridges = response.json()
    except (httpx.HTTPError, ValueError) as err:
        raise BridgeError(f"Discovery del bridge fallita: {err}") from err

    if not isinstance(bridges, list) or not bridges:
        raise BridgeError("Nessun bridge trovato dalla discovery cloud.")

    found: list[dict] = []
    for entry in bridges:
        if not isinstance(entry, dict):
            continue
        host = entry.get(DISCOVERY_IP_FIELD)
        if not host:
            continue
        found.append({"id": str(entry.get("id", "")).lower(), "host": str(host)})

    if not found:
        raise BridgeError(
            f"Risposta discovery priva del campo '{DISCOVERY_IP_FIELD}': {bridges[0]}"
        )
    return found


def discover_bridge() -> str:
    """Scopre l'IP di un Bridge Hue sulla LAN via discovery cloud.

    Interroga l'endpoint N-UPnP di Philips, che risponde con l'elenco dei bridge
    visibili dall'IP pubblico del chiamante, e restituisce l'IP interno del
    primo bridge trovato.

    Returns:
        L'indirizzo IP LAN del primo bridge (campo ``internalipaddress``).

    Raises:
        BridgeError: Se la richiesta fallisce, la risposta non e' valida o
            l'elenco dei bridge e' vuoto.
    """
    ip_address = discover_bridges()[0]["host"]
    logger.info("Bridge trovato all'IP %s", ip_address)
    return ip_address


def pair(host: str, timeout_s: float = 60.0) -> tuple[str, str]:
    """Esegue il pairing col Bridge attendendo la pressione del pulsante.

    Ritenta la POST di registrazione ogni ``PAIR_RETRY_SECONDS`` finche' il
    bridge risponde con successo o finche' scade ``timeout_s``. L'utente deve
    premere il pulsante fisico del bridge entro la finestra di timeout.

    Args:
        host: IP o hostname del Bridge Hue.
        timeout_s: Secondi massimi di attesa della pressione del pulsante.

    Returns:
        Tupla ``(username, clientkey)``: la ``username`` e' la application key
        usata come header CLIP v2, la ``clientkey`` e' la PSK per il DTLS.

    Raises:
        BridgeError: Se il timeout scade senza pairing riuscito.
    """
    url = f"{_bridge_base_url(host)}/api"
    body = {"devicetype": PAIR_DEVICETYPE, "generateclientkey": True}
    deadline = time.monotonic() + timeout_s

    logger.info("Pairing col bridge %s: premi il pulsante entro %.0f s", host, timeout_s)
    with httpx.Client(verify=BRIDGE_VERIFY_TLS, timeout=BRIDGE_TIMEOUT_SECONDS) as client:
        while time.monotonic() < deadline:
            credentials = _try_pair_once(client, url, body)
            if credentials is not None:
                logger.info("Pairing col bridge riuscito.")
                return credentials
            time.sleep(PAIR_RETRY_SECONDS)

    raise BridgeError(
        f"Pairing col bridge non completato entro {timeout_s:.0f} s (pulsante non premuto?)."
    )


def _try_pair_once(client: httpx.Client, url: str, body: dict) -> tuple[str, str] | None:
    """Tenta una singola POST di pairing.

    Args:
        client: Client HTTP gia' configurato per il bridge.
        url: URL dell'endpoint legacy ``/api``.
        body: Corpo della richiesta di registrazione.

    Returns:
        La tupla ``(username, clientkey)`` se il bridge ha risposto con
        successo, altrimenti ``None`` (tipicamente pulsante non ancora premuto).
    """
    try:
        response = client.post(url, json=body)
        response.raise_for_status()
        data = response.json()[0]
    except (httpx.HTTPError, ValueError, IndexError, KeyError) as err:
        logger.debug("Tentativo di pairing fallito, ritento: %s", err)
        return None

    if "success" in data:
        success = data["success"]
        return success["username"], success["clientkey"]

    # Presenza di "error" attesa finche' il pulsante non e' premuto.
    logger.debug("Bridge non pronto al pairing: %s", data.get("error"))
    return None


class BridgeClient:
    """Client CLIP v2 verso un Bridge Hue autenticato.

    Riusa un unico ``httpx.Client`` (configurato per il certificato self-signed
    del bridge) per tutte le chiamate. Va chiuso con `close` a fine uso, oppure
    usato come context manager.

    Attributes:
        host: IP o hostname del Bridge Hue.
    """

    def __init__(self, host: str, username: str) -> None:
        """Inizializza il client e apre la sessione HTTP verso il bridge.

        Args:
            host: IP o hostname del Bridge Hue.
            username: Application key ottenuta col pairing, usata nell'header
                ``hue-application-key``.
        """
        self.host = host
        self._base_url = _bridge_base_url(host)
        self._client = httpx.Client(
            verify=BRIDGE_VERIFY_TLS,
            timeout=BRIDGE_TIMEOUT_SECONDS,
            headers={
                AUTH_HEADER: username,
                "Content-Type": "application/json",
            },
        )

    def __enter__(self) -> "BridgeClient":
        """Restituisce l'istanza per l'uso come context manager."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Chiude la sessione HTTP all'uscita dal context manager."""
        self.close()

    def get_application_id(self) -> str:
        """Recupera lo Hue Application ID del bridge (identity PSK per il DTLS).

        Fa una GET su ``/auth/v1``: il valore sta nell'header di risposta
        ``hue-application-id`` (verificato sul codice di
        hue-entertainment-pykit, `_fetch_hue_application_id`).

        Returns:
            Lo Hue Application ID, usato come identity PSK nell'handshake DTLS.

        Raises:
            BridgeError: Se la richiesta fallisce o l'header e' assente.
        """
        response = self._get(APPLICATION_ID_PATH)
        application_id = response.headers.get(APPLICATION_ID_HEADER)
        if not application_id:
            raise BridgeError(
                f"Header '{APPLICATION_ID_HEADER}' assente nella risposta di {APPLICATION_ID_PATH}."
            )
        logger.info("Hue Application ID ottenuto dal bridge.")
        return application_id

    def list_entertainment_configurations(self) -> list[dict]:
        """Elenca le entertainment configuration definite sul bridge.

        Returns:
            La lista dei dict di configurazione (chiave ``data`` della risposta
            CLIP v2). Ogni item espone almeno ``id``, ``name``, ``status`` e
            ``channels``.

        Raises:
            BridgeError: Se la richiesta fallisce o la risposta non e' valida.
        """
        response = self._get(ENTERTAINMENT_CONFIG_PATH)
        try:
            configurations = response.json()["data"]
        except (ValueError, KeyError) as err:
            raise BridgeError(f"Risposta entertainment configuration non valida: {err}") from err
        # DEBUG e non INFO: il watchdog della sessione chiama questa lista ogni
        # 5 secondi e a INFO spammerebbe il terminale per tutta la durata di run.
        logger.debug("Trovate %d entertainment configuration.", len(configurations))
        return configurations

    def start_entertainment(self, config_id: str) -> None:
        """Avvia lo streaming di una entertainment configuration.

        Args:
            config_id: ID della entertainment configuration da avviare.

        Raises:
            BridgeError: Se la richiesta fallisce.
        """
        logger.info("Avvio entertainment configuration %s", config_id)
        self._put_action(config_id, ACTION_START)

    def stop_entertainment(self, config_id: str) -> None:
        """Ferma lo streaming di una entertainment configuration.

        Args:
            config_id: ID della entertainment configuration da fermare.

        Raises:
            BridgeError: Se la richiesta fallisce.
        """
        logger.info("Stop entertainment configuration %s", config_id)
        self._put_action(config_id, ACTION_STOP)

    def close(self) -> None:
        """Chiude la sessione HTTP sottostante."""
        self._client.close()

    def _get(self, path: str) -> httpx.Response:
        """Esegue una GET verso il bridge.

        Args:
            path: Path assoluto (con leading slash) sull'host del bridge.

        Returns:
            La risposta HTTP, gia' validata sullo status.

        Raises:
            BridgeError: Se la richiesta fallisce o lo status non e' 2xx.
        """
        return self._request("GET", path)

    def _put_action(self, config_id: str, action: str) -> None:
        """Invia un'azione start/stop su una entertainment configuration.

        Args:
            config_id: ID della entertainment configuration.
            action: Valore per il campo ``action`` del corpo (start/stop).

        Raises:
            BridgeError: Se la richiesta fallisce.
        """
        path = f"{ENTERTAINMENT_CONFIG_PATH}/{config_id}"
        self._request("PUT", path, json={"action": action})

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Esegue una richiesta al bridge con gestione errori uniforme.

        Args:
            method: Metodo HTTP.
            path: Path assoluto sull'host del bridge.
            **kwargs: Argomenti aggiuntivi passati a ``httpx.Client.request``.

        Returns:
            La risposta HTTP, con status verificato.

        Raises:
            BridgeError: Se la richiesta di rete fallisce o lo status non e' 2xx.
        """
        url = f"{self._base_url}{path}"
        try:
            response = self._client.request(method, url, **kwargs)
            response.raise_for_status()
        except httpx.HTTPError as err:
            raise BridgeError(f"Richiesta {method} {path} fallita: {err}") from err
        return response
