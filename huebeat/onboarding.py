"""Onboarding dal frontend: pairing del Bridge e collegamento Spotify.

Due manager con stato osservabile, pensati per essere pilotati dal server di
controllo (huebeat.control) senza terminale:

- `PairingManager`: scoperta dei Bridge e pairing col pulsante fisico, con
  countdown leggibile dalla UI; su successo salva le credenziali del bridge
  nello store multi-bridge (.bridges.json) e ricarica le settings.
- `SpotifyLinkManager`: salvataggio del client id e login OAuth interattivo,
  con stato leggibile dalla UI.

Regole rispettate:

- I secrets del Bridge (username/clientkey) finiscono SOLO nello store
  multi-bridge (.bridges.json della home); il client id Spotify resta nel .env
  della home. Nessun secret viene mai loggato ne' restituito: lo stato esposto
  e' solo booleani/messaggi.
- Ogni operazione bloccante (discovery, pairing, login) gira in un thread
  daemon proprio: il loop di rendering non viene mai toccato.
- `reload_settings` e' iniettabile (parametro del costruttore) cosi' i test
  possono evitare di rileggere il .env reale del processo.

Convenzione output, coerente col resto del progetto: `logging` per la
diagnostica, nessun `print` (l'onboarding parla con la UI, non col terminale).
"""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from huebeat import bridgestore, config, paths
from huebeat.bridge import BridgeError, discover_bridges, pair
from huebeat.bridgestore import BridgeCredentials, BridgeStoreError, load_bridges
from huebeat.config import settings
from huebeat.envfile import update_env_values

logger = logging.getLogger(__name__)

# --- Chiave .env del client id Spotify (i secrets del Bridge vanno nello store) ---
ENV_SPOTIFY_CLIENT_ID_KEY = "SPOTIFY_CLIENT_ID"

# --- Pairing ---
# Finestra di attesa della pressione del pulsante fisico del Bridge.
PAIR_TIMEOUT_SECONDS = 60.0


def _env_path() -> Path:
    """Percorso del file .env nella home di huebeat.

    Returns:
        Il path del .env (`paths.data_dir() / paths.ENV_FILENAME`), risolto a
        ogni chiamata cosi' un HUEBEAT_HOME impostato dopo l'import viene onorato.
    """
    return paths.data_dir() / paths.ENV_FILENAME


class PairingManager:
    """Gestisce discovery e pairing del Bridge con stato osservabile.

    Un solo tentativo alla volta: se un thread di discovery o pairing e' gia'
    attivo, nuove chiamate a `discover`/`start` vengono ignorate (con log).
    Lo stato e' un dict RIASSEGNATO in blocco (mai mutato in place), cosi' il
    server puo' leggerlo senza lock.

    Attributes:
        on_paired: Callback opzionale invocato nel thread di pairing dopo un
            successo (es. avviare il motore). callable() -> None o None.
    """

    def __init__(
        self,
        reload_settings: Callable[[], None] | None = None,
        store_dir: Path | None = None,
    ) -> None:
        """Inizializza il manager (nessun thread attivo).

        Args:
            reload_settings: Funzione da chiamare per ricaricare le settings dopo
                il pairing; None usa `config.reload_settings` (nei test si
                inietta un no-op per non rileggere il .env reale del processo).
            store_dir: Cartella dello store dei bridge (.bridges.json); None usa
                la home reale di huebeat (nei test si passa una tmp_path).
        """
        self._reload_settings = reload_settings or config.reload_settings
        self._store_dir = store_dir
        self._lock = threading.Lock()
        # True mentre un thread di discovery o pairing e' in corso: un solo
        # tentativo alla volta (guardia sotto lock, niente race sul thread vivo).
        self._busy = False
        self._state: dict = {"state": "idle", "message": "", "seconds_left": 0}
        self.on_paired: Callable[[], None] | None = None

    @property
    def state(self) -> dict:
        """Stato corrente osservabile dalla UI.

        Returns:
            Dict con almeno `state` (idle|discovering|waiting|success|error),
            `message` e `seconds_left`; dopo la discovery contiene anche `host`
            (l'host proposto) e `found` (lista dei bridge trovati con `id`,
            `host` e `paired`).
        """
        return self._state

    def discover(self) -> None:
        """Avvia in un thread la scoperta dei Bridge sulla LAN.

        Ignora la richiesta se un tentativo e' gia' in corso. I bridge trovati
        finiscono nello stato (`found` + `host` proposto + `message`); un
        fallimento porta lo stato a `error`.
        """
        if not self._begin("discover"):
            return
        threading.Thread(target=self._discover_worker, name="pair-discover", daemon=True).start()

    def start(self, host: str) -> None:
        """Avvia in un thread il pairing verso `host` col pulsante fisico.

        Ignora la richiesta se un tentativo e' gia' in corso. Se `host` e' vuoto
        usa quello scoperto in precedenza; se manca anche quello, va in `error`.
        Su successo scrive i secrets nel .env, ricarica le settings, porta lo
        stato a `success` e invoca `on_paired` se impostato.

        Args:
            host: IP o hostname del Bridge; "" per usare quello gia' scoperto.
        """
        target = host.strip() or str(self._state.get("host", ""))
        if not target:
            self._state = {
                "state": "error",
                "message": "Nessun host: cerca il bridge o inserisci l'IP.",
                "seconds_left": 0,
            }
            return
        if not self._begin("start"):
            return
        threading.Thread(
            target=self._pair_worker, args=(target,), name="pair-start", daemon=True
        ).start()

    def _begin(self, action: str) -> bool:
        """Prova a diventare il tentativo attivo (un solo thread alla volta).

        Args:
            action: Nome dell'azione richiesta (solo per il log del rifiuto).

        Returns:
            True se il chiamante puo' procedere; False se un tentativo e' gia'
            in corso (richiesta ignorata).
        """
        with self._lock:
            if self._busy:
                logger.info("Onboarding Bridge occupato: '%s' ignorato.", action)
                return False
            self._busy = True
        return True

    def _end(self) -> None:
        """Libera il tentativo attivo (chiamato dai worker nel loro finally)."""
        with self._lock:
            self._busy = False

    def _discover_worker(self) -> None:
        """Corpo del thread di discovery: cerca i Bridge e aggiorna lo stato."""
        self._state = {"state": "discovering", "message": "Cerco il bridge...", "seconds_left": 0}
        try:
            bridges = discover_bridges()
        except BridgeError as err:
            logger.warning("Discovery del bridge fallita: %s", err)
            self._state = {"state": "error", "message": str(err), "seconds_left": 0}
            return
        finally:
            self._end()

        # Id gia' presenti nello store: marcano i bridge come gia' accoppiati.
        try:
            paired_ids = {creds.id for creds in load_bridges(self._store_dir)}
        except BridgeStoreError:
            paired_ids = set()
        found = [
            {"id": b["id"], "host": b["host"], "paired": b["id"] in paired_ids and bool(b["id"])}
            for b in bridges
        ]
        # Host proposto: il primo bridge non ancora accoppiato, altrimenti il primo.
        proposed = next((b["host"] for b in found if not b["paired"]), found[0]["host"])
        message = (
            f"Bridge trovato: {proposed}" if len(found) == 1 else f"Trovati {len(found)} bridge"
        )
        self._state = {
            "state": "idle",
            "message": message,
            "seconds_left": 0,
            "host": proposed,
            "found": found,
        }

    def _pair_worker(self, host: str) -> None:
        """Corpo del thread di pairing: attende il pulsante, salva i secrets.

        Args:
            host: Host del Bridge verso cui fare il pairing.
        """
        deadline = time.monotonic() + PAIR_TIMEOUT_SECONDS
        self._state = {
            "state": "waiting",
            "message": "Premi il pulsante sul Bridge.",
            "seconds_left": int(PAIR_TIMEOUT_SECONDS),
            "host": host,
        }

        # Countdown parallelo: pair() blocca col suo polling interno, un thread
        # a parte tiene aggiornato seconds_left finche' il pairing e' in corso.
        stop_countdown = threading.Event()
        countdown = threading.Thread(
            target=self._countdown_worker,
            args=(deadline, host, stop_countdown),
            name="pair-countdown",
            daemon=True,
        )
        countdown.start()

        try:
            username, clientkey = pair(host, timeout_s=PAIR_TIMEOUT_SECONDS)
        except BridgeError as err:
            logger.warning("Pairing col bridge fallito: %s", err)
            self._state = {"state": "error", "message": str(err), "seconds_left": 0}
            return
        finally:
            stop_countdown.set()
            self._end()

        # Anagrafica del bridge (bridgeid, name) dall'endpoint legacy: serve
        # l'id come chiave stabile nello store. Se non risponde non salviamo
        # nulla (senza id lo store non e' indicizzabile) e lo diciamo.
        try:
            info = bridgestore.fetch_bridge_info(host)
        except BridgeStoreError as err:
            logger.warning(
                "Pairing riuscito ma anagrafica del bridge %s non leggibile: %s", host, err
            )
            self._state = {
                "state": "error",
                "message": "Pairing riuscito ma bridge non interrogabile: riprova.",
                "seconds_left": 0,
                "host": host,
            }
            return

        bridgestore.add_bridge(
            BridgeCredentials(
                id=str(info["bridgeid"]).lower(),
                host=host,
                username=username,
                clientkey=clientkey,
                name=info.get("name", ""),
            ),
            self._store_dir,
        )
        self._reload_settings()
        self._state = {
            "state": "success",
            "message": f"Pairing riuscito con {host}.",
            "seconds_left": 0,
            "host": host,
        }
        logger.info("Pairing col bridge riuscito, credenziali salvate nello store.")
        if self.on_paired is not None:
            try:
                self.on_paired()
            except Exception:  # noqa: BLE001 - il callback e' dell'host, best-effort
                logger.exception("Callback on_paired fallito.")

    def _countdown_worker(self, deadline: float, host: str, stop: threading.Event) -> None:
        """Aggiorna `seconds_left` una volta al secondo mentre si attende.

        Args:
            deadline: Istante monotonic di scadenza dell'attesa.
            host: Host in pairing (riportato nello stato per la UI).
            stop: Event che ferma il countdown quando il pairing finisce.
        """
        while not stop.is_set():
            remaining = int(round(deadline - time.monotonic()))
            if remaining <= 0:
                break
            # Aggiorna solo finche' siamo ancora in attesa: se il pairing e'
            # gia' passato a success/error non sovrascrivere quello stato.
            if self._state.get("state") != "waiting":
                break
            self._state = {
                "state": "waiting",
                "message": "Premi il pulsante sul Bridge.",
                "seconds_left": remaining,
                "host": host,
            }
            stop.wait(1.0)


class SpotifyLinkManager:
    """Gestisce client id e login Spotify dalla UI, con stato osservabile.

    Il login OAuth interattivo (che apre il browser e blocca fino a 180 s) gira
    in un thread daemon. Lo stato e' un dict riassegnato in blocco.

    Attributes:
        on_linked: Callback opzionale invocato nel thread di login dopo un
            successo (es. power-cycle del motore per far partire il poller).
            callable() -> None o None.
    """

    def __init__(self, reload_settings: Callable[[], None] | None = None) -> None:
        """Inizializza il manager (nessun login attivo).

        Args:
            reload_settings: Funzione per ricaricare le settings dopo aver
                scritto il .env; None usa `config.reload_settings`.
        """
        self._reload_settings = reload_settings or config.reload_settings
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._login_state = "idle"
        self._login_message = ""
        self.on_linked: Callable[[], None] | None = None

    def status(self) -> dict:
        """Stato di configurazione e collegamento Spotify per la UI.

        Returns:
            Dict con `configured` (client id presente), `authenticated` (token
            file presente), `state` (idle|waiting|success|error) e `message`.
        """
        return {
            "configured": settings.spotify_client_id is not None,
            "authenticated": self._has_tokens(),
            "state": self._login_state,
            "message": self._login_message,
        }

    def set_client_id(self, client_id: str) -> None:
        """Salva il client id Spotify nel .env e ricarica le settings.

        Args:
            client_id: Client ID dell'app Spotify (client pubblico PKCE).
        """
        update_env_values(_env_path(), {ENV_SPOTIFY_CLIENT_ID_KEY: client_id.strip()})
        self._reload_settings()
        logger.info("Client id Spotify salvato nel .env.")

    def start_login(self) -> None:
        """Avvia in un thread il login OAuth interattivo (apre il browser).

        Ignora la richiesta se un login e' gia' in corso. Se manca il client id
        va subito in `error`. Su successo invoca `on_linked` se impostato.
        """
        if settings.spotify_client_id is None:
            self._set_login("error", "Client id Spotify mancante: salvalo prima.")
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.info("Login Spotify gia' in corso: richiesta ignorata.")
                return
            self._thread = threading.Thread(
                target=self._login_worker, name="spotify-login", daemon=True
            )
            self._set_login("waiting", "Completa il login nel browser...")
            self._thread.start()

    def _login_worker(self) -> None:
        """Corpo del thread di login: esegue interactive_login e riporta l'esito."""
        from huebeat.spotify import SpotifyAuth, SpotifyError

        client_id = settings.spotify_client_id.get_secret_value()  # type: ignore[union-attr]
        token_path = paths.data_dir() / paths.SPOTIFY_TOKENS_FILENAME
        try:
            SpotifyAuth(client_id, token_path).interactive_login()
        except SpotifyError as err:
            logger.warning("Login Spotify fallito: %s", err)
            self._set_login("error", str(err))
            return
        self._set_login("success", "Spotify collegato.")
        logger.info("Login Spotify riuscito, token salvati.")
        if self.on_linked is not None:
            try:
                self.on_linked()
            except Exception:  # noqa: BLE001 - callback dell'host, best-effort
                logger.exception("Callback on_linked fallito.")

    def _set_login(self, state: str, message: str) -> None:
        """Aggiorna lo stato del login (assegnazioni atomiche, niente lock).

        Args:
            state: Nuovo stato del login (idle|waiting|success|error).
            message: Messaggio per la UI.
        """
        self._login_state = state
        self._login_message = message

    def _has_tokens(self) -> bool:
        """Indica se esiste gia' un file di token Spotify nella home.

        Returns:
            True se il client id e' configurato e il file token esiste.
        """
        if settings.spotify_client_id is None:
            return False
        from huebeat.spotify import SpotifyAuth

        token_path = paths.data_dir() / paths.SPOTIFY_TOKENS_FILENAME
        return SpotifyAuth(settings.spotify_client_id.get_secret_value(), token_path).has_tokens()
