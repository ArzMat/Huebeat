"""Gestione della sessione Entertainment: setup, chiusura, watchdog e riconnessione.

Qui vive tutto il ciclo di vita della sessione verso il Bridge:

- `setup_session` / `close_session`: apertura e chiusura ordinata delle risorse
  (client REST, action start/stop, stream DTLS).
- `SessionKeeper`: watchdog che tiene VIVA la sessione durante `run`. Il Bridge
  puo' chiudere la sessione senza che il client se ne accorga (lo stream e' UDP:
  i send "riescono" anche nel vuoto); il keeper controlla periodicamente lo
  status dell'area via REST e, se risulta inattiva mentre stiamo streammando
  (o se un send fallisce), riconnette con retry finche' non riesce.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from huebeat.bridge import BridgeClient, discover_bridge
from huebeat.config import settings
from huebeat.streaming import EntertainmentStream, FrameValidationError

logger = logging.getLogger(__name__)

# Intervallo tra due controlli REST dello status dell'area (secondi).
WATCHDOG_INTERVAL_SECONDS = 5.0
# Grazia dopo una (ri)connessione prima di fidarsi dei controlli di status:
# lo stato sul Bridge impiega qualche istante ad allinearsi.
WATCHDOG_GRACE_SECONDS = 5.0
# Controlli di status "non active" consecutivi richiesti prima di dichiarare la
# sessione morta: uno solo puo' essere un falso positivo (glitch REST del Bridge).
WATCHDOG_NON_ACTIVE_POLLS = 2
# Cap all'attesa del join del watchdog allo stop: puo' essere dentro una GET
# REST con timeout ~5 s, oltre non ha senso trattenere il chiamante.
WATCHDOG_JOIN_TIMEOUT_SECONDS = 6.0
# Backoff esponenziale tra tentativi di riconnessione: primo ritardo, cap e
# fattore di crescita. Azzerato dopo una connessione rimasta stabile.
RECONNECT_BACKOFF_START_SECONDS = 2.0
RECONNECT_BACKOFF_MAX_SECONDS = 30.0
RECONNECT_BACKOFF_FACTOR = 2.0
# Durata minima di una connessione perche' sia considerata "stabile": oltre
# questa soglia il backoff riparte dal minimo alla prossima riconnessione.
STABLE_CONNECTION_SECONDS = 30.0
# Dopo quanti fallimenti di riconnessione consecutivi tentare una ri-discovery
# dell'host (il Bridge potrebbe aver preso un IP diverso via DHCP).
REDISCOVERY_AFTER_FAILURES = 5
# Tentativi di handshake DTLS per ciclo di RICONNESSIONE (il setup ne usa molti
# di piu'): a ~5 s l'uno un ciclo pieno resterebbe muto ~50 s e la UI sembra
# bloccata. Cicli corti + backoff esistente = piu' tentativi visibili, stesso
# tempo totale di attesa.
RECONNECT_HANDSHAKE_ATTEMPTS = 3


@dataclass
class Session:
    """Risorse attive di una sessione Entertainment, da chiudere in ordine.

    Attributes:
        client: Client REST del Bridge (start/stop entertainment).
        stream: Stream UDP/DTLS verso il Bridge per l'invio dei colori.
        config_id: Id della entertainment configuration attiva.
        area_name: Nome leggibile dell'area (per messaggi utente).
        channels: Dict grezzi dei canali dell'area (campi ``channel_id`` e
            ``position`` con x/y/z), come restituiti dal Bridge.
        host: Host del Bridge (per ricostruire lo stream in riconnessione).
        application_id: Hue application id (PSK identity dello stream).
        clientkey: PSK DTLS del bridge (per ricostruire lo stream in
            riconnessione; con piu' bridge ognuno ha la sua).
    """

    client: BridgeClient
    stream: EntertainmentStream
    config_id: str
    area_name: str
    channels: list[dict]
    host: str
    application_id: str
    clientkey: str


def resolve_host() -> str:
    """Restituisce l'host del Bridge da config o via auto-discovery.

    Returns:
        L'IP/host del Bridge Hue.
    """
    if settings.bridge_host:
        logger.debug("Uso bridge_host da config: %s", settings.bridge_host)
        return settings.bridge_host

    logger.info("bridge_host vuoto: avvio auto-discovery del Bridge.")
    return discover_bridge()


def config_name(config: dict) -> str:
    """Nome leggibile di una entertainment configuration.

    In CLIP v2 il nome puo' stare top-level oppure in metadata.name: gestisce
    entrambi i casi.

    Args:
        config: Dict di configurazione come restituito dal BridgeClient.

    Returns:
        Il nome trovato, o stringa vuota.
    """
    return config.get("name") or config.get("metadata", {}).get("name", "")


def select_configuration(configs: list[dict]) -> dict:
    """Sceglie la entertainment configuration da usare.

    Con `entertainment_area` valorizzato cerca per nome (case-insensitive),
    altrimenti prende la prima disponibile.

    Args:
        configs: Lista di configurazioni come restituite dal BridgeClient.

    Returns:
        La configurazione scelta.

    Raises:
        ValueError: Se non c'e' nessuna configurazione o il nome richiesto non
            corrisponde a nessuna.
    """
    if not configs:
        raise ValueError(
            "Nessuna entertainment area trovata sul Bridge. Creane una "
            "dall'app Philips Hue (sezione Entertainment) e riprova."
        )

    wanted = settings.entertainment_area.strip()
    if not wanted:
        return configs[0]

    needle = wanted.casefold()
    for config in configs:
        if config_name(config).casefold() == needle:
            return config

    available = ", ".join(repr(config_name(c)) for c in configs)
    raise ValueError(f"Entertainment area '{wanted}' non trovata. Disponibili: {available}.")


def _default_credentials() -> tuple[str | None, str, str]:
    """Credenziali per il percorso legacy senza parametri espliciti.

    Prima scelta: i secrets del .env (comportamento storico). Se mancano ma lo
    store multi-bridge ha almeno una voce, si usa il PRIMO bridge dello store
    (caso: pairing fatto solo dalla UI multi-bridge, .env mai popolato).

    Returns:
        Tupla ``(host, username, clientkey)``; host None = da risolvere.

    Raises:
        ValueError: Se non c'e' nessuna credenziale configurata.
    """
    if settings.hue_username and settings.hue_clientkey:
        return (
            None,
            settings.hue_username.get_secret_value(),
            settings.hue_clientkey.get_secret_value(),
        )
    from huebeat.bridgestore import BridgeStoreError, load_bridges

    try:
        stored = load_bridges()
    except BridgeStoreError:
        stored = []
    if stored:
        first = stored[0]
        return first.host, first.username, first.clientkey
    raise ValueError("Nessun bridge configurato: esegui il pairing (wizard o `huebeat pair`).")


def _pick_configuration(configs: list[dict], area_id: str) -> dict:
    """Sceglie l'area per id esplicito, o col criterio legacy se id vuoto.

    Args:
        configs: Lista di configurazioni come restituite dal BridgeClient.
        area_id: Id dell'entertainment configuration voluta; "" = criterio
            legacy (nome da settings.entertainment_area, altrimenti la prima).

    Returns:
        La configurazione scelta.

    Raises:
        ValueError: Se l'id richiesto non esiste sul bridge.
    """
    if not area_id:
        return select_configuration(configs)
    for config in configs:
        if config.get("id") == area_id:
            return config
    available = ", ".join(f"{config_name(c) or '?'} ({c.get('id')})" for c in configs)
    raise ValueError(f"Area con id '{area_id}' non trovata sul bridge. Disponibili: {available}.")


def setup_session(
    host: str | None = None,
    username: str | None = None,
    clientkey: str | None = None,
    area_id: str = "",
    abort: Callable[[], bool] | None = None,
) -> Session:
    """Prepara una sessione Entertainment pronta a ricevere colori.

    Blocco comune a `test-lights`, `run` e ai worker multi-bridge: risolve
    l'host, crea il client, sceglie l'area, avvia l'entertainment e apre lo
    stream. In caso di errore dopo aver avviato risorse, le chiude prima di
    propagare. Senza argomenti usa il percorso legacy (settings/.env, con
    fallback al primo bridge dello store).

    Args:
        host: Host del bridge; None = da settings o auto-discovery.
        username: Application key CLIP v2; None = credenziali di default.
        clientkey: PSK DTLS; None = credenziali di default.
        area_id: Id dell'area da sincronizzare; "" = criterio legacy.
        abort: Callable consultato durante l'handshake DTLS: True lo
            interrompe subito (stop del chiamante durante il setup).

    Returns:
        La sessione con client e stream aperti.

    Raises:
        ValueError: Se non si riesce a scegliere una entertainment area, o se
            l'area risulta gia' in streaming da un altro processo.
        BridgeError: Propagata da BridgeClient.
        StreamingError: Propagata da EntertainmentStream.
    """
    if username is None or clientkey is None:
        default_host, username, clientkey = _default_credentials()
        host = host or default_host
    if host is None:
        host = resolve_host()

    client = BridgeClient(host, username)
    stream: EntertainmentStream | None = None
    config_id = ""
    try:
        application_id = client.get_application_id()
        configs = client.list_entertainment_configurations()
        config = _pick_configuration(configs, area_id)
        config_id = config["id"]
        area_name = config_name(config) or config_id

        # Fail-fast se QUALSIASI area e' gia' in streaming: lo slot DTLS del
        # Bridge e' UNICO (globale, non per area), quindi anche un'area altrui
        # attiva farebbe fallire l'handshake dopo ~50 s di tentativi; e sul
        # rollback della nostra area attiva fermeremmo la sessione di CHI la
        # sta usando (verificato sul campo, due volte). Colpevole tipico:
        # Hue Sync desktop avviato in automatico con Windows.
        streaming = next((c for c in configs if c.get("status") == "active"), None)
        if streaming is not None:
            streaming_name = config_name(streaming) or streaming.get("id", "?")
            is_ours = streaming.get("id") == config_id
            config_id = ""  # niente stop nel rollback: lo slot non e' nostro
            if is_ours:
                raise ValueError(
                    f"L'area '{area_name}' e' gia' in streaming (Hue Sync, app Hue del "
                    "telefono o un altro run di huebeat): chiudi l'altro streamer e riprova."
                )
            raise ValueError(
                f"Un'altra area ('{streaming_name}') e' in streaming e il Bridge ha un "
                "solo slot DTLS: chiudi l'altro streamer (Hue Sync? app Hue?) e riprova."
            )
        # Dict grezzi dei canali: servono channel_id e position (x/y/z).
        channels = list(config.get("channels", []))

        logger.info(
            "Area '%s' (%d canali) su %s, %d fps.",
            area_name,
            len(channels),
            host,
            settings.stream_fps,
        )

        client.start_entertainment(config_id)
        stream = EntertainmentStream(host, application_id, clientkey, config_id)
        stream.connect(abort)
    except Exception:
        # Rollback: chiudi cio' che e' stato aperto prima di rilanciare.
        if stream is not None:
            stream.close()
        if config_id:
            client.stop_entertainment(config_id)
        client.close()
        raise

    return Session(client, stream, config_id, area_name, channels, host, application_id, clientkey)


def close_session(session: Session) -> None:
    """Chiude in ordine stream, entertainment e client, senza sollevare.

    Args:
        session: La sessione da chiudere.
    """
    try:
        session.stream.close()
    except Exception:
        logger.exception("Errore chiudendo lo stream.")
    try:
        session.client.stop_entertainment(session.config_id)
    except Exception:
        logger.exception("Errore fermando l'entertainment.")
    try:
        session.client.close()
    except Exception:
        logger.exception("Errore chiudendo il client.")


class SessionKeeper:
    """Tiene viva la sessione Entertainment: watchdog di status + riconnessione.

    Uso dal loop di rendering: chiamare `send(entries)` al posto di
    `session.stream.send(entries)`. Il keeper riconnette in due casi:

    - un send fallisce (errore socket/DTLS);
    - il watchdog (thread daemon) trova l'area "inactive" sul Bridge mentre
      dovremmo essere in streaming: e' la morte SILENZIOSA, il caso tipico
      (lo stream UDP continua a "riuscire" nel vuoto).

    La riconnessione avviene nel thread che chiama `send` (nel multi-bridge e'
    il thread di invio dedicato del worker, mai il loop di render) e ritenta
    senza limite, con backoff esponenziale e cap: Ctrl+C/stop restano possibili
    in ogni momento. Il client REST httpx e' thread-safe, quindi watchdog e
    riconnessione possono condividerlo.

    Un errore di VALIDAZIONE del frame (`FrameValidationError`) e'
    deterministico: `send` lo salta (log una volta) senza riconnettere, perche'
    riconnettere non lo risolverebbe. Solo gli errori di TRASPORTO innescano la
    riconnessione.
    """

    def __init__(
        self,
        session: Session,
        on_rediscover: Callable[[], str | None] | None = None,
    ) -> None:
        """Prepara il keeper per una sessione gia' connessa.

        Args:
            session: La sessione attiva da sorvegliare.
            on_rediscover: Callback opzionale invocato dopo
                REDISCOVERY_AFTER_FAILURES riconnessioni fallite consecutive;
                deve ritornare un nuovo host (gia' persistito dal chiamante) se
                il Bridge e' stato ritrovato a un IP diverso, altrimenti None.
        """
        self._session = session
        self._on_rediscover = on_rediscover
        self._stop_event = threading.Event()
        self._needs_reconnect = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_connect_monotonic = time.monotonic()
        self._backoff_seconds = RECONNECT_BACKOFF_START_SECONDS
        self._frame_validation_warned = False
        # Cicli di riconnessione falliti consecutivi: letto dalla UI via
        # `reconnect_attempts` per dire a che tentativo siamo (assegnazione di
        # int: atomica, nessun lock necessario tra thread di invio e snapshot).
        self._reconnect_failures = 0

    def start(self) -> None:
        """Avvia il watchdog in background (idempotente)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="session-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ferma il watchdog e fa uscire una riconnessione in corso.

        Setta lo stop event: oltre a fermare il watchdog, fa terminare subito il
        loop di `_reconnect` eventualmente in corso nel thread di invio del
        worker. Il join del watchdog e' con cap (WATCHDOG_JOIN_TIMEOUT_SECONDS):
        essendo daemon, oltre il timeout si abbandona senza trattenere lo
        shutdown.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=WATCHDOG_JOIN_TIMEOUT_SECONDS)
            self._thread = None

    def send(self, entries: list[tuple[int, tuple[float, float, float]]]) -> None:
        """Invia un frame, riconnettendo prima/dopo se necessario.

        Un errore di validazione del frame (`FrameValidationError`) e'
        deterministico: viene loggato una sola volta e il frame saltato, SENZA
        riconnettere. Un errore di trasporto setta `_needs_reconnect` (cosi' la
        UI vede subito lo stato) e riconnette.

        Args:
            entries: Le entry colore nel formato di `EntertainmentStream.send`.
        """
        if self._needs_reconnect.is_set():
            self._reconnect()
        try:
            self._session.stream.send(entries)
        except FrameValidationError as err:
            if not self._frame_validation_warned:
                logger.warning("Frame non valido, lo salto senza riconnettere: %s", err)
                self._frame_validation_warned = True
        except Exception as err:
            logger.warning("Invio frame fallito (%s): riconnetto.", err)
            self._needs_reconnect.set()
            self._reconnect()

    @property
    def reconnecting(self) -> bool:
        """Indica se e' stata rilevata una sessione morta non ancora riconnessa.

        Returns:
            True tra la rilevazione del problema e la riconnessione riuscita.
        """
        return self._needs_reconnect.is_set()

    @property
    def reconnect_attempts(self) -> int:
        """Tentativi di riconnessione gia' falliti nel ciclo in corso.

        Returns:
            Il numero di cicli falliti consecutivi mentre si sta riconnettendo
            (0 a riconnessione riuscita o quando non si sta riconnettendo).
        """
        if not self._needs_reconnect.is_set():
            return 0
        return self._reconnect_failures

    def _run(self) -> None:
        """Loop del watchdog: controlla lo status dell'area sul Bridge.

        Richiede WATCHDOG_NON_ACTIVE_POLLS letture "non active" consecutive prima
        di chiedere la riconnessione: una lettura isolata puo' essere un falso
        positivo (glitch REST del Bridge subito dopo lo start).
        """
        consecutive_non_active = 0
        while not self._stop_event.wait(WATCHDOG_INTERVAL_SECONDS):
            if self._needs_reconnect.is_set():
                # Riconnessione gia' in corso: azzero il conteggio.
                consecutive_non_active = 0
                continue
            if time.monotonic() - self._last_connect_monotonic < WATCHDOG_GRACE_SECONDS:
                continue
            try:
                configs = self._session.client.list_entertainment_configurations()
            except Exception as err:
                # REST momentaneamente giu' non implica sessione morta: non e' una
                # lettura "non active", lascio il conteggio invariato.
                logger.debug("Watchdog: controllo status fallito (%s).", err)
                continue
            status = next(
                (c.get("status") for c in configs if c.get("id") == self._session.config_id),
                None,
            )
            if status is not None and status != "active":
                consecutive_non_active += 1
                if consecutive_non_active >= WATCHDOG_NON_ACTIVE_POLLS:
                    logger.warning(
                        "Watchdog: area '%s' risulta '%s' per %d controlli: riconnessione.",
                        self._session.area_name,
                        status,
                        consecutive_non_active,
                    )
                    self._needs_reconnect.set()
            else:
                consecutive_non_active = 0

    def _reconnect(self) -> None:
        """Riconnette stream (e action start) con backoff, finche' non riesce.

        Gira nel thread che ha chiamato `send` (nel multi-bridge, il thread di
        invio del worker: mai il loop di render). Tra i tentativi applica un
        backoff esponenziale con cap, azzerato quando l'ultima connessione e'
        durata abbastanza (STABLE_CONNECTION_SECONDS): cosi' un drop genuino dopo
        ore riparte veloce, mentre il thrash "riconnesso ma subito rifallito" non
        busy-spinna. Ogni attesa passa da `_stop_event.wait`, quindi lo stop la
        interrompe subito. Dopo REDISCOVERY_AFTER_FAILURES fallimenti consecutivi
        tenta una ri-discovery dell'host (IP cambiato via DHCP).
        """
        session = self._session
        try:
            session.stream.close()
        except Exception:
            logger.debug("Chiusura del vecchio stream fallita (ignoro).")

        stable = time.monotonic() - self._last_connect_monotonic >= STABLE_CONNECTION_SECONDS
        if stable:
            self._backoff_seconds = RECONNECT_BACKOFF_START_SECONDS

        self._reconnect_failures = 0
        first_attempt = True
        while not self._stop_event.is_set():
            # Dopo una connessione stabile il primo tentativo parte SUBITO (drop
            # genuino: riprendere in fretta). Negli altri casi si attende il
            # backoff anche prima del primo tentativo, cosi' il percorso
            # "riconnesso e gia' rifallito" non gira senza sleep (busy-spin).
            if not (first_attempt and stable):
                if self._stop_event.wait(self._backoff_seconds):
                    return
                self._backoff_seconds = min(
                    self._backoff_seconds * RECONNECT_BACKOFF_FACTOR,
                    RECONNECT_BACKOFF_MAX_SECONDS,
                )
            first_attempt = False
            try:
                session.client.start_entertainment(session.config_id)
                stream = EntertainmentStream(
                    session.host,
                    session.application_id,
                    session.clientkey,
                    session.config_id,
                )
                # Abort sul proprio stop: una riconnessione in corso durante
                # lo shutdown non deve trattenere il thread per ~50 s. Cicli di
                # handshake corti: il feedback all'utente arriva prima e ci
                # pensa il backoff a distanziare i tentativi.
                stream.connect(
                    self._stop_event.is_set,
                    max_attempts=RECONNECT_HANDSHAKE_ATTEMPTS,
                )
            except Exception as err:
                failures = self._reconnect_failures + 1
                self._reconnect_failures = failures
                logger.warning(
                    "Riconnessione fallita (tentativo %d) verso %s: %s. Prossimo tra %.0f s.",
                    failures,
                    session.host,
                    err,
                    self._backoff_seconds,
                )
                self._maybe_rediscover(failures)
                continue
            failures = self._reconnect_failures
            self._reconnect_failures = 0
            session.stream = stream
            self._last_connect_monotonic = time.monotonic()
            self._needs_reconnect.clear()
            logger.info("Sessione Entertainment riconnessa (dopo %d fallimenti).", failures)
            return

    def _maybe_rediscover(self, failures: int) -> None:
        """Dopo N fallimenti consecutivi tenta di aggiornare l'host del Bridge.

        Invoca il callback di ri-discovery ogni REDISCOVERY_AFTER_FAILURES
        fallimenti: se restituisce un nuovo host, lo applica alla sessione (i
        tentativi successivi useranno il nuovo indirizzo). La persistenza nello
        store e' responsabilita' del callback.

        Args:
            failures: Numero di fallimenti consecutivi accumulati.
        """
        if self._on_rediscover is None or failures % REDISCOVERY_AFTER_FAILURES != 0:
            return
        new_host = self._on_rediscover()
        if new_host and new_host != self._session.host:
            logger.info("Bridge ritrovato a un nuovo host: %s -> %s.", self._session.host, new_host)
            self._session.host = new_host
