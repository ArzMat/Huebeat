"""Coordinamento multi-bridge: quali bridge sincronizzare e i loro worker.

Questo modulo permette N bridge accoppiati, ognuno con la SUA scelta: quale
entertainment area sincronizzare, oppure "nessuna" (bridge spento). Qui vive
la logica comune a motore e frontend:

- `effective_sync`: applica la regola di default alla mappa persistita
  `[bridge_sync]` (vuota = compat: un solo bridge sincronizza in automatico);
- `resolve_plans`: store (+ migrazione del .env legacy) + mappa di sync ->
  lista dei bridge DA sincronizzare, con l'area scelta;
- `BridgeWorker`: ciclo di vita di UN bridge durante run: setup con retry in
  un thread proprio (i rifiuti DTLS transitori non bloccano gli altri bridge),
  SessionKeeper per la riconnessione, invio frame solo quando pronto.

Il loop di rendering resta l'unico a toccare `worker.effect` (effetto per
area: i canali si conoscono solo a sessione stabilita); il worker pubblica
`ready` per ULTIMO, cosi' il loop non vede mai una sessione a meta'.
"""

import logging
import threading
import time
from dataclasses import dataclass, field

from huebeat.bridgestore import (
    BridgeCredentials,
    BridgeStoreError,
    load_bridges,
    migrate_legacy,
    update_bridge_host,
)
from huebeat.config import settings
from huebeat.session import (
    REDISCOVERY_AFTER_FAILURES,
    Session,
    SessionKeeper,
    close_session,
    setup_session,
)

logger = logging.getLogger(__name__)

# Tipo di un frame: entry colore nel formato di EntertainmentStream.send.
Frame = list[tuple[int, tuple[float, float, float]]]

# Valori della mappa [bridge_sync] (config.local.toml, scritta dalla UI).
SYNC_OFF = ""  # il bridge non sincronizza
SYNC_AUTO = "auto"  # prima area disponibile (o entertainment_area legacy)

# Attesa tra due tentativi di setup di un worker: i rifiuti dell'handshake
# DTLS sono spesso transitori e spariscono da soli (visto sul campo).
SETUP_RETRY_SECONDS = 5.0
# Attesa massima del join del thread del worker allo stop (il thread puo' stare
# dentro una chiamata HTTP/DTLS bloccante: oltre, essendo daemon, si abbandona).
STOP_JOIN_TIMEOUT_SECONDS = 10.0
# Cap complessivo (secondi) all'attesa dello stop di TUTTI i worker in parallelo.
STOP_ALL_JOIN_TIMEOUT_SECONDS = 15.0
# Cadenza massima di invio effettivo verso il Bridge: il loop di analisi puo'
# salire oltre, ma non ha senso superare i 60 Hz sullo streaming DTLS.
MIN_SEND_INTERVAL_SECONDS = 1.0 / 60.0
# Timeout dell'attesa di un nuovo frame nel thread di invio: cappa la latenza di
# reazione allo stop anche se nessun frame arriva (lo stop sveglia comunque).
SEND_WAIT_TIMEOUT_SECONDS = 0.5
# Id fittizio del bridge legacy (solo .env, store non ancora migrato).
LEGACY_BRIDGE_ID = "legacy"


def effective_sync(bridge_ids: list[str], sync_map: dict[str, str]) -> dict[str, str]:
    """Applica la regola di default alla mappa di sync persistita.

    Args:
        bridge_ids: Id dei bridge configurati.
        sync_map: Mappa persistita bridge_id -> ""|"auto"|area_id.

    Returns:
        Mappa completa bridge_id -> ""|"auto"|area_id per TUTTI i bridge dati:
        con mappa vuota, un solo bridge sincronizza in "auto" (compat col
        comportamento a bridge singolo) e con piu' bridge partono tutti
        spenti (si sceglie dalla UI); con mappa non vuota, i bridge senza
        voce esplicita restano spenti.
    """
    if not sync_map:
        if len(bridge_ids) == 1:
            return {bridge_ids[0]: SYNC_AUTO}
        return {bridge_id: SYNC_OFF for bridge_id in bridge_ids}
    return {bridge_id: sync_map.get(bridge_id, SYNC_OFF) for bridge_id in bridge_ids}


def configured_bridges() -> list[BridgeCredentials]:
    """Bridge configurati: store (+ migrazione legacy) o fallback .env puro.

    Prova a migrare il .env legacy nello store al primo giro (serve il bridge
    raggiungibile per leggerne l'id); se la migrazione non riesce ma il .env
    e' configurato, restituisce una voce effimera con id "legacy" cosi' il
    motore funziona comunque offline.

    Returns:
        La lista dei bridge noti (eventualmente vuota).
    """
    try:
        bridges = load_bridges()
        if not bridges:
            migrated = migrate_legacy()
            bridges = [migrated] if migrated is not None else []
    except BridgeStoreError as err:
        logger.warning("Store dei bridge illeggibile (%s): uso il fallback legacy.", err)
        bridges = []
    if not bridges and settings.bridge_paired():
        bridges = [
            BridgeCredentials(
                id=LEGACY_BRIDGE_ID,
                host=settings.bridge_host,
                username=settings.hue_username.get_secret_value(),  # type: ignore[union-attr]
                clientkey=settings.hue_clientkey.get_secret_value(),  # type: ignore[union-attr]
                name="",
            )
        ]
    return bridges


@dataclass
class BridgePlan:
    """Un bridge da sincronizzare, con l'area scelta.

    Attributes:
        creds: Credenziali del bridge.
        area_id: Id dell'area da sincronizzare; "" = automatica (prima
            area, o entertainment_area legacy per nome).
    """

    creds: BridgeCredentials
    area_id: str = ""

    @property
    def key(self) -> tuple[str, str]:
        """Identita' del piano per il diff a caldo (bridge + area).

        Returns:
            Tupla (bridge_id, area_id).
        """
        return (self.creds.id, self.area_id)


def resolve_plans(sync_map: dict[str, str] | None = None) -> list[BridgePlan]:
    """Calcola i bridge da sincronizzare e con quale area.

    Args:
        sync_map: Mappa bridge_id -> ""|"auto"|area_id; None per usare
            quella delle settings ([bridge_sync]).

    Returns:
        I piani dei soli bridge accesi (mai quelli su "nessuno").
    """
    bridges = configured_bridges()
    if sync_map is None:
        sync_map = dict(settings.bridge_sync)
    wanted = effective_sync([b.id for b in bridges], sync_map)
    plans: list[BridgePlan] = []
    for creds in bridges:
        choice = wanted.get(creds.id, SYNC_OFF)
        if choice == SYNC_OFF:
            continue
        plans.append(BridgePlan(creds, "" if choice == SYNC_AUTO else choice))
    return plans


class BridgeWorker:
    """Ciclo di vita di un bridge durante run: setup con retry, keeper, invio.

    UN solo thread per worker, che vive per tutta la vita del worker: prima fa
    il setup (con retry ogni SETUP_RETRY_SECONDS: rifiuti DTLS transitori,
    bridge spento, area occupata; gli altri bridge non aspettano), poi diventa
    il thread DI INVIO (consumer). A setup completato parte il SessionKeeper
    (watchdog + riconnessione) e `ready` diventa True.

    `send` e' NON bloccante: deposita l'ultimo frame in uno slot single-slot
    "ultimo frame vince" (i frame vecchi non ancora inviati vengono
    sovrascritti). Il thread di invio consuma lo slot a max MIN_SEND_INTERVAL
    (60 Hz) e chiama `keeper.send`; QUALSIASI riconnessione avviene li' dentro,
    mai nel loop di render (che quindi non si blocca mai su un bridge morto).

    `effect` e `black_entries` sono di proprieta' del loop di rendering (unico
    scrittore/lettore).

    Attributes:
        plan: Il piano (bridge + area) di questo worker.
        effect: Effetto dell'area, creato dal loop quando il worker e' pronto.
        light_names: Mappa channel_id -> nome luce (CLIP v2, best-effort).
        black_entries: Frame nero dell'area (per la sospensione luci).
        last_error: Ultimo errore di setup leggibile dalla UI (None se ok).
    """

    def __init__(self, plan: BridgePlan) -> None:
        """Prepara il worker (nessun thread attivo).

        Args:
            plan: Bridge e area da sincronizzare.
        """
        self.plan = plan
        self.session: Session | None = None
        self.keeper: SessionKeeper | None = None
        self.effect: object | None = None
        self.light_names: dict[int, str] = {}
        self.black_entries: Frame = []
        self.last_error: str | None = None
        self._ready = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Slot single-slot "ultimo frame vince": il loop di render deposita qui
        # senza bloccare, il thread di invio consuma sempre l'ultimo disponibile.
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._pending_frame: Frame | None = None

    @property
    def ready(self) -> bool:
        """Indica se la sessione e' stabilita e si puo' inviare.

        Returns:
            True a setup completato (e keeper attivo).
        """
        return self._ready

    @property
    def reconnecting(self) -> bool:
        """Indica se il keeper sta riconnettendo una sessione morta.

        Returns:
            True durante una riconnessione, False altrimenti (o se non pronto).
        """
        return self.keeper is not None and self.keeper.reconnecting

    @property
    def reconnect_attempts(self) -> int:
        """Tentativi di riconnessione gia' falliti nel ciclo in corso.

        Returns:
            Il contatore del keeper (0 se il worker non ha ancora un keeper o
            non sta riconnettendo).
        """
        return self.keeper.reconnect_attempts if self.keeper is not None else 0

    @property
    def display_name(self) -> str:
        """Nome leggibile del bridge per log e UI.

        Returns:
            Il nome dello store, o l'host come ripiego.
        """
        return self.plan.creds.name or self.plan.creds.host or self.plan.creds.id

    def start(self) -> None:
        """Avvia il thread del worker (setup + invio); idempotente."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"bridge-{self.plan.creds.id[:8] or 'legacy'}",
            daemon=True,
        )
        self._thread.start()

    def send(self, entries: Frame) -> None:
        """Deposita l'ultimo frame nello slot (non bloccante); no-op se non pronto.

        Non contatta mai il Bridge dal thread chiamante: si limita a scrivere lo
        slot single-slot e svegliare il thread di invio, cosi' il loop di render
        non si blocca neppure con un bridge morto o in riconnessione.

        Args:
            entries: Entry colore nel formato di EntertainmentStream.send.
        """
        if not self._ready:
            return
        with self._frame_lock:
            self._pending_frame = entries
        self._frame_event.set()

    def stop(self) -> None:
        """Ferma il worker e chiude la sessione (mai solleva).

        Ordine studiato per non stallare: prima segnala lo stop (e sveglia il
        thread di invio), poi ferma il keeper - il che fa uscire subito una
        `_reconnect` eventualmente in corso nel thread di invio - e solo dopo fa
        il join del thread, cosi' non lo si attende mentre e' bloccato in una
        riconnessione. Il keeper viene fermato di nuovo DOPO il join: il thread
        potrebbe averlo creato proprio mentre arrivava lo stop, e keeper.stop()
        e' idempotente, cosi' il watchdog non resta mai vivo.
        """
        self._stop_event.set()
        self._frame_event.set()
        if self.keeper is not None:
            self.keeper.stop()
        if self._thread is not None:
            self._thread.join(timeout=STOP_JOIN_TIMEOUT_SECONDS)
            self._thread = None
        self._ready = False
        if self.keeper is not None:
            self.keeper.stop()
            self.keeper = None
        if self.session is not None:
            close_session(self.session)
            self.session = None

    def _run(self) -> None:
        """Corpo del thread del worker: prima il setup, poi il loop di invio."""
        if not self._setup():
            return
        self._send_loop()

    def _setup(self) -> bool:
        """Setup con retry finche' riesce o arriva lo stop.

        Ogni REDISCOVERY_AFTER_FAILURES fallimenti consecutivi tenta una
        ri-discovery dell'host: un cambio IP via DHCP ad app spenta (o a sync
        fermo) altrimenti non verrebbe mai recuperato, perche' la ri-discovery
        del keeper copre solo la riconnessione a sessione gia' stabilita.

        Returns:
            True se la sessione e' pronta (worker `ready`), False se lo stop e'
            arrivato prima di completare il setup.
        """
        from huebeat.bridge import BridgeError
        from huebeat.names import resolve_channel_names
        from huebeat.streaming import StreamingError

        creds = self.plan.creds
        failures = 0
        while not self._stop_event.is_set():
            try:
                # abort: uno stop durante l'handshake (~50 s a ciclo pieno)
                # interrompe il tentativo tra un handshake e l'altro, cosi' il
                # thread muore subito e il suo rollback non puo' stoppare
                # l'area di una sessione piu' nuova (visto sul campo).
                session = setup_session(
                    host=creds.host or None,
                    username=creds.username,
                    clientkey=creds.clientkey,
                    area_id=self.plan.area_id,
                    abort=self._stop_event.is_set,
                )
            except (ValueError, BridgeError, StreamingError) as err:
                failures += 1
                self.last_error = f"{err} (riprovo tra {SETUP_RETRY_SECONDS:.0f} s)"
                logger.warning(
                    "Bridge '%s': setup fallito: %s. Riprovo tra %.0f s.",
                    self.display_name,
                    err,
                    SETUP_RETRY_SECONDS,
                )
                if failures % REDISCOVERY_AFTER_FAILURES == 0:
                    new_host = self._rediscover_host()
                    if new_host is not None:
                        logger.info(
                            "Bridge '%s': trovato su %s (era %s), riprovo il setup.",
                            self.display_name,
                            new_host,
                            creds.host,
                        )
                        creds.host = new_host
                if self._stop_event.wait(SETUP_RETRY_SECONDS):
                    return False
                continue
            if self._stop_event.is_set():
                # Stop arrivato mentre il setup era in volo: niente orfani.
                close_session(session)
                return False
            self.light_names = resolve_channel_names(
                session.host, creds.username, session.config_id
            )
            self.session = session
            self.black_entries = [
                (channel["channel_id"], (0.0, 0.0, 0.0)) for channel in session.channels
            ]
            keeper = SessionKeeper(session, on_rediscover=self._rediscover_host)
            keeper.start()
            self.keeper = keeper
            self.last_error = None
            # Pubblicato per ULTIMO: da qui il loop vede il worker completo.
            self._ready = True
            logger.info(
                "Bridge '%s': sync attivo su '%s' (%d canali).",
                self.display_name,
                session.area_name,
                len(session.channels),
            )
            return True
        return False

    def _send_loop(self) -> None:
        """Consuma lo slot e invia via keeper a max 60 Hz (riconnessione inclusa).

        Attende un frame, applica la cadenza (se l'ultimo invio e' troppo recente
        dorme il resto dell'intervallo lasciando che i frame nuovi sovrascrivano
        lo slot: i sorpassati vengono droppati), poi invia. `keeper.send` puo'
        bloccare a lungo durante una riconnessione: e' voluto, blocca SOLO questo
        thread, non il loop di render.
        """
        keeper = self.keeper
        if keeper is None:
            return
        last_send = 0.0
        while not self._stop_event.is_set():
            if not self._frame_event.wait(SEND_WAIT_TIMEOUT_SECONDS):
                continue
            if self._stop_event.is_set():
                return
            wait = MIN_SEND_INTERVAL_SECONDS - (time.monotonic() - last_send)
            if wait > 0.0 and self._stop_event.wait(wait):
                return
            with self._frame_lock:
                frame = self._pending_frame
                self._pending_frame = None
                self._frame_event.clear()
            if frame is None:
                continue
            last_send = time.monotonic()
            keeper.send(frame)

    def _rediscover_host(self) -> str | None:
        """Ri-discovery dell'host del bridge dopo riconnessioni fallite (DHCP).

        Cerca il bridge per bridgeid (case-insensitive) tra quelli scoperti
        sulla LAN. Se l'host e' cambiato lo persiste nello store col nuovo
        `update_bridge_host` e lo restituisce, cosi' il keeper aggiorna la
        sessione. Il bridge legacy (solo .env) non ha un bridgeid reale nello
        store: aggiornamento best-effort a runtime, nessuna persistenza.

        Returns:
            Il nuovo host se e' cambiato, altrimenti None.
        """
        from huebeat.bridge import BridgeError, discover_bridges

        creds = self.plan.creds
        current_host = self.session.host if self.session is not None else creds.host
        try:
            found = discover_bridges()
        except BridgeError as err:
            logger.debug("Bridge '%s': ri-discovery fallita (%s).", self.display_name, err)
            return None

        if creds.id == LEGACY_BRIDGE_ID:
            # Legacy: niente bridgeid reale per il match e store non coinvolto
            # (solo .env). Store vuoto implica un setup a bridge singolo: uso il
            # primo trovato, a runtime, senza persistere.
            new_host = found[0]["host"]
            return new_host if new_host != current_host else None

        target = creds.id.casefold()
        for entry in found:
            if entry["id"].casefold() == target and entry["host"] != current_host:
                new_host = entry["host"]
                try:
                    update_bridge_host(creds.id, new_host)
                except BridgeStoreError as err:
                    logger.warning(
                        "Bridge '%s': host aggiornato a runtime ma non persistito (%s).",
                        self.display_name,
                        err,
                    )
                return new_host
        return None


@dataclass
class WorkerPool:
    """I worker attivi, indicizzati per bridge, con diff a caldo dei piani.

    Attributes:
        workers: Mappa bridge_id -> worker attivo.
    """

    workers: dict[str, BridgeWorker] = field(default_factory=dict)
    # Thread di servizio che fanno lo stop dei worker rimossi fuori dal loop di
    # render (uno stop puo' bloccare fino a ~10 s): non tenerli nel chiamante.
    _reapers: list[threading.Thread] = field(default_factory=list)

    def apply_plans(self, plans: list[BridgePlan]) -> None:
        """Allinea i worker ai piani: ferma i rimossi/cambiati, avvia i nuovi.

        Un bridge il cui piano non cambia (stessa area) resta connesso: si
        puo' accendere/spegnere un bridge senza interrompere gli altri. Lo stop
        dei worker rimossi e' delegato a un thread di servizio (reaper): la
        chiamata non blocca il loop di render neppure di un frame.

        Args:
            plans: I piani desiderati (solo bridge accesi).
        """
        desired = {plan.creds.id: plan for plan in plans}
        for bridge_id in list(self.workers):
            plan = desired.get(bridge_id)
            if plan is None or self.workers[bridge_id].plan.key != plan.key:
                worker = self.workers.pop(bridge_id)
                logger.info("Bridge '%s': sync fermato.", worker.display_name)
                self._reap(worker)
        for bridge_id, plan in desired.items():
            if bridge_id not in self.workers:
                worker = BridgeWorker(plan)
                self.workers[bridge_id] = worker
                worker.start()

    def _reap(self, worker: BridgeWorker) -> None:
        """Ferma un worker in un thread di servizio (fuori dal chiamante).

        Args:
            worker: Il worker rimosso da spegnere.
        """
        self._reapers = [t for t in self._reapers if t.is_alive()]
        thread = threading.Thread(
            target=worker.stop,
            name=f"reap-{worker.plan.creds.id[:8] or 'legacy'}",
            daemon=True,
        )
        thread.start()
        self._reapers.append(thread)

    def stop_all(self) -> None:
        """Ferma tutti i worker IN PARALLELO (fine run), con cap complessivo.

        Ogni stop puo' bloccare per la chiusura DTLS/REST: farli in sequenza
        significherebbe sommare le attese. Li avvia in parallelo e ne attende la
        fine entro STOP_ALL_JOIN_TIMEOUT_SECONDS complessivi (oltre, essendo
        daemon, si abbandonano). Include anche i reaper ancora in volo.
        """
        workers = list(self.workers.values())
        self.workers.clear()
        threads = [
            threading.Thread(
                target=worker.stop,
                name=f"stop-{worker.plan.creds.id[:8] or 'legacy'}",
                daemon=True,
            )
            for worker in workers
        ]
        for thread in threads:
            thread.start()
        pending = threads + [t for t in self._reapers if t.is_alive()]
        deadline = time.monotonic() + STOP_ALL_JOIN_TIMEOUT_SECONDS
        for thread in pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            thread.join(timeout=remaining)
        self._reapers.clear()
