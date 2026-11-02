"""Canale di controllo del frontend: bus condiviso + server HTTP/SSE locale.

Due pezzi:

- `ControlBus`: lo stato condiviso tra il server (thread propri) e il loop di
  rendering a 47 fps. REGOLA FERREA: il loop legge/scrive SOLO attributi con
  riassegnazione atomica (stesso pattern di `SpectrumPunchEffect.set_palette`)
  o `threading.Event`; nessun lock e' mai tenuto dal loop. Il server scrive le
  richieste (effetto, device, ruoli), il loop le consuma; il loop pubblica lo
  snapshot di stato, il server lo serve.
- `ControlServer`: HTTP su 127.0.0.1 (stdlib, thread daemon): serve la UI
  (webui/index.html), lo stato via Server-Sent Events (push ~10 Hz) e i
  comandi via POST. Niente dipendenze nuove per il canale di controllo.

Il server NON parla mai col Bridge: la sessione DTLS resta unica e vive nel
loop di run (vincolo del Bridge: un solo streamer).
"""

import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

from huebeat import bridgestore, localconfig, tuning
from huebeat.bridge import BridgeClient, BridgeError
from huebeat.effects import EFFECT_NAMES

if TYPE_CHECKING:
    from huebeat.onboarding import PairingManager, SpotifyLinkManager

logger = logging.getLogger(__name__)

# Bind SOLO su loopback: il pannello controlla le luci, non deve
# essere raggiungibile dalla LAN.
BIND_HOST = "127.0.0.1"
# Porta di default del server di controllo (override: control_port in config).
DEFAULT_PORT = 8775
# Periodo di push dello snapshot via SSE (secondi): ~10 Hz, letto dal dict
# pubblicato dal loop senza mai toccarne il thread.
SSE_INTERVAL_SECONDS = 0.1
# Ruoli assegnabili a un canale (piu' stringa vuota = distribuzione spaziale).
# "drums"/"wash" pilotano un canale dalle componenti HPSS (percussiva/armonica).
ROLE_NAMES = ("bass", "mid", "high", "rest", "voice", "drums", "wash")
# Prefisso della chiave override dei ruoli per-bridge in config.local.toml
# (sezione [channel_roles_by_bridge.<id>]): usato sia in salvataggio sia in
# rimozione al forget del bridge.
CHANNEL_ROLES_OVERRIDE_PREFIX = "channel_roles_by_bridge."

_WEBUI_PATH = Path(__file__).parent / "webui" / "index.html"

# Comandi con I/O di rete sincrono verso Spotify: gestiti FUORI dal lock dei
# comandi (non toccano ne' _overrides ne' lo stato versionato del bus), cosi'
# un timeout di rete non blocca gli altri comandi del pannello.
_UNLOCKED_COMMANDS = frozenset({"/api/spotify/playback", "/api/spotify/volume"})

# TTL delle cache dello stato servito via SSE (state_payload gira a ~10 Hz per
# client): senza cache si leggerebbero registro (autostart) e filesystem
# (bridge configurati) a ogni frame. Le cache sono anche invalidate a mano dai
# comandi che cambiano quegli stati (toggle autostart, pair, forget).
AUTOSTART_CACHE_TTL_SECONDS = 5.0
BRIDGE_CONFIGURED_CACHE_TTL_SECONDS = 2.0

# Parte COSTANTE del payload di stato (specs dei knob, effetti, ruoli): non
# cambia mai a runtime, quindi si costruisce una volta sola invece che a ogni
# frame SSE. Solo letture: mai mutato dopo la creazione.
_CONSTANT_STATE_PART = {
    "tuning_specs": tuning.TUNING_SPECS,
    "effects": list(EFFECT_NAMES),
    "roles": list(ROLE_NAMES),
}


class _QuietHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer senza traceback per le disconnessioni brusche.

    Un client che chiude la connessione keep-alive di colpo (UI chiusa,
    refresh del browser, curl) e' routine, non un errore: il default
    stamperebbe un traceback su stderr a ogni chiusura.
    """

    # NIENTE riuso dell'indirizzo: su Windows SO_REUSEADDR permette a una
    # seconda istanza di bindarsi sulla porta di un'istanza viva (e le
    # richieste finiscono a una delle due, imprevedibile: visto sul campo).
    # Con False la seconda istanza riceve un bind error e chiede a chi ascolta
    # sulla porta se e' un huebeat (app._handle_busy_port).
    allow_reuse_address = False

    def handle_error(self, request: object, client_address: object) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class ControlBus:
    """Stato condiviso tra server di controllo e loop di rendering.

    Direzioni di scrittura (un solo scrittore per attributo, letture atomiche):

    - server -> loop: `effect_request`, `device_request`, `roles_request`,
      `bridges_request`, `lights_enabled` (Event, condiviso anche con l'hotkey);
    - loop -> server: `snapshot` (dict nuovo riassegnato in blocco, mai mutato
      dopo la pubblicazione).

    I callback (`on_power`, `on_hotkey_change`, `on_autostart`, `on_window`)
    sono forniti dall'host (cmd_run o app) ed eseguiti NEL THREAD DEL SERVER:
    mai nel loop di rendering.

    Attributes:
        lights_enabled: Event condiviso col loop e con l'hotkey (set = luci on).
        effect_request: Nome dell'effetto desiderato; il loop lo confronta col
            corrente e ricrea l'effetto al cambio.
        device_request: Nome (sottostringa) del device audio richiesto, "" per
            il default; None = nessuna richiesta pendente. Consumato dal loop.
        roles_request: Tupla (versione, dict bridge_id -> {canale -> ruolo});
            il loop applica quando vede una versione nuova; i bridge assenti
            dalla mappa usano il flat legacy delle settings. Dict mai mutato
            dopo la pubblicazione.
        snapshot: Ultimo stato pubblicato dal loop (dict immutabile per patto).
        engine_running: True quando il loop di run e' attivo.
    """

    def __init__(self, effect: str = "spectrum") -> None:
        """Inizializza il bus.

        Args:
            effect: Nome dell'effetto attivo alla partenza.
        """
        self.lights_enabled = threading.Event()
        self.lights_enabled.set()
        self.effect_request: str = effect
        self.device_request: str | None = None
        self.roles_request: tuple[int, dict[str, dict[int, str]]] = (0, {})
        # Mappa bridge_id -> ""|"auto"|area_id, versionata ESATTAMENTE come
        # roles_request: il motore la consuma confrontando la versione.
        self.bridges_request: tuple[int, dict[str, str]] = (0, {})
        self.snapshot: dict = {}
        self.engine_running: bool = False
        # Ultimo errore fatale del motore (str) o None: la UI lo mostra quando
        # il motore muore invece di restare muta. Azzerato a ogni avvio.
        self.last_error: str | None = None
        # Callback host (opzionali), eseguiti nel thread del server.
        self.on_power = None
        self.on_hotkey_change = None
        self.on_autostart = None
        self.on_window = None
        # Notifica di sospensione/riattivazione avvenuta via UI (per riallineare
        # lo stato interno dell'hotkey): callable(bool) o None.
        self.on_lights_sync = None
        # Lettura dello stato dell'avvio automatico: callable() -> bool o None
        # (None = host che non lo gestisce, la UI nasconde l'opzione).
        self.get_autostart = None
        # Enumeratore dei device loopback, iniettato dall'host (evita che
        # questo modulo importi pyaudio: testabile senza hardware).
        self.list_devices = None
        # Enumeratore delle app con sessione audio (pycaw), iniettato dall'host.
        self.list_audio_apps = None
        # Richiesta di cambio sorgente audio: tupla (tipo, nome) con tipo
        # "device" | "process"; None = nessuna richiesta. Consumata dal loop.
        self.source_request: tuple[str, str] | None = None

    def request_roles(self, bridge_id: str, roles: dict[int, str]) -> None:
        """Pubblica i ruoli canale di UN bridge per il loop.

        La mappa pubblicata e' annidata (bridge_id -> {canale -> ruolo}) e
        conserva le voci degli altri bridge gia' pubblicate.

        Args:
            bridge_id: Id del bridge a cui appartengono i canali.
            roles: Mappa channel_id -> ruolo (dict nuovo, non mutato dopo).
        """
        version, current = self.roles_request
        merged = {**current, bridge_id: dict(roles)}
        self.roles_request = (version + 1, merged)


class ControlServer:
    """Server HTTP di controllo su 127.0.0.1, in thread daemon.

    Endpoints:

    - GET  /                UI (webui/index.html)
    - GET  /api/state       Stato completo (snapshot + tuning + specs)
    - GET  /api/devices     Elenco device loopback (enumerazione on-demand)
    - GET  /api/bridges     Bridge configurati (aree, sync, stato live)
    - GET  /events          SSE: push dello snapshot ~10 Hz
    - POST /api/effect      {"name": "adaptive"}
    - POST /api/device      {"name": "..."} ("" = output di default)
    - POST /api/lights      {"enabled": true|false} (sospendi/riattiva)
    - POST /api/power       {"on": true|false} (start/stop del motore, se
                            l'host lo supporta; altrimenti alias di lights)
    - POST /api/tuning      {"punch_gain": 0.7, ...}
    - POST /api/roles       {"bridge_id": "...", "roles": {"4": "bass", ...}}
                            ("" = spaziale; i ruoli valgono per QUEL bridge)
    - POST /api/hotkey      {"hotkey": "ctrl+shift+a"}
    - POST /api/autostart   {"enabled": true|false}
    - POST /api/window      {"action": "minimize"|"show"|"close"|"view:strip"|"view:console"}
    - POST /api/pair/discover   {} (cerca il Bridge sulla LAN)
    - POST /api/pair/start      {"host": "..."} ("" = usa quello scoperto)
    - POST /api/spotify/client_id {"client_id": "..."}
    - POST /api/spotify/login   {} (apre il browser per il login OAuth)
    - POST /api/spotify/playback {"action": "play"|"pause"|"next"|"previous"}
    - POST /api/spotify/volume  {"percent": 0-100} (volume del device attivo)
    - POST /api/bridges/sync    {"id": "...", "area": ""|"auto"|area_id}
    - POST /api/bridges/forget  {"id": "..."} (rimuove il bridge dallo store)

    Ogni modifica persistente viene riscritta in config.local.toml.
    """

    def __init__(
        self,
        bus: ControlBus,
        port: int = DEFAULT_PORT,
        persisted: dict | None = None,
        pairing: PairingManager | None = None,
        spotify_link: SpotifyLinkManager | None = None,
    ) -> None:
        """Prepara il server senza avviarlo.

        Args:
            bus: Il bus condiviso col loop di rendering.
            port: Porta TCP su cui ascoltare (loopback only).
            persisted: Override gia' persistiti da cui ripartire (effetto,
                device, hotkey, ruoli), per non perderli al riavvio.
            pairing: Manager di onboarding del Bridge (wizard); None se l'host
                non offre il pairing dalla UI (gli endpoint /api/pair/* danno 501).
            spotify_link: Manager di collegamento Spotify dalla UI; None se non
                offerto (gli endpoint /api/spotify/* danno 501).
        """
        self.bus = bus
        self.port = port
        self.pairing = pairing
        self.spotify_link = spotify_link
        # Stato persistito corrente: ogni modifica lo aggiorna e riscrive
        # config.local.toml per intero (file generato).
        self._overrides: dict = dict(persisted or {})
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        # ThreadingHTTPServer serve ogni richiesta in un thread: un solo lock
        # serializza i comandi che mutano stato condiviso (_overrides, bus
        # versionato), eliminando le read-modify-write non atomiche. Le
        # operazioni sotto lock sono brevi; la rete Spotify resta fuori (vedi
        # _UNLOCKED_COMMANDS).
        self._command_lock = threading.Lock()
        # Cache dello stato servito via SSE (~10 Hz per client): l'autostart
        # legge il registro, paired legge il filesystem. Tupla (istante, valore)
        # o None se mai letto; invalidata a mano dai comandi che li cambiano.
        self._autostart_cache: tuple[float, bool | None] | None = None
        self._bridge_configured_cache: tuple[float, bool] | None = None

    def start(self) -> None:
        """Avvia il server in un thread daemon (idempotente).

        Con port = 0 il sistema assegna una porta libera (utile nei test):
        leggerla da `bound_port`.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        handler = _build_handler(self)
        self._httpd = _QuietHTTPServer((BIND_HOST, self.port), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="control-server", daemon=True
        )
        self._thread.start()
        logger.info("Server di controllo su http://%s:%d/", BIND_HOST, self.bound_port)

    @property
    def bound_port(self) -> int:
        """Porta effettivamente in ascolto.

        Returns:
            La porta del socket se il server e' avviato, altrimenti quella
            richiesta.
        """
        if self._httpd is not None:
            return self._httpd.server_address[1]
        return self.port

    def stop(self) -> None:
        """Ferma il server e attende la terminazione del thread."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    # --- Persistenza ---

    def persist(self, **changes: object) -> None:
        """Aggiorna gli override e riscrive config.local.toml.

        Args:
            **changes: Coppie chiave -> valore da persistere (i dict diventano
                sezioni TOML, es. tuning={...}, channel_roles={...}).
        """
        self._overrides.update(changes)
        try:
            # Copia difensiva: localconfig.render itera il dict e un'altra
            # richiesta non deve poterlo mutare a meta' (dict changed size).
            localconfig.save(dict(self._overrides))
        except OSError:
            logger.exception("Scrittura di config.local.toml fallita.")

    # --- Spotify ---

    def _spotify_auth(self):
        """Costruisce l'autenticatore Spotify dai settings correnti.

        Import pigri come il resto del modulo: httpx/config non vengono
        caricati all'import di control (stesso pattern di bridges_payload).

        Returns:
            Tupla ``(auth, errore)``: auth = SpotifyAuth pronto e errore None,
            oppure auth None e errore = tupla (status, body) da restituire al
            client quando Spotify non e' configurato o collegato.
        """
        from huebeat import paths
        from huebeat.config import settings
        from huebeat.spotify import get_auth

        if settings.spotify_client_id is None:
            return None, (
                400,
                {"error": "Spotify non configurato: salva il client id e fai il login."},
            )
        token_path = paths.data_dir() / paths.SPOTIFY_TOKENS_FILENAME
        # Istanza condivisa (registro di modulo): stesso autenticatore del poller,
        # niente due oggetti che rinnovano/riscrivono lo stesso file token.
        auth = get_auth(settings.spotify_client_id.get_secret_value(), token_path)
        if not auth.has_tokens():
            return None, (400, {"error": "Spotify non collegato: fai il login dal pannello."})
        return auth, None

    # --- Comandi (eseguiti nel thread del server) ---

    def handle_command(self, path: str, payload: dict) -> tuple[int, dict]:
        """Esegue un comando POST e restituisce (status_code, corpo JSON).

        Serializza i comandi che mutano stato condiviso con un lock unico
        (ThreadingHTTPServer serve richieste concorrenti); i comandi con I/O di
        rete verso Spotify (`_UNLOCKED_COMMANDS`) restano fuori dal lock, cosi'
        un timeout di rete non blocca gli altri comandi.

        Args:
            path: Percorso della richiesta (es. "/api/effect").
            payload: Corpo JSON della richiesta gia' decodificato.

        Returns:
            Tupla (HTTP status, dict di risposta).
        """
        if path in _UNLOCKED_COMMANDS:
            return self._dispatch_command(path, payload)
        with self._command_lock:
            return self._dispatch_command(path, payload)

    def _dispatch_command(self, path: str, payload: dict) -> tuple[int, dict]:
        """Smista un comando POST verso l'handler del suo endpoint.

        Chiamato da `handle_command` (sotto lock per i comandi che mutano stato
        condiviso). Non acquisisce lock: e' `handle_command` a decidere.

        Args:
            path: Percorso della richiesta (es. "/api/effect").
            payload: Corpo JSON della richiesta gia' decodificato.

        Returns:
            Tupla (HTTP status, dict di risposta).
        """
        bus = self.bus
        if path == "/api/effect":
            name = str(payload.get("name", ""))
            if name not in EFFECT_NAMES:
                return 400, {"error": f"effetto sconosciuto: {name}"}
            bus.effect_request = name
            self.persist(effect=name)
            return 200, {"ok": True}

        if path == "/api/device":
            name = str(payload.get("name", ""))
            bus.device_request = name
            self.persist(audio_device=name)
            return 200, {"ok": True}

        if path == "/api/source":
            source_type = str(payload.get("type", ""))
            name = str(payload.get("name", ""))
            if source_type not in ("device", "process"):
                return 400, {"error": f"tipo sorgente sconosciuto: {source_type}"}
            bus.source_request = (source_type, name)
            if source_type == "device":
                self.persist(audio_source=source_type, audio_device=name)
            else:
                self.persist(audio_source=source_type, audio_process=name)
            return 200, {"ok": True}

        if path == "/api/lights":
            enabled = bool(payload.get("enabled"))
            if enabled:
                bus.lights_enabled.set()
            else:
                bus.lights_enabled.clear()
            if bus.on_lights_sync is not None:
                bus.on_lights_sync(enabled)
            return 200, {"ok": True}

        if path == "/api/power":
            on = bool(payload.get("on"))
            has_power = bus.on_power is not None
            if on:
                # Power = interruttore unico: riaccende SEMPRE anche le luci
                # sospese via hotkey, poi avvia il motore se l'host lo gestisce.
                bus.lights_enabled.set()
                if bus.on_lights_sync is not None:
                    bus.on_lights_sync(True)
                if has_power:
                    bus.on_power(True)
                    return 200, {"ok": True}
                return 200, {"ok": True, "degraded": "lights"}
            # on == False: ferma il motore; la sospensione via hotkey resta
            # indipendente (non tocchiamo lights_enabled quando c'e' il motore).
            if has_power:
                bus.on_power(False)
                return 200, {"ok": True}
            # Host senza gestione del motore: degrada a sospensione luci.
            bus.lights_enabled.clear()
            if bus.on_lights_sync is not None:
                bus.on_lights_sync(False)
            return 200, {"ok": True, "degraded": "lights"}

        if path == "/api/tuning":
            tuning.live.apply({k: v for k, v in payload.items() if isinstance(v, (int, float))})
            self.persist(tuning=tuning.live.as_dict())
            return 200, {"ok": True, "tuning": tuning.live.as_dict()}

        if path == "/api/roles":
            bridge_id = str(payload.get("bridge_id", ""))
            if not bridge_id:
                return 400, {"error": "bridge_id mancante"}
            raw_roles = payload.get("roles")
            if not isinstance(raw_roles, dict):
                return 400, {"error": "campo roles mancante o non valido"}
            roles: dict[int, str] = {}
            for key, value in raw_roles.items():
                if value == "":
                    continue  # "" = distribuzione spaziale: canale non elencato
                if value not in ROLE_NAMES:
                    return 400, {"error": f"ruolo sconosciuto: {value}"}
                try:
                    roles[int(key)] = str(value)
                except ValueError:
                    return 400, {"error": f"channel_id non numerico: {key}"}
            bus.request_roles(bridge_id, roles)
            persist_key = f"{CHANNEL_ROLES_OVERRIDE_PREFIX}{bridge_id}"
            self.persist(**{persist_key: {str(k): v for k, v in roles.items()}})
            return 200, {"ok": True}

        if path == "/api/hotkey":
            hotkey = str(payload.get("hotkey", ""))
            if bus.on_hotkey_change is not None:
                error = bus.on_hotkey_change(hotkey)
                if error:
                    return 400, {"error": error}
            self.persist(toggle_hotkey=hotkey)
            return 200, {"ok": True}

        if path == "/api/autostart":
            if bus.on_autostart is None:
                return 501, {"error": "avvio automatico non gestito da questo host"}
            bus.on_autostart(bool(payload.get("enabled")))
            self._autostart_cache = None  # forza la rilettura al prossimo stato
            return 200, {"ok": True}

        if path == "/api/window":
            if bus.on_window is None:
                return 501, {"error": "nessuna finestra da controllare"}
            bus.on_window(str(payload.get("action", "")))
            return 200, {"ok": True}

        if path == "/api/pair/discover":
            if self.pairing is None:
                return 501, {"error": "onboarding del Bridge non gestito da questo host"}
            self.pairing.discover()
            return 200, {"ok": True}

        if path == "/api/pair/start":
            if self.pairing is None:
                return 501, {"error": "onboarding del Bridge non gestito da questo host"}
            host = str(payload.get("host", ""))
            self.pairing.start(host)
            state = self.pairing.state
            if state.get("state") == "error":
                return 400, {"error": state.get("message", "pairing non avviato")}
            # Il pairing e' asincrono (thread): il paired vero arriva dopo. La
            # cache scade comunque entro il TTL; qui la azzeriamo per freschezza.
            self._bridge_configured_cache = None
            return 200, {"ok": True}

        if path == "/api/spotify/client_id":
            if self.spotify_link is None:
                return 501, {"error": "collegamento Spotify non gestito da questo host"}
            client_id = str(payload.get("client_id", "")).strip()
            if not client_id:
                return 400, {"error": "client id Spotify vuoto"}
            self.spotify_link.set_client_id(client_id)
            return 200, {"ok": True}

        if path == "/api/spotify/login":
            if self.spotify_link is None:
                return 501, {"error": "collegamento Spotify non gestito da questo host"}
            self.spotify_link.start_login()
            status = self.spotify_link.status()
            if status.get("state") == "error":
                return 400, {"error": status.get("message", "login non avviato")}
            return 200, {"ok": True}

        if path == "/api/spotify/playback":
            if self.spotify_link is None:
                return 501, {"error": "collegamento Spotify non gestito da questo host"}
            from huebeat.spotify import PLAYBACK_ACTIONS, SpotifyError, send_playback_command

            action = str(payload.get("action", ""))
            if action not in PLAYBACK_ACTIONS:
                return 400, {"error": f"azione sconosciuta: {action}"}
            auth, error = self._spotify_auth()
            if auth is None:
                return error
            # I/O di rete verso l'API Spotify: ammesso qui (thread del server),
            # mai nel loop di rendering.
            try:
                send_playback_command(auth, action)
            except SpotifyError as err:
                return 400, {"error": str(err)}
            return 200, {"ok": True}

        if path == "/api/spotify/volume":
            if self.spotify_link is None:
                return 501, {"error": "collegamento Spotify non gestito da questo host"}
            from huebeat.spotify import (
                VOLUME_MAX_PERCENT,
                VOLUME_MIN_PERCENT,
                SpotifyError,
                set_volume,
            )

            percent = payload.get("percent")
            # Dal JSON puo' arrivare float: accetta solo valori interi in range.
            if (
                isinstance(percent, bool)
                or not isinstance(percent, (int, float))
                or int(percent) != percent
                or not (VOLUME_MIN_PERCENT <= percent <= VOLUME_MAX_PERCENT)
            ):
                return 400, {"error": f"percentuale volume non valida: {percent!r}"}
            auth, error = self._spotify_auth()
            if auth is None:
                return error
            try:
                set_volume(auth, int(percent))
            except SpotifyError as err:
                return 400, {"error": str(err)}
            return 200, {"ok": True}

        if path == "/api/bridges/sync":
            bridge_id = str(payload.get("id", ""))
            if not bridge_id:
                return 400, {"error": "id del bridge mancante"}
            area = payload.get("area", "")
            if not isinstance(area, str):
                return 400, {"error": "area non valida"}
            sync_map = dict(self._overrides.get("bridge_sync", {}))
            sync_map[bridge_id] = area
            self.persist(bridge_sync=sync_map)
            version = bus.bridges_request[0] + 1
            bus.bridges_request = (version, dict(sync_map))
            return 200, {"ok": True}

        if path == "/api/bridges/forget":
            bridge_id = str(payload.get("id", ""))
            if not bridgestore.remove_bridge(bridge_id):
                return 400, {"error": "bridge sconosciuto"}
            # Rimuove anche l'override dei ruoli per-bridge, altrimenti resta
            # una sezione orfana in config.local.toml dopo il forget.
            self._overrides.pop(f"{CHANNEL_ROLES_OVERRIDE_PREFIX}{bridge_id}", None)
            sync_map = dict(self._overrides.get("bridge_sync", {}))
            sync_map.pop(bridge_id, None)
            self.persist(bridge_sync=sync_map)
            version = bus.bridges_request[0] + 1
            bus.bridges_request = (version, dict(sync_map))
            self._bridge_configured_cache = None  # potrebbe non restare alcun bridge
            return 200, {"ok": True}

        return 404, {"error": f"endpoint sconosciuto: {path}"}

    def _cached_autostart(self) -> bool | None:
        """Stato dell'avvio automatico, con cache a TTL (evita di leggere il registro a 10 Hz).

        Returns:
            True/False se l'host espone la lettura, None altrimenti (o se la
            lettura del registro fallisce).
        """
        if self.bus.get_autostart is None:
            return None
        now = time.monotonic()
        cache = self._autostart_cache
        if cache is not None and now - cache[0] < AUTOSTART_CACHE_TTL_SECONDS:
            return cache[1]
        try:
            value: bool | None = bool(self.bus.get_autostart())
        except Exception:  # noqa: BLE001 - lettura best-effort del registro
            logger.debug("Lettura stato avvio automatico fallita.", exc_info=True)
            value = None
        self._autostart_cache = (now, value)
        return value

    def _cached_bridge_configured(self) -> bool:
        """Se esiste almeno un bridge utilizzabile, con cache a TTL (evita I/O a 10 Hz).

        Returns:
            True se lo store o il .env legacy hanno un bridge.
        """
        now = time.monotonic()
        cache = self._bridge_configured_cache
        if cache is not None and now - cache[0] < BRIDGE_CONFIGURED_CACHE_TTL_SECONDS:
            return cache[1]
        value = bridgestore.any_bridge_configured()
        self._bridge_configured_cache = (now, value)
        return value

    def state_payload(self) -> dict:
        """Compone lo stato completo per GET /api/state.

        La parte costante (specs, effetti, ruoli) e' precomputata; autostart e
        paired passano da cache a TTL: a ~10 Hz per client non si legge piu'
        registro/filesystem a ogni frame.

        Returns:
            Dict con snapshot del loop, tuning corrente, specs dei knob,
            effetti disponibili e ruoli possibili.
        """
        return {
            "snapshot": self.bus.snapshot,
            "engine_running": self.bus.engine_running,
            "engine_error": self.bus.last_error,
            "lights_enabled": self.bus.lights_enabled.is_set(),
            "tuning": tuning.live.as_dict(),
            "autostart": self._cached_autostart(),
            # Onboarding: la UI mostra il wizard quando paired e' False e i
            # pannelli Spotify secondo lo stato. Solo booleani/messaggi: nessun
            # secret esce da qui.
            "paired": self._cached_bridge_configured(),
            "pairing": self.pairing.state if self.pairing else None,
            "spotify": self.spotify_link.status() if self.spotify_link else None,
            **_CONSTANT_STATE_PART,
        }

    def bridges_payload(self) -> dict:
        """Compone l'elenco dei bridge per GET /api/bridges.

        Per ogni bridge configurato riporta sync effettivo, aree disponibili
        (best-effort, interrogando il bridge) e stato live preso dallo snapshot
        del motore. MAI username/clientkey nella risposta.

        Returns:
            Dict ``{"bridges": [...]}``; con ``"error"`` in testa se la lettura
            dei bridge configurati fallisce del tutto.
        """
        from huebeat import multibridge

        try:
            bridges = multibridge.configured_bridges()
        except Exception as err:  # noqa: BLE001 - la UI mostra l'errore, non crasha
            logger.warning("Elenco bridge configurati non leggibile: %s", err)
            return {"bridges": [], "error": str(err)}

        sync_map = self._overrides.get("bridge_sync", {})
        wanted = multibridge.effective_sync([b.id for b in bridges], sync_map)
        # Stato live pubblicato dal motore: bridge_id -> voce con active/etc.
        live = {entry.get("id"): entry for entry in self.bus.snapshot.get("bridges", [])}

        result: list[dict] = []
        for creds in bridges:
            areas, area_error = self._list_areas(creds)
            live_entry = live.get(creds.id, {})
            result.append(
                {
                    "id": creds.id,
                    "name": creds.name,
                    "host": creds.host,
                    "sync": wanted.get(creds.id, multibridge.SYNC_OFF),
                    "areas": areas,
                    "active": bool(live_entry.get("active", False)),
                    "reconnecting": bool(live_entry.get("reconnecting", False)),
                    "error": area_error or str(live_entry.get("error", "")),
                }
            )
        return {"bridges": result}

    @staticmethod
    def _list_areas(creds: bridgestore.BridgeCredentials) -> tuple[list[dict], str]:
        """Elenca le entertainment area di un bridge (best-effort).

        Args:
            creds: Credenziali del bridge da interrogare.

        Returns:
            Tupla ``(aree, errore)``: aree = lista di ``{"id","name"}`` (vuota
            su errore), errore = messaggio se l'interrogazione fallisce, altrimenti "".
        """
        try:
            with BridgeClient(creds.host, creds.username) as client:
                configs = client.list_entertainment_configurations()
        except BridgeError as err:
            return [], str(err)
        areas = [
            {
                "id": config.get("id"),
                "name": config.get("name") or config.get("metadata", {}).get("name", ""),
            }
            for config in configs
        ]
        return areas, ""


def _build_handler(server: ControlServer) -> type[BaseHTTPRequestHandler]:
    """Costruisce la classe handler legata a un'istanza di ControlServer.

    Args:
        server: Il server a cui l'handler inoltra comandi e letture di stato.

    Returns:
        La classe handler per ThreadingHTTPServer.
    """

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            """Reindirizza i log di accesso al logging (livello DEBUG)."""
            logger.debug("HTTP %s", fmt % args)

        def _send_json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _host_allowed(self) -> bool:
            """Indica se l'header Host punta al loopback (anti DNS-rebinding).

            Accetta 127.0.0.1/localhost con o senza la porta in ascolto; ogni
            altro Host (es. un nome DNS che risolve a 127.0.0.1, usato per
            aggirare la same-origin policy) viene rifiutato.

            Returns:
                True se l'Host e' nell'allowlist loopback.
            """
            host = self.headers.get("Host", "").lower()
            port = server.bound_port
            allowed = {f"{BIND_HOST}:{port}", f"localhost:{port}", BIND_HOST, "localhost"}
            return host in allowed

        def _origin_allowed(self) -> bool:
            """Indica se l'header Origin manca o e' loopback (anti CSRF sui POST).

            Origin assente = richiesta non-browser (webview, curl): consentita.
            Se presente deve essere http://127.0.0.1:porta o http://localhost:porta.

            Returns:
                True se l'Origin manca o e' loopback.
            """
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            port = server.bound_port
            allowed = {f"http://{BIND_HOST}:{port}", f"http://localhost:{port}"}
            return origin.lower() in allowed

        def do_GET(self) -> None:  # noqa: N802 - firma imposta da BaseHTTPRequestHandler
            if not self._host_allowed():
                self._send_json(403, {"error": "host non consentito"})
                return
            if self.path == "/" or self.path == "/index.html":
                try:
                    page = _WEBUI_PATH.read_bytes()
                except OSError:
                    self._send_json(500, {"error": "webui/index.html mancante"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return

            if self.path == "/api/state":
                self._send_json(200, server.state_payload())
                return

            if self.path == "/api/devices":
                if server.bus.list_devices is None:
                    self._send_json(200, {"devices": []})
                    return
                try:
                    devices = server.bus.list_devices()
                except Exception as err:  # noqa: BLE001 - enumerazione best-effort
                    self._send_json(500, {"error": str(err)})
                    return
                self._send_json(200, {"devices": devices})
                return

            if self.path == "/api/audio_sources":
                devices = []
                apps = []
                if server.bus.list_devices is not None:
                    try:
                        devices = server.bus.list_devices()
                    except Exception:  # noqa: BLE001 - enumerazione best-effort
                        logger.warning("Enumerazione device fallita.", exc_info=True)
                if server.bus.list_audio_apps is not None:
                    try:
                        apps = server.bus.list_audio_apps()
                    except Exception:  # noqa: BLE001 - enumerazione best-effort
                        logger.warning("Enumerazione app audio fallita.", exc_info=True)
                self._send_json(200, {"devices": devices, "apps": apps})
                return

            if self.path == "/api/bridges":
                self._send_json(200, server.bridges_payload())
                return

            if self.path == "/events":
                self._serve_events()
                return

            self._send_json(404, {"error": "not found"})

        def _serve_events(self) -> None:
            """Stream SSE: pubblica lo stato ~10 Hz finche' il client resta."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    payload = json.dumps(server.state_payload())
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(SSE_INTERVAL_SECONDS)
            except BrokenPipeError, ConnectionResetError, ConnectionAbortedError:
                return  # client andato via: normale

        def do_POST(self) -> None:  # noqa: N802 - firma imposta da BaseHTTPRequestHandler
            if not self._host_allowed() or not self._origin_allowed():
                self._send_json(403, {"error": "richiesta non consentita (host/origin)"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "JSON invalido"})
                return
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "atteso un oggetto JSON"})
                return
            status, body = server.handle_command(self.path, payload)
            self._send_json(status, body)

    return _Handler
