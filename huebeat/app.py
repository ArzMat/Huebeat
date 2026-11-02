"""App desktop di huebeat: finestra nativa (pywebview) + tray + avvio con Windows.

Il comando `python -m huebeat app` avvia:

- il server di controllo (huebeat.control) su 127.0.0.1;
- il motore di sync (huebeat.cli.run_engine) in un THREAD di lavoro, con vero
  start/stop dal pulsante power della UI (a differenza di `run`, dove il
  processo vive quanto il loop);
- una finestra nativa senza cornice (pywebview su WebView2: nessun browser)
  che carica la UI dal server locale;
- opzionale: icona nella tray (pystray) e avvio automatico con Windows
  (chiave Run di HKCU).

Il main thread appartiene a pywebview (vincolo della GUI su Windows); il loop
di rendering gira nel thread di lavoro e comunica solo via ControlBus.
"""

import argparse
import ctypes
import logging
import sys
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx

from huebeat import paths, tuning
from huebeat.config import settings
from huebeat.control import BIND_HOST, DEFAULT_PORT, ControlBus, ControlServer

logger = logging.getLogger(__name__)

APP_TITLE = "huebeat"
# Dimensioni della vista console e della vista strip (px).
CONSOLE_SIZE = (1180, 800)
STRIP_SIZE = (400, 660)
# Adattamento della finestra al contenuto (azione fit: dalla UI): correzione
# massima per richiesta e altezza minima di sicurezza (px fisici).
FIT_MAX_DELTA_PX = 500
FIT_MIN_HEIGHT_PX = 400

# Secondo avvio a porta occupata: attesa massima della risposta di chi ascolta
# e icona di errore del message box di Windows (MB_ICONERROR).
RUNNING_INSTANCE_TIMEOUT_SECONDS = 3.0
MESSAGE_BOX_ICON_ERROR = 0x10

# Avvio automatico: chiave Run per-utente (niente privilegi amministrativi).
AUTOSTART_REG_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
AUTOSTART_VALUE_NAME = "huebeat"

# Power-cycle del motore dopo il collegamento Spotify: attesa massima che il
# thread del loop esca (stop cooperativo) e passo di polling dell'attesa.
ENGINE_RESTART_TIMEOUT_SECONDS = 3.0
ENGINE_RESTART_POLL_SECONDS = 0.05

# Attesa massima all'uscita perche' il thread del motore chiuda sessione DTLS
# ed entertainment sul bridge (finally di run_engine): senza questo join i
# thread daemon verrebbero uccisi a meta' chiusura e l'area resterebbe "active"
# sul bridge, facendo fallire il riavvio ("area gia' in streaming").
ENGINE_STOP_JOIN_TIMEOUT_SECONDS = 12.0

# Attesa massima del thread "engine-restart" perche' il vecchio motore muoia
# prima di avviare quello nuovo: lo shutdown (join dei worker, chiusura DTLS,
# stop entertainment via REST) puo' durare diversi secondi.
ENGINE_RESTART_JOIN_TIMEOUT_SECONDS = 20.0


class EngineController:
    """Avvia e ferma il loop di sync in un thread di lavoro.

    Lo stop e' cooperativo: setta lo stop event e il loop esce al blocco audio
    successivo (al piu' ~0.1 s), ma la chiusura completa (join dei worker,
    chiusura DTLS, stop entertainment via REST) dura secondi. Per questo uno
    `start` che arriva mentre lo stop e' ancora in corso NON viene perso: viene
    ACCODATO a un thread di servizio che aspetta la morte del vecchio thread e
    poi avvia il motore fresco (un power OFF seguito subito da un power ON
    lasciava altrimenti il motore spento fino al riavvio dell'app).
    """

    def __init__(self, bus: ControlBus) -> None:
        """Prepara il controller (motore fermo).

        Args:
            bus: Il bus condiviso con server e UI.
        """
        self.bus = bus
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Protegge thread, stop event e flag del riavvio accodato: start e stop
        # arrivano dal thread del server di controllo, il riavvio da un thread
        # di servizio suo.
        self._lock = threading.Lock()
        self._pending_start = False

    @property
    def running(self) -> bool:
        """Indica se il thread del motore e' attivo.

        Returns:
            True se il loop di sync sta girando.
        """
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Avvia il motore, accodando il riavvio se uno stop e' in corso.

        Tre casi:

        - motore fermo: parte subito;
        - motore vivo senza stop in corso: no-op (e' gia' attivo);
        - motore vivo con stop in corso: il riavvio viene accodato a un thread
          "engine-restart" che attende la fine della chiusura e poi avvia il
          motore fresco. Pressioni ripetute restano idempotenti (un solo
          riavvio accodato) e un `stop` successivo lo annulla.
        """
        with self._lock:
            thread = self._thread
            if thread is not None and thread.is_alive():
                if not self._stop_event.is_set():
                    return  # motore gia' attivo
                if self._pending_start:
                    return  # riavvio gia' accodato
                self._pending_start = True
                threading.Thread(
                    target=self._restart_when_stopped,
                    args=(thread,),
                    name="engine-restart",
                    daemon=True,
                ).start()
                logger.info("Stop del motore in corso: riavvio accodato.")
                return
            self._start_locked()

    def stop(self) -> None:
        """Chiede al motore di fermarsi e annulla un riavvio accodato.

        Non attende (stop cooperativo). Un power OFF che arriva mentre un
        riavvio e' in coda deve vincere: il motore resta spento.
        """
        with self._lock:
            self._pending_start = False
            self._stop_event.set()

    def _start_locked(self) -> None:
        """Crea stop event e thread del motore; da chiamare col lock preso."""
        self.bus.last_error = None
        self._stop_event = threading.Event()
        # Lo stop event viaggia come argomento: se un riavvio ne crea uno nuovo
        # subito dopo, il thread appena avviato resta legato al PROPRIO.
        self._thread = threading.Thread(
            target=self._run,
            args=(self._stop_event,),
            name="huebeat-engine",
            daemon=True,
        )
        self._thread.start()

    def _restart_when_stopped(self, previous: threading.Thread) -> None:
        """Attende la fine del vecchio motore e avvia quello nuovo.

        Corpo del thread "engine-restart". Se nel frattempo e' arrivato uno
        stop (flag `_pending_start` azzerato) il riavvio viene abbandonato.

        Args:
            previous: Il thread del motore in fase di chiusura.
        """
        previous.join(ENGINE_RESTART_JOIN_TIMEOUT_SECONDS)
        with self._lock:
            if not self._pending_start:
                logger.info("Riavvio del motore annullato: e' arrivato uno stop.")
                return
            self._pending_start = False
            if previous.is_alive():
                # Chiusura anomala (bloccata oltre il timeout): si avvia
                # comunque, il lock di istanza singola del motore impedisce
                # che due loop pilotino le luci insieme.
                logger.warning(
                    "Il motore precedente non e' terminato entro %.0f s: avvio comunque.",
                    ENGINE_RESTART_JOIN_TIMEOUT_SECONDS,
                )
            self._start_locked()

    def join(self, timeout: float | None = None) -> None:
        """Attende la terminazione del thread del motore dopo `stop`.

        Da chiamare all'uscita, dopo `stop()`, cosi' il finally di run_engine
        chiude sessione DTLS ed entertainment sul bridge prima che i thread
        daemon vengano uccisi (altrimenti l'area resta "active" sul bridge).

        Args:
            timeout: Secondi massimi di attesa; None attende indefinitamente.
        """
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run(self, stop_event: threading.Event) -> None:
        """Corpo del thread: esegue run_engine e pubblica gli errori fatali.

        Args:
            stop_event: Lo stop event di QUESTA esecuzione del motore.
        """
        from huebeat.cli import run_engine

        try:
            run_engine(self.bus, stop_event)
        except Exception as exc:
            # Oltre al log: l'errore finisce nello stato del server, cosi' la
            # UI puo' dire PERCHE' il motore e' morto invece di restare muta.
            self.bus.last_error = str(exc)
            logger.exception("Motore terminato con errore.")


def is_autostart_enabled() -> bool:
    """Controlla se l'avvio automatico con Windows e' configurato.

    Returns:
        True se la chiave Run contiene la voce di huebeat.
    """
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, AUTOSTART_REG_PATH) as key:
            winreg.QueryValueEx(key, AUTOSTART_VALUE_NAME)
        return True
    except OSError:
        return False


def set_autostart(enabled: bool) -> None:
    """Attiva/disattiva l'avvio automatico con Windows (chiave Run di HKCU).

    Usa pythonw.exe (niente finestra console) se presente accanto
    all'interprete corrente, e avvia l'app direttamente in tray.

    Args:
        enabled: True per registrare l'avvio automatico, False per rimuoverlo.
    """
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, AUTOSTART_REG_PATH, 0, winreg.KEY_SET_VALUE
    ) as key:
        if enabled:
            if paths.is_frozen():
                # Da exe PyInstaller sys.executable E' l'eseguibile stesso.
                command = f'"{Path(sys.executable)}" --tray'
            else:
                interpreter = Path(sys.executable)
                windowless = interpreter.with_name("pythonw.exe")
                if windowless.exists():
                    interpreter = windowless
                command = f'"{interpreter}" -m huebeat app --tray'
            winreg.SetValueEx(key, AUTOSTART_VALUE_NAME, 0, winreg.REG_SZ, command)
            logger.info("Avvio automatico registrato: %s", command)
        else:
            try:
                winreg.DeleteValue(key, AUTOSTART_VALUE_NAME)
            except FileNotFoundError:
                pass
            logger.info("Avvio automatico rimosso.")


def _icon_path() -> Path | None:
    """Percorso dell'icona dell'app (assets/huebeat.ico), se esiste.

    In sviluppo cerca in assets/ nella root del repo; da exe PyInstaller
    nella cartella del bundle (sys._MEIPASS, inclusa via --add-data).

    Returns:
        Il Path dell'icona, o None se il file non c'e' (fallback generato).
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    candidate = base / "assets" / "huebeat.ico"
    return candidate if candidate.exists() else None


def _tray_icon_image():
    """Icona della tray: assets/huebeat.ico se presente, altrimenti generata.

    Returns:
        L'immagine PIL per pystray (il fallback e' il quadrato sfumato
        ciano-magenta 32x32 storico).
    """
    from PIL import Image

    icon_file = _icon_path()
    if icon_file is not None:
        try:
            return Image.open(icon_file).convert("RGBA")
        except Exception:  # noqa: BLE001 - icona corrotta: meglio il fallback
            logger.warning("Icona %s illeggibile: uso quella generata.", icon_file)

    size = 32
    image = Image.new("RGB", (size, size))
    # Sfumatura diagonale ciano (59,232,216) -> magenta (242,95,208).
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * size - 2)
            image.putpixel(
                (x, y),
                (
                    int(59 + (242 - 59) * t),
                    int(232 + (95 - 232) * t),
                    int(216 + (208 - 216) * t),
                ),
            )
    return image


def _ask_running_instance_to_show(port: int) -> int | None:
    """Chiede a chi ascolta sulla porta di mostrare la finestra di huebeat.

    Args:
        port: La porta di controllo occupata.

    Returns:
        Lo status HTTP della risposta, None se non risponde nessuno.
    """
    try:
        response = httpx.post(
            f"http://{BIND_HOST}:{port}/api/window",
            json={"action": "show"},
            timeout=RUNNING_INSTANCE_TIMEOUT_SECONDS,
            trust_env=False,
        )
    except httpx.HTTPError:
        return None
    return response.status_code


def _show_error(message: str) -> None:
    """Mostra un errore bloccante: l'exe non ha console su cui scriverlo."""
    ctypes.windll.user32.MessageBoxW(None, message, APP_TITLE, MESSAGE_BOX_ICON_ERROR)


def _handle_busy_port(port: int) -> int:
    """Gestisce il secondo avvio: la porta di controllo e' gia' occupata.

    Occupata non vuol dire huebeat (il server rifiuta il riuso dell'indirizzo
    apposta, ma la porta puo' essere di un altro programma), quindi prima si
    chiede a chi risponde. Un'istanza viva mostra la SUA finestra: una seconda
    finestra agganciata al suo server resterebbe orfana alla sua chiusura.

    Args:
        port: La porta di controllo occupata.

    Returns:
        Exit code del comando.
    """
    status = _ask_running_instance_to_show(port)
    if status == HTTPStatus.OK:
        logger.info("huebeat e' gia' in esecuzione: mostro la sua finestra.")
        return 0
    if status == HTTPStatus.NOT_IMPLEMENTED:
        # Istanza senza finestra propria (comando `run`): qui una finestra
        # agganciata e' l'unico modo di avere la UI. Cornice nativa (non
        # frameless): i bottoni finestra della UI non hanno chi comandare.
        import webview

        logger.info("huebeat gira senza finestra: apro la UI sul suo server.")
        webview.create_window(
            APP_TITLE,
            f"http://{BIND_HOST}:{port}/",
            width=CONSOLE_SIZE[0],
            height=CONSOLE_SIZE[1],
            frameless=False,
            background_color="#0B0B0F",
        )
        webview.start()
        return 0
    logger.error("Porta di controllo %d occupata da un altro programma (status %s).", port, status)
    _show_error(
        f"La porta {port} e' occupata da un altro programma e huebeat non puo' partire.\n"
        "Chiudi quel programma o cambia control_port nella configurazione."
    )
    return 1


class WindowController:
    """Esegue i comandi alla finestra nativa, ricordando se e' nascosta.

    Il resize di pywebview su Windows passa SWP_SHOWWINDOW a SetWindowPos: su una
    finestra nascosta la farebbe comparire. Da nascosta quindi niente resize: la
    UI si rimisura quando la finestra torna visibile.
    """

    def __init__(self, window: Any, *, hidden: bool) -> None:
        """Prepara il controller.

        Args:
            window: La finestra di pywebview.
            hidden: True se la finestra parte nascosta nella tray.
        """
        self._window = window
        self._hidden = hidden
        # Dimensione corrente della vista: base per l'adattamento al contenuto
        # (azione "fit:<delta>" dalla UI, che misura il proprio scrollHeight).
        self._width, self._height = CONSOLE_SIZE

    def handle(self, action: str) -> None:
        """Esegue un comando finestra (dal thread del server o della tray).

        Args:
            action: Una delle azioni di POST /api/window.
        """
        if action == "minimize":
            self._window.minimize()
        elif action == "show":
            self._show()
        elif action == "close":
            self._hidden = True
            self._window.hide()  # resta in tray; uscita vera dal menu tray
        elif action == "view:strip":
            self._set_view(STRIP_SIZE)
        elif action == "view:console":
            self._set_view(CONSOLE_SIZE)
        elif action.startswith("fit:"):
            self._fit(action)
        else:
            logger.warning("Azione finestra sconosciuta: %r", action)

    def _show(self) -> None:
        """Mostra la finestra e chiede alla UI di rimisurarsi."""
        self._hidden = False
        self._window.show()
        self._window.restore()
        # In un thread: evaluate_js attende il caricamento della pagina e il
        # chiamante (server o tray) non deve restare appeso.
        threading.Thread(target=self._request_fit, name="window-refit", daemon=True).start()

    def _request_fit(self) -> None:
        """Fa ripartire l'adattamento al contenuto dalla UI."""
        try:
            self._window.evaluate_js("scheduleFit()")
        except Exception:  # noqa: BLE001 - e' un ritocco estetico, mai bloccante
            logger.warning("Adattamento della finestra non riuscito.", exc_info=True)

    def _set_view(self, size: tuple[int, int]) -> None:
        """Passa alla dimensione base di una vista, se la finestra e' visibile."""
        if self._hidden:
            return
        self._width, self._height = size
        self._window.resize(*size)

    def _fit(self, action: str) -> None:
        """Adatta l'altezza al contenuto, se la finestra e' visibile.

        La UI chiede di crescere/stringersi di delta pixel FISICI (gia'
        moltiplicati per il devicePixelRatio): la finestra si adatta al
        contenuto e la barra di scorrimento sparisce, con qualsiasi scaling di
        Windows.
        """
        if self._hidden:
            return
        try:
            delta = int(action.split(":", 1)[1])
        except ValueError:
            logger.warning("Delta fit non numerico: %r", action)
            return
        delta = max(-FIT_MAX_DELTA_PX, min(FIT_MAX_DELTA_PX, delta))
        self._height = max(FIT_MIN_HEIGHT_PX, self._height + delta)
        self._window.resize(self._width, self._height)
        logger.info(
            "Finestra adattata al contenuto: %+d px -> %dx%d.",
            delta,
            self._width,
            self._height,
        )


def cmd_app(args: argparse.Namespace) -> int:
    """Avvia l'app desktop: server di controllo, motore, finestra e tray.

    Args:
        args: Argomenti del sottocomando (flag --tray per partire nascosta).

    Returns:
        Exit code del comando.
    """
    import webview

    from huebeat.audio import list_loopback_devices
    from huebeat.cli import _initial_overrides
    from huebeat.onboarding import PairingManager, SpotifyLinkManager
    from huebeat.process_audio import list_audio_apps

    # Niente _require_paired: l'app parte anche NON accoppiata e mostra il
    # wizard. Il motore parte solo a pairing gia' fatto (vedi in fondo).
    tuning.live.apply(settings.tuning.model_dump())

    bus = ControlBus(settings.effect)
    bus.list_devices = list_loopback_devices
    bus.list_audio_apps = list_audio_apps

    controller = EngineController(bus)

    def _on_power(on: bool) -> None:
        if on:
            controller.start()
        else:
            controller.stop()

    def _on_autostart(enabled: bool) -> None:
        try:
            set_autostart(enabled)
        except OSError:
            logger.exception("Configurazione avvio automatico fallita.")

    bus.on_power = _on_power
    bus.on_autostart = _on_autostart
    bus.get_autostart = is_autostart_enabled

    def _power_cycle() -> None:
        """Riavvia il motore SE era in esecuzione (per far partire il poller).

        Chiamato dopo il collegamento Spotify: il poller Spotify viene creato
        all'avvio del loop, quindi va fatto ripartire per prenderlo su. Se il
        motore era fermo non si fa nulla: partira' gia' col poller alla prossima
        accensione. Lo stop e' cooperativo (esce al blocco audio successivo):
        si attende che il thread termini prima di riavviare, altrimenti lo start
        idempotente non farebbe nulla.
        """
        if not controller.running:
            return
        controller.stop()
        deadline = time.monotonic() + ENGINE_RESTART_TIMEOUT_SECONDS
        while controller.running and time.monotonic() < deadline:
            time.sleep(ENGINE_RESTART_POLL_SECONDS)
        controller.start()

    pairing = PairingManager()
    pairing.on_paired = controller.start
    spotify_link = SpotifyLinkManager()
    spotify_link.on_linked = _power_cycle

    port = settings.control_port or DEFAULT_PORT
    server = ControlServer(
        bus,
        port,
        persisted=_initial_overrides(),
        pairing=pairing,
        spotify_link=spotify_link,
    )
    try:
        server.start()
    except OSError:
        return _handle_busy_port(port)

    start_hidden = bool(getattr(args, "tray", False))
    window = webview.create_window(
        APP_TITLE,
        f"http://{BIND_HOST}:{port}/",
        width=CONSOLE_SIZE[0],
        height=CONSOLE_SIZE[1],
        frameless=True,
        easy_drag=False,
        hidden=start_hidden,
        background_color="#0B0B0F",
    )

    window_controller = WindowController(window, hidden=start_hidden)
    bus.on_window = window_controller.handle

    # Tray: mostra/nascondi la finestra ed esci. Facoltativa: se pystray o
    # Pillow mancano l'app funziona comunque (chiudere = nascondere).
    tray = None
    try:
        import pystray

        def _tray_show(_icon: object, _item: object) -> None:
            window_controller.handle("show")

        def _tray_quit(icon: object, _item: object) -> None:
            icon.stop()  # type: ignore[attr-defined]
            controller.stop()
            # Aspetta che il motore chiuda sessione ed entertainment sul bridge
            # PRIMA di fermare server e finestra (l'ordine conta: il finally di
            # run_engine non dipende da server/window ma dai thread ancora vivi).
            controller.join(ENGINE_STOP_JOIN_TIMEOUT_SECONDS)
            server.stop()
            window.destroy()

        tray = pystray.Icon(
            APP_TITLE,
            _tray_icon_image(),
            APP_TITLE,
            menu=pystray.Menu(
                pystray.MenuItem("Apri huebeat", _tray_show, default=True),
                pystray.MenuItem("Esci", _tray_quit),
            ),
        )
        threading.Thread(target=tray.run, name="tray", daemon=True).start()
    except Exception:  # noqa: BLE001 - la tray e' un extra, mai bloccante
        logger.warning("Tray non disponibile (pystray/Pillow mancanti?).", exc_info=True)

    # Il motore parte subito SOLO con almeno un bridge accoppiato (store
    # multi-bridge o .env legacy): l'app E' il modo comodo di avviare il sync.
    # Non accoppiati, la finestra mostra il wizard e il motore partira' da
    # on_paired al termine del pairing.
    from huebeat.bridgestore import any_bridge_configured

    if any_bridge_configured():
        controller.start()

    try:
        webview.start()  # blocca il main thread finche' la finestra vive
    finally:
        if tray is not None:
            tray.stop()
        controller.stop()
        # Attende la chiusura pulita di sessione DTLS ed entertainment sul
        # bridge prima di fermare il server (vedi _tray_quit).
        controller.join(ENGINE_STOP_JOIN_TIMEOUT_SECONDS)
        server.stop()

    return 0
