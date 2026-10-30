"""Cattura audio per-processo via helper nativo Windows (Process Loopback).

Espone `ProcessCapture`, un context manager con la stessa interfaccia di
`huebeat.audio.LoopbackCapture`: apre l'helper nativo `hb_loopback.exe`, che
scrive su stdout PCM float32 interleaved stereo a 48000 Hz (silenzio = zeri,
mai gap), e restituisce blocchi mono float32 da `block_size` frame.

A differenza del loopback WASAPI di sistema, qui si cattura solo l'audio di
un'applicazione scelta per nome (via pycaw, tra le sessioni audio attive).
L'helper viene lanciato come `hb_loopback.exe <pid>`; se l'app non c'e' ancora
o si chiude, `read_block` restituisce silenzio sintetico e riaggancia l'app
appena ricompare (riavvio dell'app incluso).
"""

import logging
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import comtypes
import numpy as np
import psutil
from pycaw.pycaw import AudioUtilities

from huebeat import paths
from huebeat.analysis import HOP_DIVISOR

logger = logging.getLogger(__name__)

# Sample rate imposto all'helper nativo dal chiamante (Hz). L'API Windows
# Process Loopback lascia scegliere il formato a chi cattura: fissiamo 48000
# per coerenza col resto della pipeline.
PROCESS_SAMPLE_RATE = 48000

# Canali del flusso sorgente prodotto dall'helper (stereo interleaved).
SOURCE_CHANNELS = 2

# Attesa massima di dati prima di restituire un blocco di silenzio sintetico
# (secondi). Stessa semantica di audio.py: read_block NON deve mai bloccare
# oltre ~0.1 s, altrimenti il loop di run smetterebbe di mandare frame al
# Bridge e la sessione entertainment cadrebbe.
SILENCE_TIMEOUT_SECONDS = 0.1

# Passo di polling della disponibilita' dati durante l'attesa (secondi). A
# 0.001 s il jitter di rilevazione del blocco pronto scende da ~5 a ~1 ms:
# Python 3.11+ su Windows usa il timer ad alta risoluzione per gli sleep brevi,
# quindi lo sleep di 1 ms e' effettivamente ~1 ms e non arrotondato a ~15 ms.
AVAILABILITY_POLL_SECONDS = 0.001

# Intervallo minimo tra due tentativi di risoluzione del PID e (ri)avvio
# dell'helper (secondi): evita di martellare pycaw e Popen quando l'app manca.
RESOLVE_RETRY_SECONDS = 2.0

# Secondi di solo silenzio (nessun blocco con audio reale) con l'helper
# agganciato oltre i quali si ri-verifica il PID: se il PID radice risolto e'
# cambiato (app riavviata, o eravamo agganciati a un sottoalbero che non suona)
# si ri-aggancia l'helper. Il silenzio REALE a PID invariato (musica in pausa)
# risolve sempre allo stesso PID e quindi non causa retarget.
SILENT_REVERIFY_SECONDS = 30.0

# Numero di byte letti a ogni read() bloccante dal thread lettore. Un blocco
# stereo da 1024 frame float32 e' 1024*2*4 = 8192 byte: 4096 riempie il buffer
# a passi fini senza troppe syscall.
READER_CHUNK_BYTES = 4096

# Numero di byte per campione float32.
FLOAT32_BYTES = 4

# Nome dell'eseguibile dell'helper nativo.
HELPER_FILENAME = "hb_loopback.exe"

# Codice di uscita dell'helper quando l'attivazione della cattura fallisce.
HELPER_EXIT_ACTIVATION_FAILED = 2

# Codice di uscita dell'helper quando il processo target muore.
HELPER_EXIT_TARGET_DIED = 3


class ProcessCaptureError(Exception):
    """Errore di cattura audio per-processo."""


def _com_initialize() -> bool:
    """Inizializza COM sul thread corrente per l'uso di pycaw.

    pycaw usa COM, che va inizializzato PER THREAD: motore e server del
    frontend chiamano queste funzioni da thread propri, e chi arriva senza
    init otterrebbe "CoInitialize non e' stato chiamato" (visto sul campo
    dall'exe: dipendeva da QUALE thread importava comtypes per primo).

    Returns:
        True se questo CoInitialize va bilanciato con CoUninitialize; False
        se il thread era gia' inizializzato in una modalita' diversa (COM
        comunque utilizzabile, nessun bilanciamento dovuto).
    """
    try:
        comtypes.CoInitialize()
        return True
    except OSError:  # RPC_E_CHANGED_MODE: gia' inizializzato in altra modalita'
        return False


def _process_basename(process: psutil.Process) -> str:
    """Nome eseguibile del processo, casefold e senza suffisso `.exe`.

    Args:
        process: Il processo di cui leggere il nome (psutil.Process).

    Returns:
        Il nome normalizzato, oppure "" se psutil non riesce a leggerlo
        (processo sparito, accesso negato, zombie): "" non combacia mai con un
        nome cercato non vuoto, quindi il processo viene semplicemente saltato.
    """
    try:
        return process.name().casefold().removesuffix(".exe")
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return ""


def _root_ancestor_pid(process: psutil.Process, needle: str) -> int:
    """Risale i genitori omonimi e restituisce il PID dell'antenato radice.

    Parte dal processo della sessione audio e sale finche' il genitore esiste ed
    ha lo STESSO nome (casefold, senza `.exe`): restituisce il PID dell'antenato
    radice con quel nome. Serve alle app multi-processo come Chrome, dove l'audio
    nasce in un processo renderer figlio che il browser puo' riciclare:
    agganciare l'helper (in modalita' include-process-tree) alla radice cattura
    l'intero albero e sopravvive al riciclo del processo audio interno. Le app
    monoprocesso (es. Spotify) non hanno un genitore omonimo e restituiscono il
    proprio PID invariato.

    Un errore psutil durante la risalita (processo sparito, accesso negato,
    zombie) ferma la salita e restituisce il miglior antenato omonimo visto
    finora: fallback robusto al riciclo dei processi.

    Args:
        process: Processo della sessione audio da cui partire (psutil.Process).
        needle: Nome eseguibile cercato, casefold e senza suffisso `.exe`.

    Returns:
        Il PID dell'antenato radice omonimo (mai None: al minimo il PID di
        partenza).
    """
    root_pid = process.pid
    current = process
    try:
        while True:
            parent = current.parent()
            if parent is None or _process_basename(parent) != needle:
                break
            root_pid = parent.pid
            current = parent
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        # Il processo puo' sparire durante la risalita (Chrome ricicla i figli):
        # resta il miglior antenato omonimo raggiunto finora.
        pass
    return root_pid


def _pick_session_pid(sessions: list, needle: str) -> int | None:
    """Sceglie il PID da agganciare tra le sessioni audio, risalendo alla radice.

    Trova la prima sessione il cui processo ha il nome cercato (casefold, senza
    `.exe`), poi ne restituisce il PID dell'antenato radice omonimo (vedi
    `_root_ancestor_pid`). Le sessioni senza processo associato vengono saltate.
    Funzione pura (nessun COM, nessun effetto): iniettabile nei test con sessioni
    finte.

    Args:
        sessions: Sessioni audio (ognuna con attributo `.Process`, un
            psutil.Process o None).
        needle: Nome eseguibile cercato, casefold e senza suffisso `.exe`.

    Returns:
        Il PID dell'antenato radice del primo processo che combacia, oppure None
        se nessuna sessione combacia.
    """
    for session in sessions:
        process = session.Process
        if process is None:
            continue
        if _process_basename(process) != needle:
            continue
        return _root_ancestor_pid(process, needle)
    return None


def resolve_pid(process_name: str) -> int | None:
    """Risolve il PID radice di un'applicazione tra le sessioni audio attive.

    Cerca (case-insensitive, con o senza suffisso `.exe`) tra le sessioni audio
    note a Windows via pycaw; per la prima corrispondenza risale ai genitori
    omonimi e restituisce il PID dell'antenato radice con quel nome (vedi
    `_pick_session_pid` e `_root_ancestor_pid`). Cosi' l'helper nativo,
    agganciato in modalita' include-process-tree, cattura l'intero albero
    dell'app (Chrome compreso) invece di un singolo renderer che puo' non suonare
    o essere riciclato. Il pattern `_com_initialize` per-thread resta invariato.

    Args:
        process_name: Nome dell'eseguibile da cercare, con o senza `.exe`
            (es. "Spotify" o "spotify.exe").

    Returns:
        Il PID radice della prima sessione col nome corrispondente, oppure None
        se nessuna sessione attiva combacia.
    """
    needle = process_name.casefold().removesuffix(".exe")
    balanced = _com_initialize()
    try:
        return _pick_session_pid(AudioUtilities.GetAllSessions(), needle)
    finally:
        if balanced:
            comtypes.CoUninitialize()


def list_audio_apps() -> list[dict]:
    """Elenca le applicazioni con una sessione audio (per il menu del frontend).

    Un'app compare una sola volta anche se ha piu' sessioni; le sessioni senza
    processo associato (es. suoni di sistema) vengono saltate. Operazione da
    chiamare on-demand, mai nel loop di rendering.

    Returns:
        Lista di dict con `name` (nome eseguibile) e `pid`, ordinata per nome.
    """
    apps: dict[str, int] = {}
    balanced = _com_initialize()
    try:
        for session in AudioUtilities.GetAllSessions():
            process = session.Process
            if process is None:
                continue
            name = process.name()
            if name not in apps:
                apps[name] = process.pid
    finally:
        if balanced:
            comtypes.CoUninitialize()
    return [{"name": name, "pid": pid} for name, pid in sorted(apps.items())]


def default_helper_path() -> Path:
    """Risolve il percorso dell'helper nativo secondo dev o exe congelato.

    In un exe PyInstaller (`paths.is_frozen()`) l'helper vive nella cartella
    dei binari del bundle (`sys._MEIPASS`, che in onedir con PyInstaller 6
    punta a `_internal`), con fallback accanto all'eseguibile; in sviluppo sta
    in `<root repo>/native/hb_loopback/` (la root e' il parent del package).

    Returns:
        Il percorso atteso dell'helper (non ne verifica l'esistenza).
    """
    if paths.is_frozen():
        bundle_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidate = bundle_dir / HELPER_FILENAME
        if candidate.exists():
            return candidate
        return Path(sys.executable).parent / HELPER_FILENAME
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "native" / "hb_loopback" / HELPER_FILENAME


class ProcessCapture:
    """Cattura l'audio di una singola applicazione via helper nativo.

    Da usare come context manager: `__enter__` verifica la presenza
    dell'helper su disco e avvia il thread lettore; `__exit__` chiude tutto.
    L'helper effettivo viene lanciato in modo pigro dentro `read_block` (quando
    l'app da catturare compare tra le sessioni audio) e riavviato in automatico
    se l'app si riavvia.

    Attributes:
        process_name: Nome dell'applicazione da catturare (con o senza `.exe`).
        block_size: Numero di frame restituiti da ogni `read_block`.
    """

    def __init__(
        self,
        process_name: str,
        block_size: int = 1024,
        helper_path: Path | None = None,
        pid_resolver: Callable[[str], int | None] | None = None,
        command_builder: Callable[[int], list[str]] | None = None,
    ) -> None:
        """Inizializza la configurazione della cattura senza avviare nulla.

        Args:
            process_name: Nome dell'eseguibile dell'app da catturare.
            block_size: Numero di frame per blocco restituito da `read_block`.
            helper_path: Percorso dell'helper nativo; None per risolverlo con
                `default_helper_path`.
            pid_resolver: Funzione nome-app -> PID (o None); iniettabile solo
                per i test, default `resolve_pid`.
            command_builder: Funzione PID -> lista argomenti del comando da
                lanciare; iniettabile solo per i test, default
                `[str(helper_path), str(pid)]`.
        """
        self.process_name = process_name
        self.block_size = block_size
        # Hop tra finestre consecutive: read_block restituisce una finestra di
        # block_size campioni che avanza di hop (overlap del 50% con HOP_DIVISOR
        # = 2), raddoppiando il frame rate a parita' di risoluzione FFT.
        self._hop: int = max(1, block_size // HOP_DIVISOR)
        self._helper_path = helper_path if helper_path is not None else default_helper_path()
        self._pid_resolver = pid_resolver if pid_resolver is not None else resolve_pid
        self._command_builder = (
            command_builder if command_builder is not None else self._default_command
        )

        # Processo helper attivo e i suoi thread ausiliari (lettore stdout +
        # logger stderr). None quando l'app non e' catturata.
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

        # Buffer thread-safe dei byte grezzi letti dall'helper. Il thread
        # lettore fa append sotto _buffer_lock (memcpy brevi); read_block
        # preleva sotto lo stesso lock solo estrazioni veloci.
        self._buffer: deque[bytes] = deque()
        self._buffered_bytes = 0
        self._buffer_lock = threading.Lock()

        # Segnale di EOF (helper morto: stdout chiuso). Scritto dal thread
        # lettore, letto da read_block nel thread del loop.
        self._eof = False

        # Istante dell'ultimo tentativo di risoluzione/avvio helper (monotonic);
        # 0.0 = mai tentato (primo giro parte subito).
        self._last_resolve_attempt = 0.0

        # Nome app richiesto a caldo dal frontend, consumato da read_block;
        # None = nessuna richiesta. Riassegnazione atomica: scrivibile da
        # altri thread (stesso pattern di LoopbackCapture).
        self._pending_process_name: str | None = None

        # True se l'app e' attualmente catturata (helper vivo e agganciato):
        # guida il device_display_name ("in attesa" quando l'app non c'e').
        self._attached = False

        # PID radice a cui l'helper e' attualmente agganciato (per la ri-verifica
        # su silenzio prolungato); None quando non agganciato.
        self._attached_pid: int | None = None
        # Istante monotonic dell'ultimo blocco con audio REALE (campioni non a
        # zero). Impostato all'aggancio e aggiornato a ogni blocco non
        # silenzioso: guida la ri-verifica del PID dopo SILENT_REVERIFY_SECONDS
        # di solo silenzio (sintetico o zeri dallo stream).
        self._last_real_audio_monotonic: float = 0.0

        # Ultimo blocco PRE-downmix (shape (n, 2) float32) per il percorso
        # stereo parallelo; None se non c'e' ancora dato o sul silenzio
        # sintetico. Il flusso dell'helper e' sempre stereo (SOURCE_CHANNELS),
        # quindi qui non c'e' mai il caso "sorgente mono". Riassegnazione
        # atomica: leggibile da altri thread via last_stereo (nessun lock).
        self._last_stereo: np.ndarray | None = None

        # Ring buffer della finestra scorrevole: la finestra corrente (mono e la
        # sua versione pre-downmix) da cui, avanzando di hop, si compone la
        # prossima. `_primed` e' False finche' non si e' letta la prima finestra
        # piena; poi ogni giro consuma solo hop frame nuovi. Si azzera quando
        # l'helper viene fermato/riavviato (discontinuita' del flusso).
        self._primed: bool = False
        self._prev_window: np.ndarray | None = None
        self._prev_window_stereo: np.ndarray | None = None

    def _default_command(self, pid: int) -> list[str]:
        """Costruisce il comando di default per lanciare l'helper nativo.

        Args:
            pid: PID del processo target da passare all'helper.

        Returns:
            Lista argomenti `[str(helper_path), str(pid)]`.
        """
        return [str(self._helper_path), str(pid)]

    def request_device(self, process_name: str) -> None:
        """Richiede il passaggio a un'altra applicazione a caldo.

        Thread-safe: la richiesta e' una singola riassegnazione di attributo,
        consumata dal thread del loop alla prossima `read_block` (chiusura e
        riavvio dell'helper avvengono sempre nel thread del loop, mai qui).

        Args:
            process_name: Nome dell'app desiderata; un nome vuoto viene
                ignorato con un warning (non si puo' catturare "niente").
        """
        if not process_name:
            logger.warning("request_device con nome vuoto ignorata.")
            return
        self._pending_process_name = process_name

    def __enter__(self) -> "ProcessCapture":
        """Verifica l'helper su disco e prepara la cattura.

        L'helper vero viene avviato pigramente da `read_block` quando l'app
        compare tra le sessioni audio; qui si controlla solo che l'eseguibile
        esista (condizione permanente: se manca, la cattura non partira' mai).

        Returns:
            L'istanza stessa, pronta a `read_block`.

        Raises:
            ProcessCaptureError: Se l'helper nativo non esiste su disco.
        """
        if not self._helper_path.exists():
            raise ProcessCaptureError(
                f"Helper nativo non trovato: '{self._helper_path}'. "
                "Compilare hb_loopback.exe (cartella native/hb_loopback)."
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Chiude l'helper e ferma i thread ausiliari in modo pulito."""
        self.close()

    @property
    def sample_rate(self) -> int:
        """Sample rate del flusso in Hz (fisso, imposto all'helper).

        Returns:
            `PROCESS_SAMPLE_RATE`.
        """
        return PROCESS_SAMPLE_RATE

    @property
    def hop(self) -> int:
        """Passo di avanzamento della finestra tra due `read_block`, in campioni.

        Returns:
            L'hop (block_size // HOP_DIVISOR): il numero di frame nuovi tra una
            finestra e la successiva. Il chiamante ne ricava il block rate del
            loop come `sample_rate / hop`.
        """
        return self._hop

    @property
    def device_display_name(self) -> str:
        """Nome mostrato al frontend per la sorgente corrente.

        Returns:
            `"App: <nome>"` quando l'app e' catturata, `"App: <nome> (in
            attesa)"` finche' l'app non compare o dopo che si e' chiusa.
        """
        if self._attached:
            return f"App: {self.process_name}"
        return f"App: {self.process_name} (in attesa)"

    @property
    def last_stereo(self) -> np.ndarray | None:
        """Ultimo blocco PRE-downmix come immagine stereo (percorso parallelo).

        Returns:
            L'ultimo blocco letto in forma `(n, 2)` float32 (stereo
            interleaved dell'helper, pre-downmix), oppure None se non e' ancora
            arrivato nessun dato reale o se l'ultimo giro era silenzio
            sintetico. Il flusso per-processo e' sempre stereo, quindi non
            restituisce mai None per "sorgente mono".
        """
        return self._last_stereo

    def read_block(self) -> np.ndarray:
        """Legge un blocco e lo restituisce mono float32 normalizzato.

        Non blocca mai oltre ~`SILENCE_TIMEOUT_SECONDS`. Sequenza:
        1. consuma un eventuale retarget a caldo (chiude l'helper corrente e
           riparte sulla nuova app);
        2. se l'helper non e' attivo, ritenta risoluzione PID + avvio non piu'
           spesso di `RESOLVE_RETRY_SECONDS`; se l'app manca -> silenzio;
        3. se attivo, attende (fino al timeout) `block_size` frame stereo, li
           converte in mono; se non arrivano in tempo -> silenzio sintetico;
        4. se l'helper segnala EOF (app chiusa) -> cleanup e stato "in attesa"
           (riaggancio automatico quando l'app riappare).

        Non solleva mai per condizioni di runtime (app assente, helper morto):
        restituisce sempre silenzio e ritenta.

        Returns:
            Array numpy di shape `(block_size,)`, dtype float32; tutti zeri se
            l'app non e' catturata o non ha prodotto dati in tempo.
        """
        # 1. Retarget a caldo richiesto dal frontend.
        pending = self._pending_process_name
        if pending is not None:
            self._pending_process_name = None
            self.process_name = pending
            self._stop_helper()

        # 4. EOF rilevato dal thread lettore: l'helper e' morto, ripulisci.
        if self._eof:
            self._stop_helper()

        # 2. Helper non attivo: prova ad agganciare l'app (con throttling).
        if self._proc is None:
            self._try_spawn_helper()
            if self._proc is None:
                return self._synthetic_silence_after_timeout()

        # 3. Helper attivo: attendi un blocco stereo completo, poi verifica se un
        #    silenzio prolungato richiede di ri-agganciare l'helper.
        block = self._read_block_from_buffer()
        self._maybe_reverify_pid()
        return block

    def _need_bytes(self) -> int:
        """Byte dei frame nuovi da consumare per la finestra corrente.

        Al primo giro (finestra non innescata) serve un blocco pieno; dopo,
        solo `hop` frame nuovi (il resto viene dalla coda precedente).

        Returns:
            `need_frames * SOURCE_CHANNELS * FLOAT32_BYTES`, con need_frames pari
            a `block_size` finche' la finestra non e' innescata, poi `hop`.
        """
        need_frames = self.block_size if not self._primed else self._hop
        return need_frames * SOURCE_CHANNELS * FLOAT32_BYTES

    def _read_block_from_buffer(self) -> np.ndarray:
        """Estrae una finestra scorrevole dal buffer, o silenzio se in ritardo.

        La finestra e' larga `block_size` e avanza di `hop`: la prima chiamata
        consuma un blocco pieno, le successive solo `hop` frame nuovi ricomposti
        con la coda precedente. Sul timeout senza dati riparte "non innescata".

        Returns:
            Array mono float32 `(block_size,)`: dati reali downmixati se il
            buffer raggiunge i frame necessari entro il timeout, altrimenti
            silenzio sintetico.
        """
        needed = self._need_bytes()
        waited = 0.0
        while self._buffered_bytes < needed and waited < SILENCE_TIMEOUT_SECONDS:
            # EOF durante l'attesa: l'app si e' chiusa, esci col silenzio.
            if self._eof:
                break
            time.sleep(AVAILABILITY_POLL_SECONDS)
            waited += AVAILABILITY_POLL_SECONDS

        raw = self._take_bytes(needed)
        if raw is None:
            # Timeout senza dati: nessuno stereo reale, finestra da re-innescare.
            self._last_stereo = None
            self._primed = False
            return np.zeros(self.block_size, dtype=np.float32)

        # float32 stereo interleaved -> (need_frames, 2), mono = media dei canali.
        frames = np.frombuffer(raw, dtype=np.float32).reshape(-1, SOURCE_CHANNELS)
        new_stereo = frames.astype(np.float32)
        new_mono = frames.mean(axis=1).astype(np.float32)

        if np.any(new_stereo):
            # Audio reale (campioni non a zero): rinvia la ri-verifica del PID.
            # Il silenzio dello stream e' zeri esatti (contratto dell'helper),
            # quindi non azzera il conteggio; l'audio, anche debole, si'.
            self._last_real_audio_monotonic = time.monotonic()

        window, window_stereo = self._advance_window(new_mono, new_stereo)
        # Stereo coerente con la finestra restituita (stessa porzione, pre-downmix).
        self._last_stereo = window_stereo
        return window

    def _advance_window(
        self, new_mono: np.ndarray, new_stereo: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compone la finestra scorrevole dai frame nuovi appena estratti.

        Al primo giro (non innescata) i frame nuovi sono gia' una finestra
        piena; nei giri successivi la finestra e' la coda della precedente
        (scartati i primi hop frame) piu' i `hop` frame nuovi. Lo stereo segue
        esattamente la stessa finestra (pre-downmix).

        Args:
            new_mono: Frame mono nuovi appena estratti.
            new_stereo: Loro versione pre-downmix `(need, 2)` (mai None: il
                flusso dell'helper e' sempre stereo).

        Returns:
            Coppia `(window, window_stereo)`: la finestra mono `(block_size,)` e
            la sua immagine stereo `(block_size, 2)`.
        """
        hop = self._hop
        if not self._primed or self._prev_window is None or self._prev_window_stereo is None:
            window = new_mono
            window_stereo = new_stereo
            self._primed = True
        else:
            window = np.concatenate([self._prev_window[hop:], new_mono])
            window_stereo = np.concatenate([self._prev_window_stereo[hop:], new_stereo], axis=0)
        self._prev_window = window
        self._prev_window_stereo = window_stereo
        return window, window_stereo

    def _take_bytes(self, count: int) -> bytes | None:
        """Preleva esattamente `count` byte dal buffer, se disponibili.

        Sotto `_buffer_lock` per un tempo breve: solo estrazione e ricucitura
        degli chunk, nessuna conversione numpy.

        Args:
            count: Numero di byte richiesti.

        Returns:
            I `count` byte in testa al buffer, oppure None se non ce ne sono
            abbastanza.
        """
        with self._buffer_lock:
            if self._buffered_bytes < count:
                return None
            out = bytearray()
            while len(out) < count:
                chunk = self._buffer[0]
                take = count - len(out)
                if len(chunk) <= take:
                    out.extend(chunk)
                    self._buffer.popleft()
                else:
                    out.extend(chunk[:take])
                    self._buffer[0] = chunk[take:]
            self._buffered_bytes -= count
            return bytes(out)

    def _synthetic_silence_after_timeout(self) -> np.ndarray:
        """Attende il timeout senza dati e restituisce silenzio.

        Usato quando l'app non e' catturata: mantiene la cadenza del loop
        (~un blocco ogni `SILENCE_TIMEOUT_SECONDS`) senza mai bloccare oltre.

        Returns:
            Array mono float32 `(block_size,)` di zeri.
        """
        # App non catturata: nessun dato stereo reale in questo giro.
        self._last_stereo = None
        time.sleep(SILENCE_TIMEOUT_SECONDS)
        return np.zeros(self.block_size, dtype=np.float32)

    def _try_spawn_helper(self) -> None:
        """Tenta di risolvere il PID e avviare l'helper, con throttling.

        Non fa nulla se non e' passato `RESOLVE_RETRY_SECONDS` dall'ultimo
        tentativo. Se il PID non si risolve o l'avvio fallisce, resta senza
        helper (read_block restituira' silenzio).
        """
        now = time.monotonic()
        if now - self._last_resolve_attempt < RESOLVE_RETRY_SECONDS:
            return
        self._last_resolve_attempt = now

        pid = self._pid_resolver(self.process_name)
        if pid is None:
            return

        command = self._command_builder(pid)
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except OSError:
            logger.exception("Avvio dell'helper nativo fallito: %s", command)
            return

        logger.info(
            "Helper avviato per '%s' (PID target %d, comando %s).",
            self.process_name,
            pid,
            command,
        )
        self._proc = proc
        self._eof = False
        self._attached = True
        self._attached_pid = pid
        # Il conteggio del silenzio riparte dall'aggancio: l'helper ha appena
        # cominciato, non ha ancora prodotto audio reale.
        self._last_real_audio_monotonic = time.monotonic()
        self._start_reader(proc)
        self._start_stderr_logger(proc)

    def _maybe_reverify_pid(self) -> None:
        """Ri-verifica il PID dopo un lungo silenzio e ri-aggancia se e' cambiato.

        Se l'helper e' agganciato ma per piu' di `SILENT_REVERIFY_SECONDS` non
        arriva NESSUN blocco con audio reale (solo silenzio, sintetico o zeri
        dallo stream), rirsolve il PID dell'app. Se il PID radice risolto e'
        DIVERSO da quello agganciato (l'app e' stata riavviata, oppure eravamo
        agganciati a un sottoalbero che non suona) ferma l'helper: il prossimo
        `read_block` lo riavvia sul PID nuovo. Se il PID e' uguale o None non fa
        nulla e riparte il conteggio: il silenzio REALE (musica in pausa) col PID
        giusto risolve sempre allo stesso PID, quindi non causa retarget.

        Eseguita nel thread del loop, come `_try_spawn_helper`: la risoluzione
        passa dallo stesso `_pid_resolver` (COM inizializzato per chiamata).
        """
        if self._proc is None:
            return
        now = time.monotonic()
        if now - self._last_real_audio_monotonic < SILENT_REVERIFY_SECONDS:
            return
        # Riparte il conteggio: una sola ri-verifica per finestra di silenzio.
        self._last_real_audio_monotonic = now
        attached_pid = self._attached_pid
        new_pid = self._pid_resolver(self.process_name)
        if new_pid is None or new_pid == attached_pid:
            return
        logger.info(
            "Silenzio oltre %.0f s: PID radice cambiato da %s a %s, ri-aggancio l'helper.",
            SILENT_REVERIFY_SECONDS,
            attached_pid,
            new_pid,
        )
        self._stop_helper()
        # Respawn immediato al prossimo read_block (salta il throttling di retry).
        self._last_resolve_attempt = 0.0

    def _start_reader(self, proc: subprocess.Popen[bytes]) -> None:
        """Avvia il thread che legge stdout dell'helper e riempie il buffer.

        Args:
            proc: Processo helper da cui leggere.
        """
        thread = threading.Thread(
            target=self._reader_loop,
            args=(proc,),
            name="hb-loopback-reader",
            daemon=True,
        )
        self._reader_thread = thread
        thread.start()

    def _reader_loop(self, proc: subprocess.Popen[bytes]) -> None:
        """Legge stdout a blocchi e accumula nel buffer finche' arriva EOF.

        Args:
            proc: Processo helper da cui leggere lo stdout.
        """
        stdout = proc.stdout
        if stdout is None:
            self._eof = True
            return
        try:
            while True:
                chunk = stdout.read(READER_CHUNK_BYTES)
                if not chunk:
                    break
                with self._buffer_lock:
                    self._buffer.append(chunk)
                    self._buffered_bytes += len(chunk)
        except (OSError, ValueError):
            # stdout chiuso mentre leggevamo (close concorrente): trattalo
            # come EOF, la cleanup e' gestita da read_block/close.
            pass
        finally:
            self._eof = True

    def _start_stderr_logger(self, proc: subprocess.Popen[bytes]) -> None:
        """Avvia il thread che logga a DEBUG le righe di stderr dell'helper.

        Args:
            proc: Processo helper di cui drenare lo stderr.
        """
        thread = threading.Thread(
            target=self._stderr_loop,
            args=(proc,),
            name="hb-loopback-stderr",
            daemon=True,
        )
        self._stderr_thread = thread
        thread.start()

    def _stderr_loop(self, proc: subprocess.Popen[bytes]) -> None:
        """Drena stderr dell'helper riga per riga a livello DEBUG.

        Args:
            proc: Processo helper di cui leggere lo stderr.
        """
        stderr = proc.stderr
        if stderr is None:
            return
        try:
            for raw_line in iter(stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    logger.debug("hb_loopback[stderr]: %s", line)
        except (OSError, ValueError):
            pass

    def _stop_helper(self) -> None:
        """Termina l'helper corrente e ripulisce buffer e stato.

        Idempotente: senza helper attivo non fa nulla. Dopo la chiamata lo
        stato torna "in attesa" e read_block ritentera' l'aggancio.
        """
        proc = self._proc
        if proc is not None:
            self._terminate_process(proc)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)

        self._proc = None
        self._reader_thread = None
        self._stderr_thread = None
        self._attached = False
        self._attached_pid = None
        self._eof = False
        # Flusso interrotto: la finestra scorrevole va re-innescata da capo.
        self._primed = False
        self._prev_window = None
        self._prev_window_stereo = None
        with self._buffer_lock:
            self._buffer.clear()
            self._buffered_bytes = 0

    @staticmethod
    def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
        """Termina il processo helper: terminate, attesa breve, poi kill.

        Args:
            proc: Processo helper da terminare.
        """
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                logger.warning("Helper nativo non terminato nemmeno dopo kill.")

    def close(self) -> None:
        """Chiude l'helper e ferma i thread, idempotente."""
        self._stop_helper()
