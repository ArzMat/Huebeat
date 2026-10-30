"""Cattura audio di sistema via WASAPI loopback (PyAudioWPatch).

Espone `LoopbackCapture`, un context manager che apre lo stream loopback del
device di output (default o scelto per nome) e restituisce blocchi mono float32
normalizzati, pronti per l'analisi FFT/onset a valle.
"""

import logging
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio

from huebeat.analysis import HOP_DIVISOR

logger = logging.getLogger(__name__)

# Pa_Initialize/Pa_Terminate di PortAudio NON sono thread-safe: creare o
# terminare istanze PyAudio in parallelo (l'apertura della cattura nel thread
# del motore vs l'enumerazione device del frontend nel thread del server)
# corrompe la tabella dei device e abortisce il processo con l'assertion C
# "defaultInputDevice < deviceCount" o un access violation (riprodotto con
# stress test, visto sul campo all'avvio dell'app). OGNI
# creazione/enumerazione/terminazione passa da questo lock; il percorso
# per-frame di read_block non lo tocca mai. Rientrante perche' la cleanup
# d'errore di __enter__ gira gia' sotto lock.
_pa_lifecycle_lock = threading.RLock()

# Formato di campionamento fisso: int16 (2 byte per campione, range -32768..32767).
_SAMPLE_FORMAT = pyaudio.paInt16

# Divisore per normalizzare int16 in float32 [-1, 1]: 2**15.
_INT16_FULL_SCALE = 32768.0

# Attesa massima di dati prima di restituire un blocco di silenzio sintetico
# (secondi). VINCOLO: il loopback WASAPI non consegna frame quando sull'output
# non suona nulla; senza timeout read_block resterebbe bloccato, il loop di run
# smetterebbe di mandare frame e il Bridge chiuderebbe la sessione entertainment
# (timeout ~10 s senza dati). Con 0.1 s il loop scende a ~10 fps di frame neri
# durante il silenzio: sessione viva, luci spente come atteso.
_SILENCE_TIMEOUT_SECONDS = 0.1
# Passo di polling della disponibilita' dati durante l'attesa (secondi). A
# 0.001 s il jitter di rilevazione del blocco pronto scende da ~5 a ~1 ms:
# Python 3.11+ su Windows usa il timer ad alta risoluzione per gli sleep brevi,
# quindi lo sleep di 1 ms e' effettivamente ~1 ms e non arrotondato a ~15 ms.
_AVAILABILITY_POLL_SECONDS = 0.001

# Blocchi sintetici CONSECUTIVI dopo i quali ricontrollare qual e' l'output di
# default (~3 s a ~10 blocchi sintetici/s). Il ricontrollo e' costoso: PortAudio
# congela l'elenco device a init, serve una nuova istanza PyAudio; per questo
# scatta solo quando il device corrente tace, che e' il sintomo tipico di
# "l'utente ha cambiato output e l'audio ora va altrove".
_SILENT_BLOCKS_BEFORE_DEVICE_CHECK = 30


class AudioCaptureError(Exception):
    """Errore di cattura audio."""


def list_loopback_devices() -> list[dict]:
    """Elenca i device loopback WASAPI disponibili (per il frontend).

    Crea un'istanza PyAudio usa-e-getta (PortAudio congela l'elenco a init,
    quindi ogni chiamata vede lo stato attuale del sistema). Operazione
    relativamente costosa: da chiamare on-demand, mai nel loop di rendering.

    Returns:
        Lista di dict con `name`, `is_default` (True se e' il loopback
        dell'output di default) e `sample_rate` nativo.

    Raises:
        AudioCaptureError: Se WASAPI non e' disponibile.
    """
    with _pa_lifecycle_lock:
        audio = pyaudio.PyAudio()
        try:
            try:
                wasapi_info = audio.get_host_api_info_by_type(pyaudio.paWASAPI)
            except OSError as err:
                raise AudioCaptureError("WASAPI non disponibile su questo sistema.") from err
            default_name = audio.get_device_info_by_index(wasapi_info["defaultOutputDevice"])[
                "name"
            ]
            return [
                {
                    "name": loopback["name"],
                    "is_default": default_name in loopback["name"],
                    "sample_rate": int(loopback["defaultSampleRate"]),
                }
                for loopback in audio.get_loopback_device_info_generator()
            ]
        finally:
            audio.terminate()


class LoopbackCapture:
    """Cattura l'audio di sistema dal loopback di un device di output WASAPI.

    Da usare come context manager: l'apertura effettiva dello stream e la
    risoluzione del device avvengono in `__enter__`, la chiusura pulita in
    `__exit__`.

    Attributes:
        device_name: Sottostringa (case-insensitive) del nome del device da
            catturare; vuota per il loopback dell'output di default.
        block_size: Numero di frame letti a ogni `read_block`.
    """

    def __init__(self, device_name: str = "", block_size: int = 1024) -> None:
        """Inizializza la configurazione della cattura senza aprire nulla.

        Args:
            device_name: Sottostringa case-insensitive con cui cercare il device
                loopback. Vuota per usare il loopback dell'output di default.
            block_size: Numero di frame per blocco restituito da `read_block`.
        """
        self.device_name = device_name
        self.block_size = block_size
        # Hop tra finestre consecutive: read_block restituisce una finestra di
        # block_size campioni che avanza di hop (overlap del 50% con HOP_DIVISOR
        # = 2), raddoppiando il frame rate a parita' di risoluzione FFT.
        self._hop: int = max(1, block_size // HOP_DIVISOR)
        self._pyaudio: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._sample_rate: int = 0
        self._channels: int = 0
        # Ring buffer della finestra scorrevole: la finestra corrente (mono e la
        # sua versione pre-downmix) da cui, avanzando di hop, si compone la
        # prossima. `_primed` e' False finche' non si e' letta la prima finestra
        # piena (block_size campioni); dopo, ogni giro legge solo hop campioni
        # nuovi. Si azzera a ogni (ri)apertura dello stream e sul silenzio.
        self._primed: bool = False
        self._prev_window: np.ndarray | None = None
        self._prev_window_stereo: np.ndarray | None = None
        # Nome del device loopback attualmente aperto (per rilevare i cambi).
        self._device_display_name: str = ""
        # Blocchi sintetici consecutivi restituiti (azzerato a ogni blocco reale).
        self._synthetic_streak: int = 0
        # Nome device richiesto dal frontend, consumato da read_block; None =
        # nessuna richiesta. Riassegnazione atomica: scrivibile da altri thread.
        self._pending_device_name: str | None = None
        # Ultimo blocco PRE-downmix (shape (n, 2) float32) per il percorso
        # stereo parallelo; None se non c'e' ancora dato, sul silenzio
        # sintetico o se il device e' mono. Riassegnazione atomica: leggibile
        # da altri thread via la property last_stereo (nessun lock nuovo).
        self._last_stereo: np.ndarray | None = None

    def request_device(self, device_name: str) -> None:
        """Richiede il passaggio a un altro device loopback a caldo.

        Thread-safe: la richiesta e' una singola riassegnazione di attributo,
        consumata dal thread del loop alla prossima `read_block` (la riapertura
        dello stream avviene sempre nel thread del loop, mai qui).

        Args:
            device_name: Sottostringa case-insensitive del device desiderato;
                vuota per tornare al loopback dell'output di default.
        """
        self._pending_device_name = device_name

    def __enter__(self) -> LoopbackCapture:
        """Apre PyAudio e lo stream loopback sul device risolto.

        Returns:
            L'istanza stessa, con stream aperto e pronto alla lettura.

        Raises:
            AudioCaptureError: Se WASAPI non e' disponibile o il device
                richiesto non viene trovato.
        """
        with _pa_lifecycle_lock:
            self._pyaudio = pyaudio.PyAudio()
            try:
                device = self._resolve_device(self._pyaudio)
                self._open_stream(self._pyaudio, device)
            except Exception:
                # Se qualcosa fallisce dopo aver creato PyAudio, non lasciarlo
                # appeso (il lock e' rientrante: _cleanup lo riprende).
                self._cleanup()
                raise

        return self

    def _open_stream(self, audio: pyaudio.PyAudio, device: dict) -> None:
        """Apre lo stream di cattura sul device dato e aggiorna lo stato.

        Args:
            audio: Istanza PyAudio con cui aprire lo stream.
            device: Device info del loopback su cui aprire.
        """
        # Sample rate e canali NATIVI del device, letti dal device info.
        self._sample_rate = int(device["defaultSampleRate"])
        self._channels = int(device["maxInputChannels"])
        self._device_display_name = device["name"]

        logger.info(
            "Apertura loopback: '%s' (index %d, %d Hz, %d canali)",
            device["name"],
            device["index"],
            self._sample_rate,
            self._channels,
        )

        self._stream = audio.open(
            format=_SAMPLE_FORMAT,
            channels=self._channels,
            rate=self._sample_rate,
            frames_per_buffer=self.block_size,
            input=True,
            input_device_index=device["index"],
        )
        # Stream nuovo: la finestra scorrevole riparte da zero (rate e canali
        # possono essere cambiati, la continuita' col vecchio stream non vale).
        self._primed = False
        self._prev_window = None
        self._prev_window_stereo = None

    def __exit__(self, exc_type, exc, tb) -> None:
        """Chiude lo stream e termina PyAudio in modo pulito."""
        self._cleanup()

    @property
    def sample_rate(self) -> int:
        """Sample rate nativo del device in Hz.

        Returns:
            Il sample rate; 0 finche' il context manager non e' stato aperto.
        """
        return self._sample_rate

    @property
    def hop(self) -> int:
        """Passo di avanzamento della finestra tra due `read_block`, in campioni.

        Returns:
            L'hop (block_size // HOP_DIVISOR): il numero di campioni nuovi tra
            una finestra e la successiva. Il chiamante ne ricava il block rate
            del loop come `sample_rate / hop`.
        """
        return self._hop

    @property
    def device_display_name(self) -> str:
        """Nome del device loopback attualmente aperto (per lo stato del frontend).

        Returns:
            Il nome del device; stringa vuota prima dell'apertura.
        """
        return self._device_display_name

    @property
    def last_stereo(self) -> np.ndarray | None:
        """Ultimo blocco PRE-downmix come immagine stereo (percorso parallelo).

        Returns:
            L'ultimo blocco letto in forma `(n, 2)` float32 (primi due canali
            del device, pre-downmix), oppure None se non e' ancora arrivato
            nessun dato reale, se l'ultimo giro era silenzio sintetico o se il
            device e' mono.
        """
        return self._last_stereo

    def read_block(self) -> np.ndarray:
        """Legge una finestra scorrevole e la restituisce mono float32 normalizzata.

        La finestra e' larga `block_size` e AVANZA di `hop` campioni a ogni
        chiamata (overlap del 50%): la prima chiamata attende `block_size`
        campioni pieni, le successive solo `hop` campioni nuovi, ricomposti con
        la coda della finestra precedente. Cosi' il loop sale a ~2x fps a parita'
        di risoluzione FFT. I dati sono convertiti in float32 nel range [-1, 1]
        con downmix a mono (media dei canali). Se non arrivano entro
        `_SILENCE_TIMEOUT_SECONDS` (il loopback WASAPI non produce frame quando
        sull'output non suona nulla) restituisce un blocco di silenzio sintetico
        e riparte "non innescato": il chiamante continua a girare (e a mandare
        frame al Bridge) anche a musica ferma.

        Returns:
            Array numpy di shape `(block_size,)`, dtype float32, valori in
            [-1, 1]; tutti zeri se il device non ha prodotto dati in tempo.

        Raises:
            AudioCaptureError: Se lo stream non e' aperto.
        """
        if self._stream is None:
            raise AudioCaptureError(
                "Stream non aperto: usare LoopbackCapture come context manager."
            )

        # Cambio device richiesto dal frontend: consumato qui, nel thread del
        # loop, cosi' apertura/chiusura degli stream restano single-thread.
        pending = self._pending_device_name
        if pending is not None:
            self._pending_device_name = None
            self.device_name = pending
            self._reopen_if_changed()

        # Campioni nuovi da leggere: la prima finestra si riempie tutta, le
        # successive avanzano solo di hop (il resto viene dalla coda precedente).
        need = self.block_size if not self._primed else self._hop

        waited = 0.0
        while self._stream.get_read_available() < need and waited < _SILENCE_TIMEOUT_SECONDS:
            time.sleep(_AVAILABILITY_POLL_SECONDS)
            waited += _AVAILABILITY_POLL_SECONDS

        if self._stream.get_read_available() < need:
            # Il device tace da un po': forse l'output di default e' cambiato e
            # l'audio ora va altrove. Ricontrolla (solo in modalita' auto).
            self._synthetic_streak += 1
            if (
                not self.device_name
                and self._synthetic_streak >= _SILENT_BLOCKS_BEFORE_DEVICE_CHECK
            ):
                self._synthetic_streak = 0
                self._follow_default_device()
            # Nessun blocco reale pre-downmix in questo giro: azzera lo stereo
            # cosi' l'analisi parallela decade verso il centro sul silenzio. La
            # finestra riparte "non innescata": alla ripresa serve un blocco pieno.
            self._last_stereo = None
            self._primed = False
            return np.zeros(self.block_size, dtype=np.float32)

        self._synthetic_streak = 0
        raw = self._stream.read(need, exception_on_overflow=False)

        # int16 -> float32 normalizzato in [-1, 1] (i `need` campioni nuovi).
        new_samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / _INT16_FULL_SCALE

        if self._channels > 1:
            # Copia contigua (need, 2) dei primi due canali PRIMA del downmix.
            new_stereo: np.ndarray | None = np.ascontiguousarray(
                new_samples.reshape(-1, self._channels)[:, :2], dtype=np.float32
            )
            # Downmix a mono: media dei canali (layout interleaved).
            new_mono = new_samples.reshape(-1, self._channels).mean(axis=1).astype(np.float32)
        else:
            # Device mono: nessuna immagine stereo disponibile.
            new_stereo = None
            new_mono = new_samples

        window, window_stereo = self._advance_window(new_mono, new_stereo)
        # Stereo coerente con la finestra restituita (stessa porzione, pre-downmix).
        self._last_stereo = window_stereo
        return window

    def _advance_window(
        self, new_mono: np.ndarray, new_stereo: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compone la finestra scorrevole dai campioni nuovi appena letti.

        Al primo giro (non innescato) i campioni nuovi sono gia' una finestra
        piena; nei giri successivi la finestra e' la coda della precedente
        (scartati i primi hop campioni) piu' i `hop` campioni nuovi. Lo stereo
        segue esattamente la stessa finestra (pre-downmix).

        Args:
            new_mono: Campioni mono nuovi appena letti (`need` campioni).
            new_stereo: Loro versione pre-downmix `(need, 2)`, o None se il
                device e' mono.

        Returns:
            Coppia `(window, window_stereo)`: la finestra mono `(block_size,)` e
            la sua immagine stereo `(block_size, 2)` (o None su device mono).
        """
        hop = self._hop
        if not self._primed or self._prev_window is None:
            window = new_mono
            window_stereo = new_stereo
            self._primed = True
        else:
            window = np.concatenate([self._prev_window[hop:], new_mono])
            if new_stereo is not None and self._prev_window_stereo is not None:
                window_stereo = np.concatenate([self._prev_window_stereo[hop:], new_stereo], axis=0)
            else:
                # Il device e' passato a mono (o stereo appena comparso): niente
                # continuita' stereo, si usa direttamente la coda nuova.
                window_stereo = new_stereo
        self._prev_window = window
        self._prev_window_stereo = window_stereo
        return window, window_stereo

    def _follow_default_device(self) -> None:
        """Riapre la cattura sul nuovo output di default, se e' cambiato.

        Chiamato dopo un periodo prolungato di silenzio in modalita' auto
        (device_name vuoto).
        """
        self._reopen_if_changed()

    def _reopen_if_changed(self) -> None:
        """Riapre la cattura sul device risolto da `device_name`, se diverso.

        Usato sia dall'inseguimento dell'output di default sia dal cambio
        device richiesto dal frontend. PortAudio congela l'elenco dei device a init,
        quindi il ricontrollo usa una NUOVA istanza PyAudio: se il loopback di
        default risolto ha un nome diverso da quello aperto, adotta la nuova
        istanza e chiude la vecchia; altrimenti scarta la nuova. In caso di
        errore mantiene il device corrente (riprovera' al prossimo periodo di
        silenzio). Il sample rate puo' cambiare col device: il chiamante deve
        ricontrollare `sample_rate` dopo ogni lettura.
        """
        with _pa_lifecycle_lock:
            self._reopen_if_changed_locked()

    def _reopen_if_changed_locked(self) -> None:
        """Corpo di `_reopen_if_changed`, da eseguire sotto _pa_lifecycle_lock."""
        probe = pyaudio.PyAudio()
        try:
            device = self._resolve_device(probe)
        except AudioCaptureError as err:
            probe.terminate()
            logger.warning("Ricontrollo dell'output di default fallito: %s", err)
            return

        if device["name"] == self._device_display_name:
            probe.terminate()
            return

        logger.info(
            "Cattura spostata: da '%s' a '%s', riapro lo stream.",
            self._device_display_name,
            device["name"],
        )
        old_stream = self._stream
        old_pyaudio = self._pyaudio
        old_sample_rate = self._sample_rate
        old_channels = self._channels
        old_display_name = self._device_display_name
        try:
            self._open_stream(probe, device)
        except Exception:
            probe.terminate()
            # _open_stream sovrascrive rate/canali/nome prima di poter fallire:
            # ripristina TUTTO lo stato coerente col vecchio stream ancora
            # attivo (altrimenti il prossimo ricontrollo crederebbe di essere
            # gia' sul device nuovo e non riproverebbe mai).
            self._stream = old_stream
            self._sample_rate = old_sample_rate
            self._channels = old_channels
            self._device_display_name = old_display_name
            logger.exception("Riapertura sul nuovo device fallita: resto sul precedente.")
            return

        self._pyaudio = probe
        try:
            if old_stream is not None:
                old_stream.stop_stream()
                old_stream.close()
            if old_pyaudio is not None:
                old_pyaudio.terminate()
        except Exception:
            logger.exception("Errore chiudendo il vecchio device (proseguo sul nuovo).")

    def _resolve_device(self, audio: pyaudio.PyAudio) -> dict:
        """Risolve il device loopback da usare.

        Con `device_name` vuoto restituisce il loopback dell'output di default;
        altrimenti cerca per sottostringa case-insensitive tra i loopback.

        Args:
            audio: Istanza PyAudio gia' inizializzata.

        Returns:
            Il device info (dict) del loopback scelto.

        Raises:
            AudioCaptureError: Se WASAPI non e' disponibile o nessun loopback
                corrisponde al nome richiesto.
        """
        try:
            wasapi_info = audio.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError as err:
            raise AudioCaptureError("WASAPI non disponibile su questo sistema.") from err

        if not self.device_name:
            return self._default_loopback(audio, wasapi_info)

        return self._loopback_by_name(audio)

    def _default_loopback(self, audio: pyaudio.PyAudio, wasapi_info: dict) -> dict:
        """Restituisce il loopback associato all'output di default.

        Args:
            audio: Istanza PyAudio gia' inizializzata.
            wasapi_info: Host API info di WASAPI.

        Returns:
            Il device info del loopback dell'output di default.

        Raises:
            AudioCaptureError: Se non si trova il loopback dell'output di default.
        """
        default_speakers = audio.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        # L'output di default di solito non e' gia' un loopback: cerca il
        # loopback il cui nome contiene quello degli speaker di default.
        if not default_speakers["isLoopbackDevice"]:
            for loopback in audio.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    return loopback
            raise AudioCaptureError(
                f"Loopback per l'output di default '{default_speakers['name']}' non trovato. "
                f"Loopback disponibili: {self._available_loopback_names(audio)}"
            )

        return default_speakers

    def _loopback_by_name(self, audio: pyaudio.PyAudio) -> dict:
        """Cerca un loopback per sottostringa case-insensitive del nome.

        Args:
            audio: Istanza PyAudio gia' inizializzata.

        Returns:
            Il primo device info di loopback che matcha `device_name`.

        Raises:
            AudioCaptureError: Se nessun loopback contiene `device_name`.
        """
        needle = self.device_name.casefold()
        for loopback in audio.get_loopback_device_info_generator():
            if needle in loopback["name"].casefold():
                return loopback

        raise AudioCaptureError(
            f"Nessun device loopback contiene '{self.device_name}'. "
            f"Loopback disponibili: {self._available_loopback_names(audio)}"
        )

    @staticmethod
    def _available_loopback_names(audio: pyaudio.PyAudio) -> list[str]:
        """Elenca i nomi dei device loopback disponibili.

        Args:
            audio: Istanza PyAudio gia' inizializzata.

        Returns:
            Lista dei nomi dei loopback disponibili.
        """
        return [loopback["name"] for loopback in audio.get_loopback_device_info_generator()]

    def _cleanup(self) -> None:
        """Chiude stream e PyAudio se aperti, idempotente."""
        with _pa_lifecycle_lock:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
            if self._pyaudio is not None:
                self._pyaudio.terminate()
                self._pyaudio = None
