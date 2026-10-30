"""Sorgente audio commutabile a caldo: device intero oppure singola app.

Il loop di rendering parla con UN oggetto (`AudioSource`) che incapsula la
cattura corrente: `LoopbackCapture` (loopback WASAPI di un'uscita intera) o
`ProcessCapture` (solo l'audio di un processo, via helper nativo Process
Loopback). Il cambio di sorgente arriva dal frontend e viene applicato NEL
thread del loop (`swap`), cosi' apertura e chiusura delle catture restano
single-thread; il chiamante deve ricontrollare `sample_rate` dopo ogni swap
(il device puo' avere rate nativi diversi, il per-processo e' fisso a 48 kHz).
"""

import logging
from typing import TYPE_CHECKING

from huebeat.audio import LoopbackCapture
from huebeat.process_audio import ProcessCapture, ProcessCaptureError

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Tipi di sorgente accettati (config `audio_source` e POST /api/source).
SOURCE_DEVICE = "device"
SOURCE_PROCESS = "process"
SOURCE_TYPES = (SOURCE_DEVICE, SOURCE_PROCESS)


class AudioSource:
    """Facciata della cattura audio con cambio di sorgente a caldo.

    Espone la stessa superficie usata dal loop (`read_block`, `sample_rate`,
    `device_display_name`, `request_target`) piu' `swap` per cambiare TIPO di
    sorgente. Da usare come context manager; tutte le operazioni vanno fatte
    dal thread del loop di rendering.
    """

    def __init__(self, source_type: str, target: str, block_size: int) -> None:
        """Prepara la sorgente senza aprirla.

        Args:
            source_type: "device" (loopback di un'uscita) o "process" (app).
            target: Nome del device (vuoto = default) o dell'eseguibile.
            block_size: Campioni per blocco di analisi.
        """
        self._source_type = source_type
        self._target = target
        self._block_size = block_size
        self._capture: LoopbackCapture | ProcessCapture | None = None

    def __enter__(self) -> "AudioSource":
        """Apre la cattura iniziale (con fallback al device di default).

        Se la sorgente per-processo non e' apribile (helper nativo mancante),
        degrada al loopback del device di default con un warning: il motore
        deve partire comunque.

        Returns:
            L'istanza stessa, pronta alla lettura.
        """
        try:
            self._capture = self._open(self._source_type, self._target)
        except ProcessCaptureError as err:
            logger.warning("Sorgente per-processo non disponibile (%s): uso il device.", err)
            self._source_type = SOURCE_DEVICE
            self._target = ""
            self._capture = self._open(SOURCE_DEVICE, "")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Chiude la cattura corrente."""
        if self._capture is not None:
            self._capture.__exit__(exc_type, exc, tb)
            self._capture = None

    def _open(self, source_type: str, target: str):
        """Apre e restituisce la cattura del tipo richiesto.

        Args:
            source_type: "device" o "process".
            target: Nome device o eseguibile.

        Returns:
            La cattura aperta (context manager gia' entrato).
        """
        if source_type == SOURCE_PROCESS:
            capture = ProcessCapture(target, self._block_size)
        else:
            capture = LoopbackCapture(target, self._block_size)
        capture.__enter__()
        return capture

    @property
    def source_type(self) -> str:
        """Tipo di sorgente corrente ("device" o "process")."""
        return self._source_type

    @property
    def target(self) -> str:
        """Nome grezzo del target corrente (device o eseguibile; "" = default)."""
        return self._target

    @property
    def sample_rate(self) -> int:
        """Sample rate della cattura corrente."""
        return self._capture.sample_rate  # type: ignore[union-attr]

    @property
    def device_display_name(self) -> str:
        """Nome leggibile della sorgente corrente (per lo stato del frontend)."""
        return self._capture.device_display_name  # type: ignore[union-attr]

    @property
    def last_stereo(self) -> "np.ndarray | None":
        """Ultimo blocco stereo PRE-downmix della cattura corrente.

        Percorso PARALLELO al mono: delega alla cattura attiva.

        Returns:
            L'ultimo blocco `(n, 2)` float32 della cattura corrente, oppure
            None se non c'e' dato reale recente o la sorgente e' mono.
        """
        return self._capture.last_stereo  # type: ignore[union-attr]

    def read_block(self):
        """Legge un blocco mono float32 dalla cattura corrente."""
        return self._capture.read_block()  # type: ignore[union-attr]

    def request_target(self, target: str) -> None:
        """Ritarget A CALDO dentro lo stesso tipo di sorgente.

        Args:
            target: Nuovo device (vuoto = default) o nuovo eseguibile.
        """
        self._target = target
        self._capture.request_device(target)  # type: ignore[union-attr]

    def swap(self, source_type: str, target: str) -> None:
        """Cambia sorgente (anche di tipo), da chiamare NEL thread del loop.

        A parita' di tipo delega al ritarget a caldo; al cambio di tipo chiude
        la cattura corrente e apre la nuova. Se la nuova non si apre, riapre
        il device di default: il loop non resta mai senza sorgente.

        Args:
            source_type: "device" o "process".
            target: Nome device o eseguibile.
        """
        if source_type == self._source_type:
            if target != self._target:
                self.request_target(target)
            return

        logger.info("Cambio sorgente audio: %s '%s'.", source_type, target or "(default)")
        if self._capture is not None:
            self._capture.__exit__(None, None, None)
            self._capture = None
        try:
            self._capture = self._open(source_type, target)
            self._source_type = source_type
            self._target = target
        except Exception:
            logger.exception("Apertura della nuova sorgente fallita: torno al device di default.")
            self._capture = self._open(SOURCE_DEVICE, "")
            self._source_type = SOURCE_DEVICE
            self._target = ""
