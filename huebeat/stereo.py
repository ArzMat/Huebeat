"""Immagine stereo (panning per banda) come percorso PARALLELO al mono.

Modulo di puro calcolo (nessun logging, nessun effetto collaterale). Riceve
l'ultimo blocco stereo PRE-downmix (shape `(n, 2)` float32) prodotto dalle
catture audio e ne estrae il panning per banda: dove sta, sull'asse
sinistra-destra, l'energia dei bassi, dei medi e degli alti.

Il percorso mono di `huebeat.analysis` resta intatto: questo e' un binario
aggiuntivo. Le bande di frequenza sono le STESSE dell'analisi mono (importate
da `huebeat.analysis`) e la finestra e' la stessa di Hann, cosi' i due percorsi
guardano esattamente le stesse porzioni di spettro.

Interfaccia pubblica: `StereoAnalyzer`, che restituisce uno `StereoImage`.
"""

from dataclasses import dataclass

import numpy as np

from huebeat.analysis import (
    BASS_HIGH_HZ,
    BASS_LOW_HZ,
    HIGH_HIGH_HZ,
    HIGH_LOW_HZ,
    MID_HIGH_HZ,
    MID_LOW_HZ,
)

# --- Smoothing asimmetrico del pan (per banda) ---
# Tempo caratteristico dell'attacco (secondi): reagisce in fretta a un nuovo
# panning marcato (|pan| che cresce).
PAN_ATTACK_SECONDS = 0.1
# Tempo caratteristico del rilascio (secondi): rientro dolce verso il centro
# (|pan| che cala, silenzio, banda non significativa).
PAN_RELEASE_SECONDS = 0.4

# Epsilon della divisione del pan: evita 0/0 quando una banda tace su entrambi
# i canali.
PAN_EPSILON = 1e-9

# Somma di energia (|X|^2 di L+R su tutte le bande) sotto cui il blocco e'
# silenzio: tutti i pan decadono verso 0 (ed evita la divisione per zero nel
# gate a quota). Volutamente bassissima: cattura solo il silenzio vero, la
# significativita' di banda la decide il gate a quota qui sotto.
STEREO_SILENCE_FLOOR = 1e-6

# Quota minima dell'energia di una banda sul totale del blocco perche' il suo
# pan sia considerato significativo. Sotto, la banda porta solo leakage
# spettrale o rumore e il suo pan decade verso 0 (niente pan casuale su bande
# quasi vuote). E' una quota (scala-invariante), non un livello assoluto:
# misurato, un tono puro lascia nelle bande adiacenti < 0.001 del totale,
# quindi 0.02 ha margine abbondante mantenendo vive le bande reali.
BAND_ENERGY_SHARE_MIN = 0.02


@dataclass(frozen=True)
class StereoImage:
    """Panning per banda dell'immagine stereo corrente.

    Ogni valore e' in [-1, 1]: -1 tutto a sinistra, +1 tutto a destra, 0
    centrato (mono, silenzio o banda non significativa).

    Attributes:
        pan_bass: Pan della banda bassi (`BASS_LOW_HZ`..`BASS_HIGH_HZ`).
        pan_mid: Pan della banda medi (`MID_LOW_HZ`..`MID_HIGH_HZ`).
        pan_high: Pan della banda alti (`HIGH_LOW_HZ`..`HIGH_HIGH_HZ`).
    """

    pan_bass: float
    pan_mid: float
    pan_high: float


class StereoAnalyzer:
    """Estrae il panning per banda da blocchi stereo pre-downmix.

    Per ogni blocco applica la finestra di Hann e la rFFT sui due canali,
    misura l'energia per banda su ciascun canale e ne ricava il pan come
    squilibrio normalizzato `(E_r - E_l) / (E_r + E_l)`. Il pan e' poi lisciato
    con un envelope asimmetrico per banda (attacco veloce, rilascio lento); le
    bande quasi vuote e il silenzio decadono verso il centro.

    Attributes:
        sample_rate: Frequenza di campionamento in Hz.
        block_size: Numero di campioni attesi per blocco (per canale).
    """

    def __init__(self, sample_rate: int, block_size: int) -> None:
        """Inizializza l'analizzatore e precomputa finestra e maschere di banda.

        Args:
            sample_rate: Frequenza di campionamento del segnale in Hz.
            block_size: Numero di campioni per canale attesi in ogni blocco
                passato a `process`.
        """
        self.sample_rate: int = sample_rate
        self.block_size: int = block_size

        # Finestra di Hann e maschere di banda IDENTICHE a quelle dell'analisi
        # mono (huebeat.analysis): stessi bordi di banda, stessa finestra.
        self._window: np.ndarray = np.hanning(block_size).astype(np.float32)
        freqs: np.ndarray = np.fft.rfftfreq(block_size, d=1.0 / sample_rate)
        self._bass_mask: np.ndarray = (freqs >= BASS_LOW_HZ) & (freqs <= BASS_HIGH_HZ)
        self._mid_mask: np.ndarray = (freqs >= MID_LOW_HZ) & (freqs <= MID_HIGH_HZ)
        self._high_mask: np.ndarray = (freqs >= HIGH_LOW_HZ) & (freqs <= HIGH_HIGH_HZ)

        # Coefficienti one-pole dell'envelope asimmetrico (attacco/rilascio).
        block_duration: float = block_size / sample_rate
        self._attack: float = 1.0 - float(np.exp(-block_duration / PAN_ATTACK_SECONDS))
        self._release: float = 1.0 - float(np.exp(-block_duration / PAN_RELEASE_SECONDS))

        # Stato del pan per banda (lisciato).
        self._pan_bass: float = 0.0
        self._pan_mid: float = 0.0
        self._pan_high: float = 0.0

    def process(self, stereo_block: np.ndarray | None) -> StereoImage:
        """Aggiorna il panning per banda con il blocco stereo corrente.

        Args:
            stereo_block: Blocco stereo `(n, 2)` float32 PRE-downmix (colonna 0
                = sinistra, colonna 1 = destra), oppure None se non c'e' dato
                stereo reale (silenzio o sorgente mono).

        Returns:
            Lo `StereoImage` con il pan lisciato di ogni banda in [-1, 1].
        """
        if stereo_block is None:
            # Nessun dato stereo: rientro verso il centro col rilascio lento.
            self._pan_bass = self._decay(self._pan_bass)
            self._pan_mid = self._decay(self._pan_mid)
            self._pan_high = self._decay(self._pan_high)
            return self._image()

        # rFFT finestrata (Hann) per canale, energia = |X|^2.
        left: np.ndarray = stereo_block[:, 0] * self._window
        right: np.ndarray = stereo_block[:, 1] * self._window
        power_l: np.ndarray = np.abs(np.fft.rfft(left)) ** 2
        power_r: np.ndarray = np.abs(np.fft.rfft(right)) ** 2

        el_bass: float = float(np.sum(power_l[self._bass_mask]))
        er_bass: float = float(np.sum(power_r[self._bass_mask]))
        el_mid: float = float(np.sum(power_l[self._mid_mask]))
        er_mid: float = float(np.sum(power_r[self._mid_mask]))
        el_high: float = float(np.sum(power_l[self._high_mask]))
        er_high: float = float(np.sum(power_r[self._high_mask]))

        total: float = el_bass + er_bass + el_mid + er_mid + el_high + er_high

        self._pan_bass = self._update(self._pan_bass, el_bass, er_bass, total)
        self._pan_mid = self._update(self._pan_mid, el_mid, er_mid, total)
        self._pan_high = self._update(self._pan_high, el_high, er_high, total)
        return self._image()

    def _update(self, current: float, energy_l: float, energy_r: float, total: float) -> float:
        """Calcola il pan target di una banda e lo liscia con l'envelope.

        Il target e' lo squilibrio di energia `(E_r - E_l) / (E_r + E_l)`, ma
        decade verso 0 se il blocco e' silenzio (energia totale sotto
        `STEREO_SILENCE_FLOOR`) o se la banda porta troppo poca energia rispetto
        al totale (`BAND_ENERGY_SHARE_MIN`). Lo smoothing e' asimmetrico:
        attacco veloce se il pan cresce in modulo, rilascio lento se cala.

        Args:
            current: Pan lisciato corrente della banda in [-1, 1].
            energy_l: Energia della banda sul canale sinistro.
            energy_r: Energia della banda sul canale destro.
            total: Energia totale del blocco (tutte le bande, entrambi i canali).

        Returns:
            Il nuovo pan lisciato della banda.
        """
        band_energy: float = energy_l + energy_r
        if total < STEREO_SILENCE_FLOOR or band_energy < BAND_ENERGY_SHARE_MIN * total:
            target: float = 0.0
        else:
            target = (energy_r - energy_l) / (energy_r + energy_l + PAN_EPSILON)

        coeff: float = self._attack if abs(target) > abs(current) else self._release
        return current + (target - current) * coeff

    def _decay(self, current: float) -> float:
        """Fa decadere un pan verso il centro col rilascio lento.

        Args:
            current: Pan lisciato corrente della banda.

        Returns:
            Il pan avvicinato a 0 di un passo di rilascio.
        """
        return current + (0.0 - current) * self._release

    def _image(self) -> StereoImage:
        """Costruisce lo `StereoImage` dallo stato corrente, clampato in [-1, 1].

        Returns:
            Lo snapshot immutabile del pan per banda.
        """
        return StereoImage(
            pan_bass=float(np.clip(self._pan_bass, -1.0, 1.0)),
            pan_mid=float(np.clip(self._pan_mid, -1.0, 1.0)),
            pan_high=float(np.clip(self._pan_high, -1.0, 1.0)),
        )
