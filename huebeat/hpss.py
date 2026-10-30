"""HPSS lite (separazione armonico/percussivo) per huebeat.

Modulo di puro calcolo (nessun logging, nessun effetto collaterale). Separa il
contenuto ARMONICO (toni tenuti: voci, synth, corde) da quello PERCUSSIVO
(transienti: cassa, rullante, hi-hat) con la tecnica a mediane di Fitzgerald,
riusando le ampiezze rFFT finestrate gia' calcolate dall'analyzer (nessuna FFT
nuova). Restituisce due livelli continui in [0, 1] pronti a pilotare le luci.

Idea: l'energia armonica e' liscia nel TEMPO (una nota tenuta e' costante frame
dopo frame) ma stretta in FREQUENZA; quella percussiva e' liscia in FREQUENZA
(un colpo eccita tutto lo spettro) ma stretta nel TEMPO. Quindi:

- stima armonica per bin = mediana nel tempo sulla storia recente delle frame;
- stima percussiva per bin = mediana in frequenza della frame corrente (filtro
  a mediana scorrevole in numpy: il progetto e' numpy-only, niente scipy);
- maschere soft (Wiener) dalle due stime, energie = somma(ampiezza * maschera);
- ogni energia normalizzata da un AGC a inseguimento di picco con lo STESSO
  schema degli envelope di banda dell'analyzer (attack/release asimmetrico,
  pavimento incrociato sul picco condiviso).

Interfaccia pubblica: `HpssAnalyzer`.
"""

from collections import deque

import numpy as np

# --- Storia temporale per la mediana armonica ---
# Numero di frame di ampiezze tenute per la mediana nel tempo. 9 e' un buon
# compromesso: abbastanza da distinguere un tono tenuto da un transiente,
# abbastanza corto da non introdurre latenza percepibile (a ~94 Hz sono ~96 ms).
# Dispari cosi' la mediana e' un campione centrale.
HPSS_HISTORY_FRAMES = 9

# --- Kernel della mediana in frequenza per la stima percussiva ---
# Ampiezza (in bin) della mediana lungo l'asse frequenza sulla frame corrente:
# un contenuto percussivo (largo in frequenza) sopravvive alla mediana, un tono
# stretto (pochi bin) viene soppresso. ~17 bin. Dispari (richiesto da medfilt).
HPSS_FREQ_KERNEL = 17

# --- Maschere soft (Wiener) ---
# Pavimento numerico nel denominatore delle maschere h^2/(h^2+p^2+eps): evita
# 0/0 dove sia stima armonica sia percussiva sono nulle (silenzio, bordi banda).
HPSS_MASK_EPS = 1e-12

# --- Envelope AGC (stesso schema degli envelope di banda in analysis.py) ---
# Valori IDENTICI alle costanti di banda dell'analyzer (ATTACK=0.9, RELEASE=0.25,
# PEAK_DECAY_SECONDS=3.0, PEAK_FLOOR=1e-6, BAND_PEAK_FLOOR_SHARE=0.1); ridefiniti
# qui, non importati, per non creare una dipendenza circolare analysis<->hpss.
# Le costanti di banda dell'analyzer restano intoccate: queste sono istanze e
# stato nuovi con lo stesso schema.
HPSS_ATTACK = 0.9
HPSS_RELEASE = 0.25
HPSS_PEAK_DECAY_SECONDS = 3.0
HPSS_PEAK_FLOOR = 1e-6
# Pavimento incrociato: il denominatore di ciascuna energia non scende sotto
# questa frazione del picco CONDIVISO (max delle due energie). Come per le bande,
# impedisce all'energia quasi vuota (es. il percussivo su un tono tenuto puro) di
# auto-normalizzarsi a saturazione: resta bassa finche' non emerge davvero.
HPSS_PEAK_FLOOR_SHARE = 0.1

# --- Clamp dell'output ---
HPSS_OUTPUT_MIN = 0.0
HPSS_OUTPUT_MAX = 1.0


def _median_filter_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Filtro a mediana scorrevole 1D (semantica di scipy.signal.medfilt).

    Il progetto e' numpy-only: si replica la mediana in frequenza con una vista
    a finestra scorrevole su un array zero-paddato di `kernel_size // 2` per lato
    (stesso trattamento dei bordi di `medfilt`), evitando la dipendenza scipy.

    Args:
        values: Array 1D da filtrare (ampiezze per bin).
        kernel_size: Ampiezza della finestra, dispari.

    Returns:
        Array della stessa lunghezza di `values`, mediana su ogni finestra.
    """
    pad: int = kernel_size // 2
    padded: np.ndarray = np.pad(values, pad, mode="constant")
    windows: np.ndarray = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
    return np.median(windows, axis=1)


class _BandEnvelope:
    """AGC a picco + envelope asimmetrico per una singola energia.

    Replica lo schema degli envelope di banda di `FeatureAnalyzer`: insegue il
    proprio picco (decadimento lento), normalizza sul picco col pavimento
    incrociato sul picco condiviso, poi liscia con attack rapido / release piu'
    dolce. Stato indipendente per istanza.
    """

    def __init__(self, peak_decay_per_block: float) -> None:
        """Inizializza lo stato dell'envelope.

        Args:
            peak_decay_per_block: Fattore di decadimento del picco per blocco
                (exp(-durata_blocco / HPSS_PEAK_DECAY_SECONDS)).
        """
        self._peak: float = HPSS_PEAK_FLOOR
        self._out: float = HPSS_OUTPUT_MIN
        self._peak_decay: float = peak_decay_per_block

    def process(self, energy: float, shared_peak: float) -> float:
        """Normalizza e liscia l'energia del blocco corrente.

        Args:
            energy: Energia grezza dell'una o dell'altra componente nel blocco.
            shared_peak: Picco condiviso (max delle due energie) per il pavimento
                incrociato del denominatore.

        Returns:
            Livello normalizzato e lisciato in [0, 1].
        """
        self._peak = max(self._peak * self._peak_decay, energy, HPSS_PEAK_FLOOR)
        denom: float = max(self._peak, HPSS_PEAK_FLOOR_SHARE * shared_peak)
        normalized: float = energy / denom
        coeff: float = HPSS_ATTACK if normalized > self._out else HPSS_RELEASE
        self._out += (normalized - self._out) * coeff
        return float(min(max(self._out, HPSS_OUTPUT_MIN), HPSS_OUTPUT_MAX))


class HpssAnalyzer:
    """Separa energia armonica e percussiva da ampiezze rFFT a blocchi.

    Per ogni blocco tiene una storia delle ultime `HPSS_HISTORY_FRAMES` frame di
    ampiezze; stima armonica per bin = mediana nel tempo, percussiva = mediana in
    frequenza della frame corrente; ne ricava maschere soft e due energie,
    normalizzate ciascuna da un AGC con lo schema degli envelope di banda.

    Attributes:
        sample_rate: Frequenza di campionamento in Hz.
        block_size: Numero di campioni per blocco (larghezza della finestra FFT).
    """

    def __init__(self, sample_rate: int, block_size: int, hop: int | None = None) -> None:
        """Inizializza l'analizzatore e gli envelope AGC.

        Args:
            sample_rate: Frequenza di campionamento del segnale in Hz.
            block_size: Numero di campioni per blocco (larghezza della finestra
                FFT), pari alla lunghezza dei blocchi passati all'analyzer.
            hop: Passo di avanzamento tra due blocchi consecutivi, in campioni.
                None (default) equivale a `block_size` (blocchi disgiunti). Serve
                solo a convertire il tempo di decadimento del picco AGC in
                fattore per blocco: la durata del blocco e' `hop/sample_rate`,
                cosi' l'AGC resta invariante in secondi reali con l'overlap.
        """
        self.sample_rate: int = sample_rate
        self.block_size: int = block_size
        hop_effettivo: int = block_size if hop is None else hop
        block_duration: float = hop_effettivo / sample_rate

        # Storia temporale delle ampiezze (per la mediana armonica nel tempo).
        self._history: deque[np.ndarray] = deque(maxlen=HPSS_HISTORY_FRAMES)

        # Envelope AGC delle due componenti + picco condiviso (pavimento
        # incrociato), stesso schema delle bande dell'analyzer.
        peak_decay: float = float(np.exp(-block_duration / HPSS_PEAK_DECAY_SECONDS))
        self._harmonic_env = _BandEnvelope(peak_decay)
        self._percussive_env = _BandEnvelope(peak_decay)
        self._shared_peak: float = HPSS_PEAK_FLOOR
        self._peak_decay: float = peak_decay

    def process(self, magnitudes: np.ndarray) -> tuple[float, float]:
        """Aggiorna la separazione e restituisce (percussive_level, harmonic_level).

        Args:
            magnitudes: Ampiezze rFFT finestrate (Hann) del blocco corrente, gia'
                calcolate dall'analyzer con lo STESSO block_size
                (`np.abs(rfft(block * hann))`, lunghezza `block_size//2 + 1`).

        Returns:
            Coppia `(percussive_level, harmonic_level)` in [0, 1]: entrambi 0.0
            sul silenzio; sulle prime frame (storia parziale) la mediana nel
            tempo usa le frame disponibili, quindi il comportamento resta
            sensato senza casi degeneri.
        """
        self._history.append(magnitudes)

        # Stima armonica: mediana nel tempo sulla storia (per bin). Con storia
        # parziale la mediana usa le frame presenti.
        harmonic_est: np.ndarray = np.median(np.stack(self._history), axis=0)
        # Stima percussiva: mediana in frequenza della frame corrente (kernel
        # dispari; zero-padding ai bordi banda, come medfilt).
        percussive_est: np.ndarray = _median_filter_1d(magnitudes, HPSS_FREQ_KERNEL)

        # Maschere soft (Wiener) e energie = somma(ampiezza * maschera).
        h_sq: np.ndarray = harmonic_est**2
        p_sq: np.ndarray = percussive_est**2
        denom: np.ndarray = h_sq + p_sq + HPSS_MASK_EPS
        mask_h: np.ndarray = h_sq / denom
        mask_p: np.ndarray = p_sq / denom
        harmonic_energy: float = float(np.sum(magnitudes * mask_h))
        percussive_energy: float = float(np.sum(magnitudes * mask_p))

        # Picco condiviso (pavimento incrociato) e normalizzazione AGC.
        self._shared_peak = max(
            self._shared_peak * self._peak_decay,
            harmonic_energy,
            percussive_energy,
            HPSS_PEAK_FLOOR,
        )
        harmonic_level: float = self._harmonic_env.process(harmonic_energy, self._shared_peak)
        percussive_level: float = self._percussive_env.process(percussive_energy, self._shared_peak)
        return percussive_level, harmonic_level
