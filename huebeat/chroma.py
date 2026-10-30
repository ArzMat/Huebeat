"""Chromagram (profilo delle classi di altezza) per huebeat.

Modulo di puro calcolo (nessun logging, nessun effetto collaterale). A partire
dalle ampiezze rFFT finestrate gia' calcolate dall'analyzer (nessuna FFT nuova),
somma l'energia spettrale nelle 12 classi di altezza (chroma), ne liscia il
profilo con una EMA e ne ricava la tonalita' dominante e un colore (hue) mappato
sul circolo delle quinte, cosi' che tonalita' musicalmente vicine diano colori
vicini.

Interfaccia pubblica: `ChromaAnalyzer`.
"""

import numpy as np

# --- Numero di classi di altezza (semitoni per ottava) ---
NUM_PITCH_CLASSES = 12

# --- Riferimento di accordatura (Hz) ---
# A4 = 440 Hz. La classe di altezza di un bin e' round(12*log2(freq/440)) piu'
# un offset di semitoni che porta l'indicizzazione alla convenzione MIR standard
# C=0..B=11: A dista 9 semitoni sopra C, quindi +9 mette A a classe 9 (LA). La
# formula "grezza" round(12*log2(freq/440)) da sola metterebbe A a 0; il +9
# allinea gli indici alla convenzione chroma usuale (e ai criteri di verifica).
TUNING_REFERENCE_HZ = 440.0
PITCH_CLASS_OFFSET = 9

# --- Banda di frequenza considerata per il chroma (Hz) ---
# Sotto ~80 Hz il rombo dei bassi/kick non ha altezza definita utile e la
# risoluzione in frequenza di un blocco singolo e' troppo grezza; sopra ~5000 Hz
# domina il rumore percussivo/aria. Estremi inclusivi sui bin rFFT.
CHROMA_LOW_HZ = 80.0
CHROMA_HIGH_HZ = 5000.0

# --- Smoothing EMA del profilo chroma ---
# Tempo caratteristico (secondi): la tonalita' e' un carattere della sezione, non
# insegue i transienti. ~0.7 s stabilizza la classe dominante attraverso arpeggi
# e passaggi di accordo senza inerzia eccessiva. Convertito in coefficiente per
# blocco in __init__ dalla durata del blocco (hop/sample_rate), invariante al
# sample rate e all'hop.
CHROMA_SMOOTHING_SECONDS = 0.7

# --- Soglie di validita' della stima ---
# Energia chroma totale minima (somma delle 12 classi lisciate): sotto questa il
# blocco e' silenzio/quasi e la tonalita' non e' definita -> hue None, forza 0.
CHROMA_ENERGY_FLOOR = 1e-4
# Dominanza minima (quanto la classe piu' forte emerge dalla media delle 12,
# in [0, 1]): sotto questa il profilo e' piatto (rumore, percussioni, mix senza
# tonalita' chiara) e non si dichiara una tonalita' -> hue None. La forza
# (harmony_strength) resta il valore di dominanza anche sotto soglia.
CHROMA_MIN_DOMINANCE = 0.2
# Pavimento numerico del denominatore della dominanza.
DOMINANCE_EPS = 1e-12

# --- Circolo delle quinte ---
# Passo in semitoni tra tonalita' adiacenti sul circolo delle quinte: 7
# semitoni = una quinta giusta. hue = ((pc*7) mod 12) / 12 dispone le tonalita'
# vicine per quinte a colori vicini sulla ruota (DO, SOL, RE... a hue crescente).
FIFTH_STEP_SEMITONES = 7


class ChromaAnalyzer:
    """Stima classe di altezza dominante e hue tonale da ampiezze rFFT a blocchi.

    Per ogni blocco somma le ampiezze rFFT finestrate (fornite dall'analyzer,
    nessuna FFT nuova) nelle 12 classi di altezza, liscia il profilo con una EMA
    e ricava:

    - la classe dominante (argmax del profilo lisciato);
    - la sua dominanza in [0, 1] (quanto emerge dalla media delle 12);
    - un hue in [0, 1) mappato sul circolo delle quinte.

    Sotto una soglia di energia o di dominanza la tonalita' non e' definita e il
    hue e' None (forza comunque riportata).

    Attributes:
        sample_rate: Frequenza di campionamento in Hz.
        block_size: Numero di campioni per blocco (larghezza della finestra FFT).
    """

    def __init__(self, sample_rate: int, block_size: int, hop: int | None = None) -> None:
        """Inizializza l'analizzatore e precomputa la mappa bin -> classe.

        Args:
            sample_rate: Frequenza di campionamento del segnale in Hz.
            block_size: Numero di campioni per blocco (larghezza della finestra
                FFT), pari alla lunghezza dei blocchi passati all'analyzer.
            hop: Passo di avanzamento tra due blocchi consecutivi, in campioni.
                None (default) equivale a `block_size` (blocchi disgiunti). Serve
                solo a convertire il tempo dell'EMA in coefficiente per blocco:
                la durata del blocco e' `hop/sample_rate`, cosi' lo smoothing
                resta invariante in secondi reali anche con finestre sovrapposte.
        """
        self.sample_rate: int = sample_rate
        self.block_size: int = block_size
        hop_effettivo: int = block_size if hop is None else hop
        block_duration: float = hop_effettivo / sample_rate

        # Frequenza di ogni bin rFFT e maschera della banda considerata.
        freqs: np.ndarray = np.fft.rfftfreq(block_size, d=1.0 / sample_rate)
        band_mask: np.ndarray = (freqs >= CHROMA_LOW_HZ) & (freqs <= CHROMA_HIGH_HZ)
        self._band_mask: np.ndarray = band_mask

        # Classe di altezza di ogni bin in banda (convenzione C=0..B=11). I bin a
        # 0 Hz o comunque fuori banda sono esclusi dalla maschera, quindi il log2
        # opera solo su frequenze positive.
        band_freqs: np.ndarray = freqs[band_mask]
        semitones: np.ndarray = np.round(
            NUM_PITCH_CLASSES * np.log2(band_freqs / TUNING_REFERENCE_HZ)
        ).astype(np.int64)
        self._pitch_class: np.ndarray = (semitones + PITCH_CLASS_OFFSET) % NUM_PITCH_CLASSES

        # Profilo chroma lisciato (EMA) e coefficiente per blocco.
        self._chroma: np.ndarray = np.zeros(NUM_PITCH_CLASSES, dtype=np.float64)
        self._smooth_coeff: float = 1.0 - float(np.exp(-block_duration / CHROMA_SMOOTHING_SECONDS))

    def process(self, magnitudes: np.ndarray) -> tuple[float | None, float]:
        """Aggiorna il profilo chroma e restituisce (hue, forza) del blocco.

        Args:
            magnitudes: Ampiezze rFFT finestrate (Hann) del blocco corrente, gia'
                calcolate dall'analyzer con lo STESSO block_size
                (`np.abs(rfft(block * hann))`, lunghezza `block_size//2 + 1`).

        Returns:
            Coppia `(hue, strength)`: `hue` in [0, 1) sul circolo delle quinte
            della classe dominante, oppure None se la tonalita' non e' definita
            (energia o dominanza sotto soglia); `strength` in [0, 1] e' la
            dominanza lisciata della classe piu' forte (0.0 sul silenzio).
        """
        # Somma delle ampiezze per classe di altezza sui soli bin in banda, poi
        # EMA del profilo. bincount pesato = una passata sui bin mascherati.
        band_mags: np.ndarray = magnitudes[self._band_mask]
        frame_chroma: np.ndarray = np.bincount(
            self._pitch_class,
            weights=band_mags,
            minlength=NUM_PITCH_CLASSES,
        )
        self._chroma += (frame_chroma - self._chroma) * self._smooth_coeff

        total: float = float(np.sum(self._chroma))
        if total < CHROMA_ENERGY_FLOOR:
            return None, 0.0

        dominant_class: int = int(np.argmax(self._chroma))
        dominant_value: float = float(self._chroma[dominant_class])
        mean_value: float = total / NUM_PITCH_CLASSES
        # Dominanza: quanto la classe piu' forte emerge dalla media, in [0, 1).
        # 0 = profilo piatto (nessuna tonalita'), ->1 = una classe concentra
        # tutta l'energia.
        strength: float = (dominant_value - mean_value) / (dominant_value + DOMINANCE_EPS)
        strength = min(max(strength, 0.0), 1.0)

        if strength < CHROMA_MIN_DOMINANCE:
            return None, strength

        # Hue sul circolo delle quinte: tonalita' a distanza di quinta -> colori
        # adiacenti sulla ruota.
        hue: float = (
            (dominant_class * FIFTH_STEP_SEMITONES) % NUM_PITCH_CLASSES
        ) / NUM_PITCH_CLASSES
        return hue, strength
