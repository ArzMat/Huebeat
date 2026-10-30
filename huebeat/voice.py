"""Rilevamento euristico della presenza vocale per huebeat.

Modulo di puro calcolo (nessun logging, nessun effetto collaterale). Riceve
blocchi audio mono float32 e stima un livello continuo in [0, 1] di presenza
di voce cantata nel mix, pronto a pilotare le luci dedicate alla voce.

Euristica tier 1 dichiaratamente imperfetta: energia della banda vocale
(300-3000 Hz) normalizzata da un AGC a picco, moltiplicata per l'armonicita'
(periodicita' con fondamentale nel range vocale 80-400 Hz, stimata via
autocorrelazione del segnale filtrato in banda), lisciata con un envelope
attack/release asimmetrico.

Interfaccia pubblica: `VoiceDetector`.
"""

from collections import deque

import numpy as np

# --- Banda vocale (Hz) ---
# Dove vive il grosso dell'energia del canto: fondamentali alte, prime
# armoniche e formanti. Estremi inclusivi sui bin rFFT.
VOICE_LOW_HZ = 300.0
VOICE_HIGH_HZ = 3000.0

# --- Range della fondamentale vocale (Hz) ---
# Il picco di autocorrelazione deve cadere in un lag compatibile con una f0
# cantata: sotto 80 Hz e' un basso/kick, sopra 400 Hz e' un tono/lead acuto.
F0_MIN_HZ = 80.0
F0_MAX_HZ = 400.0

# --- Reiezione dei sub-multipli ---
# L'autocorrelazione di un segnale periodico ha picchi anche ai lag multipli
# del periodo vero: un tono puro a 1 kHz (f0 fuori range vocale) produce
# picchi "vocali" spuri a 2, 3, 4 volte il suo periodo. Si cerca quindi anche
# il miglior picco ai lag PIU' CORTI del range vocale (f0 tra F0_MAX_HZ e
# questo tetto, pari al limite alto della banda vocale): se e' forte quanto
# il picco vocale, la vera fondamentale sta sopra il range e il candidato
# vocale e' solo un sub-multiplo. L'armonicita' grezza e' la differenza
# (clampata a 0) tra picco vocale e picco sub-multiplo.
SUB_F0_MAX_HZ = 3000.0
# Lag minimo assoluto per la ricerca dei sub-multipli (campioni): esclude i
# primissimi lag, dominati dalla correlazione banale intorno allo zero.
SUB_LAG_MIN_SAMPLES = 2

# --- AGC a inseguimento di picco (stessa logica di analysis.py) ---
# Tempo perche' il picco memorizzato scenda a ~37% (1/e) senza segnale.
PEAK_DECAY_SECONDS = 3.0
# Pavimento del picco: evita divisioni per zero e impedisce all'AGC di
# amplificare il rumore di fondo fino a saturare.
PEAK_FLOOR = 1e-6
# Pavimento incrociato: il denominatore della banda vocale non scende sotto
# questa frazione del picco RMS dell'intero spettro. Senza, una banda vocale
# quasi vuota (solo leakage di un kick o di un hi-hat) si
# auto-normalizzerebbe a saturazione, come gia' osservato per le bande in
# analysis.py.
TOTAL_PEAK_FLOOR_SHARE = 0.2

# --- Buffer per l'autocorrelazione ---
# La periodicita' si stima su una finestra piu' lunga del singolo blocco: il
# buffer copre almeno ACF_BUFFER_LAG_FACTOR volte il lag massimo cercato
# (fondamentale minima), altrimenti il picco e' calcolato su pochi prodotti
# ed e' rumoroso. A 48 kHz con blocchi da 1024 bastano 2 blocchi (~43 ms di
# memoria), dentro il budget di latenza percettiva (<200 ms).
ACF_BUFFER_LAG_FACTOR = 3.0
# Numero minimo di blocchi nel buffer, qualunque sia il sample rate.
ACF_MIN_BLOCKS = 2
# Pavimento dell'energia a lag zero: sotto, il buffer e' silenzio e
# l'armonicita' e' 0 (evita di normalizzare rumore numerico).
ACF_ENERGY_FLOOR = 1e-12

# --- Rampa dell'armonicita' ---
# Mappa la prominenza del picco vocale (picco vocale - picco sub-multiplo)
# in [0, 1]: sotto HARM_RAMP_ZERO e' rumore di stima (le fluttuazioni
# dell'ACF del rumore filtrato in banda arrivano a ~0.1), sopra
# HARM_RAMP_FULL e' periodicita' franca.
HARM_RAMP_ZERO = 0.10
HARM_RAMP_FULL = 0.55

# --- Envelope attack/release sull'output (costanti di tempo in secondi) ---
# Attack veloce per agganciare gli ingressi della voce, release piu' lento
# per tenere il livello attraverso vibrato e consonanti. Convertiti in
# coefficienti per blocco in __init__ (il sample rate puo' variare).
ATTACK_SECONDS = 0.05
RELEASE_SECONDS = 0.20

# --- Gate di instabilita' (anti falsi positivi su toni tenuti) ---
# Una sorgente tonale TENUTA (synth lead/pad, archi, sax sostenuto) ha f0 e
# formanti fermi: supera il prodotto energia*armonicita' pur non essendo voce.
# La voce vera muove il centroide della banda vocale (vibrato + formanti che
# cambiano con le vocali). Misuriamo la variabilita' a breve termine (~0.5 s)
# della posizione normalizzata del centroide in banda vocale e ne ricaviamo un
# moltiplicatore additivo in [STABLE_MULT_FLOOR, 1] che pesa il target: fonte
# ferma -> verso il floor (soppressa), centroide che si muove -> verso 1 (voce).
# Scelto il centroide e non il lag ACF del pitch perche' (a) e' gia' derivabile
# dalle magnitudo che l'analyzer ci passa (nessun lag interno da esporre) e
# (b) cattura SIA il vibrato SIA il movimento dei formanti (le vocali), mentre
# il solo pitch mancherebbe i cambi di vocale a fondamentale costante.
INSTABILITY_WINDOW_SECONDS = 0.5
# Soglie della rampa sulla deviazione standard della posizione del centroide
# (normalizzata in [0, 1] sulla banda vocale) nella finestra. Sotto FLOOR il
# centroide e' praticamente fermo (tono tenuto: solo micro-jitter da leakage);
# sopra FULL si muove come una linea vocale. Valori di primo impianto, da
# tarare A/B su voce cantata vs synth lead/sax tenuti (criterio B10): la voce
# cantata deve restare ~1, il lead tenuto crollare verso il floor.
INSTABILITY_VAR_FLOOR = 0.008
INSTABILITY_VAR_FULL = 0.04
# Moltiplicatore minimo per una sorgente tonale perfettamente stabile: non 0
# per non spegnere del tutto un canale voice su strumentale tonale, ma basso
# abbastanza da distinguerlo nettamente dalla voce.
STABLE_MULT_FLOOR = 0.25
# Somma spettrale minima in banda vocale per definire il centroide: sotto, il
# centroide e' rumore numerico e si tiene la posizione precedente (il silenzio
# conta come "stabile", ma li' il target e' comunque ~0).
CENTROID_SUM_FLOOR = 1e-6

# --- Clamp dell'output ---
OUTPUT_MIN = 0.0
OUTPUT_MAX = 1.0


class VoiceDetector:
    """Stima la presenza di voce cantata in un flusso audio a blocchi.

    Per ogni blocco:

    - misura l'energia RMS dei bin rFFT (finestra di Hann) nella banda vocale
      e la normalizza con un AGC a inseguimento di picco, con pavimento
      incrociato sul picco RMS dell'intero spettro;
    - stima l'armonicita' come prominenza del picco di autocorrelazione nei
      lag delle fondamentali vocali (80-400 Hz). L'autocorrelazione (circolare,
      via Wiener-Khinchin) e' calcolata sullo spettro filtrato in banda vocale
      di un buffer degli ultimi blocchi, con reiezione dei sub-multipli dei
      toni con f0 sopra il range;
    - applica un gate di instabilita': un moltiplicatore in
      [STABLE_MULT_FLOOR, 1] ricavato dalla variabilita' a breve termine del
      centroide in banda vocale, che sopprime le sorgenti tonali tenute (synth
      lead/pad, archi, sax sostenuto) lasciando passare la voce vera;
    - combina i fattori (prodotto) e liscia con envelope attack/release.

    Attributes:
        sample_rate: Frequenza di campionamento in Hz.
        block_size: Numero di campioni per blocco.
    """

    def __init__(self, sample_rate: int, block_size: int, hop: int | None = None) -> None:
        """Inizializza il rilevatore e precomputa finestra, maschere e decay.

        Args:
            sample_rate: Frequenza di campionamento del segnale in Hz.
            block_size: Numero di campioni attesi in ogni blocco passato a
                `process` (larghezza della finestra).
            hop: Passo di avanzamento tra due blocchi consecutivi, in campioni.
                None (default) equivale a `block_size`: blocchi disgiunti, cioe'
                il comportamento storico. Con `block_size // 2` le finestre si
                sovrappongono al 50%: il buffer ACF appende allora solo la coda
                nuova `block[-hop:]` e le costanti temporali (envelope, finestra
                di instabilita') si ricavano dall'hop, restando invarianti in
                secondi reali.
        """
        self.sample_rate: int = sample_rate
        self.block_size: int = block_size
        # Hop effettivo: default block_size (blocchi disgiunti). La durata del
        # blocco (intervallo reale tra due process) e' il passo dell'hop.
        self._hop: int = block_size if hop is None else hop
        block_duration: float = self._hop / sample_rate

        # Finestra di Hann e maschera della banda vocale sul blocco singolo.
        self._window: np.ndarray = np.hanning(block_size).astype(np.float32)
        freqs: np.ndarray = np.fft.rfftfreq(block_size, d=1.0 / sample_rate)
        self._voice_mask: np.ndarray = (freqs >= VOICE_LOW_HZ) & (freqs <= VOICE_HIGH_HZ)
        # Frequenze dei bin in banda vocale, per il centroide del gate di
        # instabilita' (stessa maschera dell'energia).
        self._voice_freqs: np.ndarray = freqs[self._voice_mask]
        self._voice_band_span: float = VOICE_HIGH_HZ - VOICE_LOW_HZ

        # AGC: picco della banda vocale + picco dell'intero spettro come
        # pavimento incrociato, con decadimento esponenziale condiviso.
        self._peak_decay: float = float(np.exp(-block_duration / PEAK_DECAY_SECONDS))
        self._voice_peak: float = PEAK_FLOOR
        self._total_peak: float = PEAK_FLOOR

        # Lag di ricerca del picco di autocorrelazione (campioni).
        lag_max: int = int(round(sample_rate / F0_MIN_HZ))
        self._lag_min: int = int(round(sample_rate / F0_MAX_HZ))
        self._sub_lag_min: int = max(SUB_LAG_MIN_SAMPLES, int(round(sample_rate / SUB_F0_MAX_HZ)))

        # Buffer scorrevole per l'autocorrelazione: abbastanza blocchi da
        # coprire ACF_BUFFER_LAG_FACTOR volte il lag massimo.
        acf_blocks: int = max(
            ACF_MIN_BLOCKS, int(np.ceil(ACF_BUFFER_LAG_FACTOR * lag_max / block_size))
        )
        buffer_len: int = acf_blocks * block_size
        self._acf_buffer: np.ndarray = np.zeros(buffer_len, dtype=np.float32)
        acf_freqs: np.ndarray = np.fft.rfftfreq(buffer_len, d=1.0 / sample_rate)
        self._acf_mask: np.ndarray = (acf_freqs >= VOICE_LOW_HZ) & (acf_freqs <= VOICE_HIGH_HZ)
        # L'ACF circolare e' simmetrica intorno a meta' buffer: i lag utili
        # si fermano prima.
        self._lag_max: int = min(lag_max, buffer_len // 2 - 1)

        # Envelope asimmetrico: coefficienti per blocco dalle costanti di
        # tempo (coeff = 1 - exp(-durata_blocco / tau)).
        self._attack_coeff: float = 1.0 - float(np.exp(-block_duration / ATTACK_SECONDS))
        self._release_coeff: float = 1.0 - float(np.exp(-block_duration / RELEASE_SECONDS))
        self._level: float = OUTPUT_MIN

        # Gate di instabilita': storia della posizione normalizzata del centroide
        # in banda vocale su ~INSTABILITY_WINDOW_SECONDS (almeno 2 blocchi per
        # avere una std definita). Il conteggio dei blocchi segue l'hop (una
        # posizione per chiamata a process), cosi' copre gli stessi secondi
        # reali anche con le finestre sovrapposte.
        self._centroid_window: int = max(
            2, int(round(INSTABILITY_WINDOW_SECONDS * sample_rate / self._hop))
        )
        self._centroid_hist: deque[float] = deque(maxlen=self._centroid_window)
        self._centroid_pos: float = 0.0

    def process(self, block: np.ndarray, magnitudes: np.ndarray | None = None) -> float:
        """Elabora un blocco audio e restituisce il livello di presenza vocale.

        Args:
            block: Array float32 mono con valori in [-1, 1] e lunghezza pari
                a `block_size`.
            magnitudes: Ampiezze rFFT finestrate (Hann) del blocco, gia'
                calcolate dall'analyzer con la STESSA finestra e lo STESSO
                block_size (`np.abs(rfft(block * hann))`); usate per l'energia
                in banda ed evitano una seconda FFT identica. Se None o di forma
                incompatibile col block_size atteso, vengono ricalcolate qui
                (percorso sicuro). Il percorso ACF dell'armonicita' resta sempre
                interno (lavora sul buffer lungo, non su queste magnitudo).

        Returns:
            Livello di presenza vocale in [0.0, 1.0]: alto con una voce che
            canta nel mix, basso su strumentale pura o silenzio.
        """
        # 1. Energia RMS della banda vocale su rFFT finestrata (Hann),
        # normalizzata dall'AGC col pavimento incrociato sul picco totale.
        # Riusa le magnitudo dell'analyzer se compatibili, altrimenti ricalcola.
        if magnitudes is None or magnitudes.shape != self._voice_mask.shape:
            magnitudes = np.abs(np.fft.rfft(block * self._window))
        band_mags: np.ndarray = magnitudes[self._voice_mask]
        voice_energy: float = float(np.sqrt(np.mean(band_mags**2)))
        total_energy: float = float(np.sqrt(np.mean(magnitudes**2)))
        self._voice_peak = max(self._voice_peak * self._peak_decay, voice_energy, PEAK_FLOOR)
        self._total_peak = max(self._total_peak * self._peak_decay, total_energy, PEAK_FLOOR)
        denom: float = max(self._voice_peak, TOTAL_PEAK_FLOOR_SHARE * self._total_peak)
        energy_norm: float = min(voice_energy / denom, 1.0)

        # 2. Aggiorna il buffer scorrevole e stima l'armonicita'. Il buffer
        # AVANZA di hop e appende SOLO la coda nuova del blocco (block[-hop:]):
        # con l'overlap i blocchi consecutivi condividono campioni, appendere
        # l'intero blocco li duplicherebbe. Con hop = block_size (default)
        # equivale a shiftare di un blocco intero.
        hop: int = self._hop
        self._acf_buffer[:-hop] = self._acf_buffer[hop:]
        self._acf_buffer[-hop:] = block[-hop:]
        harmonicity: float = self._harmonicity()

        # 3. Gate di instabilita': variabilita' a breve termine del centroide
        # in banda vocale. Pesa (non sostituisce) il target: fonte ferma verso
        # il floor, voce con formanti/vibrato in movimento verso 1.
        instability: float = self._instability_multiplier(band_mags)

        # 4. Prodotto energia * armonicita' * gate, lisciato con envelope.
        target: float = energy_norm * harmonicity * instability
        coeff: float = self._attack_coeff if target > self._level else self._release_coeff
        self._level += (target - self._level) * coeff
        return float(min(max(self._level, OUTPUT_MIN), OUTPUT_MAX))

    def _instability_multiplier(self, band_mags: np.ndarray) -> float:
        """Moltiplicatore in [STABLE_MULT_FLOOR, 1] dalla variabilita' del centroide.

        Calcola il centroide spettrale della banda vocale, lo normalizza in
        [0, 1] sulla banda e ne accumula la storia su ~INSTABILITY_WINDOW_SECONDS.
        La deviazione standard della storia misura quanto la sorgente si muove:
        una linea vocale (vibrato + formanti che cambiano con le vocali) ha std
        alta, un tono tenuto std quasi nulla. La std viene mappata da una rampa
        (INSTABILITY_VAR_FLOOR -> INSTABILITY_VAR_FULL) sul moltiplicatore. Con
        storia ancora incompleta ritorna 1.0 (nessuna penalita': si sopprime
        solo dopo aver CONFERMATO la stabilita', coerente con precisione > recall).

        Args:
            band_mags: Ampiezze rFFT dei bin in banda vocale del blocco corrente.

        Returns:
            Moltiplicatore in [STABLE_MULT_FLOOR, 1.0].
        """
        band_sum: float = float(np.sum(band_mags))
        if band_sum > CENTROID_SUM_FLOOR:
            centroid: float = float(np.sum(self._voice_freqs * band_mags) / band_sum)
            self._centroid_pos = min(
                max((centroid - VOICE_LOW_HZ) / self._voice_band_span, 0.0), 1.0
            )
        # Sotto il pavimento (silenzio/quasi): tiene la posizione precedente,
        # cosi' il silenzio conta come stabile (li' il target e' comunque ~0).
        self._centroid_hist.append(self._centroid_pos)

        if len(self._centroid_hist) < self._centroid_window:
            return 1.0
        variability: float = float(np.std(self._centroid_hist))
        ramp: float = (variability - INSTABILITY_VAR_FLOOR) / (
            INSTABILITY_VAR_FULL - INSTABILITY_VAR_FLOOR
        )
        ramp = min(max(ramp, 0.0), 1.0)
        return STABLE_MULT_FLOOR + (1.0 - STABLE_MULT_FLOOR) * ramp

    def _harmonicity(self) -> float:
        """Armonicita' in [0, 1] del contenuto in banda vocale del buffer.

        Autocorrelazione circolare via Wiener-Khinchin (irfft del power
        spectrum mascherato sulla banda vocale): filtra e autocorrela in un
        colpo solo. La prominenza e' il picco nei lag delle f0 vocali meno il
        miglior picco ai lag piu' corti (reiezione dei sub-multipli), passata
        da una rampa per schiacciare il rumore di stima.
        """
        spectrum: np.ndarray = np.fft.rfft(self._acf_buffer)
        power: np.ndarray = np.abs(spectrum) ** 2
        power[~self._acf_mask] = 0.0
        acf: np.ndarray = np.fft.irfft(power)

        energy: float = float(acf[0])
        if energy < ACF_ENERGY_FLOOR:
            return 0.0
        acf /= energy

        vocal_peak: float = self._best_local_peak(acf, self._lag_min, self._lag_max)
        sub_peak: float = self._best_local_peak(acf, self._sub_lag_min, self._lag_min - 1)
        raw: float = max(vocal_peak - sub_peak, 0.0)
        ramp: float = (raw - HARM_RAMP_ZERO) / (HARM_RAMP_FULL - HARM_RAMP_ZERO)
        return float(min(max(ramp, 0.0), 1.0))

    @staticmethod
    def _best_local_peak(acf: np.ndarray, lag_low: int, lag_high: int) -> float:
        """Massimo tra i massimi locali di `acf` nei lag [lag_low, lag_high].

        Solo massimi locali veri (>= dei due vicini): i valori di bordo di un
        andamento monotono non contano come picco. Ritorna 0.0 se non ci sono
        picchi o se il migliore e' negativo.
        """
        if lag_high <= lag_low or lag_low < 1:
            return 0.0
        segment: np.ndarray = acf[lag_low - 1 : lag_high + 2]
        values: np.ndarray = segment[1:-1]
        is_peak: np.ndarray = (values >= segment[:-2]) & (values >= segment[2:])
        if not np.any(is_peak):
            return 0.0
        return float(max(float(np.max(values[is_peak])), 0.0))
