"""Beat tracking realtime per huebeat.

Modulo di puro calcolo (nessun logging, nessun effetto collaterale). Consuma
un valore di flux (inviluppo di onset in [0, 1], gia' normalizzato altrove)
per ogni blocco audio e stima tempo (BPM) e griglia dei beat.

Interfaccia pubblica: `BeatTracker`, che restituisce a ogni blocco un
`BeatInfo` con BPM, stato di lock, fase del beat e flag `is_beat`.

Filosofia: precisione > recall. Meglio non dichiarare un beat che
dichiararne uno sbagliato, quindi il lock richiede stime consecutive
affidabili e con silenzio o segnale stazionario il tracker non locka mai.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

# --- Intervallo di tempo valido (BPM) ---
# Copre la quasi totalita' della musica pilotabile con le luci; fuori da
# questo intervallo l'autocorrelazione aggancia armoniche/sottoarmoniche.
BPM_MIN = 60.0
BPM_MAX = 200.0

# --- Storia del flux per l'autocorrelazione ---
# Finestra abbastanza lunga da contenere ~8 beat anche a 60 BPM (stima
# stabile), abbastanza corta da riagganciare un cambio di tempo in pochi
# secondi (la finestra e' meta' vecchio tempo / meta' nuovo dopo ~4 s).
HISTORY_SECONDS = 8.0
# Sotto questa durata l'acf ai lag lunghi (60 BPM ~ 1 s) ha troppi pochi
# prodotti per essere affidabile: nessuna stima prima.
MIN_HISTORY_SECONDS = 2.5

# --- Cadenza della ristima del tempo ---
# La ristima costa O(N^2) sull'intera storia: farla ogni ~0.5 s la rende
# trascurabile a 47 fps mantenendo un riaggancio rapido sui cambi di tempo.
TEMPO_INTERVAL_SECONDS = 0.5

# --- Prior log-normale sul tempo (tecnica alla Ellis) ---
# Centro a 120 BPM: moda della distribuzione dei tempi nella musica pop/
# elettronica. Sigma in ottave: 0.9 penalizza dolcemente le ottave estreme
# senza impedire tempi legittimi a 60 o 200 BPM.
PRIOR_CENTER_BPM = 120.0
PRIOR_SIGMA_OCTAVES = 0.9

# --- Disambiguazione di ottava ---
# Se l'acf a meta' del lag vincente vale almeno questa frazione del picco,
# il periodo vero e' la meta' (il picco scelto era la sottoarmonica): per un
# treno di beat acf(P) >= acf(2P) sempre, quindi il rapporto e' ~1 quando P
# e' il periodo vero e ~0 quando il mezzo lag non e' periodico. 0.75 lascia
# margine per l'energia in levare della musica reale.
OCTAVE_ACF_RATIO = 0.75

# --- Guardia sul segnale ---
# Deviazione standard minima della storia di flux per tentare una stima:
# sotto, il segnale e' silenzio o stazionario (flux ~ costante) e l'acf
# normalizzata amplificherebbe solo rumore. Il flux e' in [0, 1] e i beat
# veri portano la std ben sopra 0.01.
FLUX_STD_FLOOR = 0.01
# Pavimento numerico del denominatore dell'interpolazione parabolica.
PARABOLIC_EPS = 1e-12

# --- Isteresi di lock ---
# Soglia di confidence per considerare buona una stima: il rumore puro
# produce picchi acf normalizzati < ~0.15, i beat reali > ~0.4.
LOCK_CONFIDENCE = 0.35
# Sotto questa soglia la stima e' considerata persa; la fascia tra le due
# soglie e' zona morta (isteresi) in cui lo stato non cambia.
UNLOCK_CONFIDENCE = 0.15
# Stime buone consecutive per lockare: 2 stime a 0.5 s = lock in ~1 s dal
# raggiungimento della storia minima, e un singolo picco spurio non locka.
LOCK_STREAK = 2
# Stime cattive consecutive per unlockare: 3 evitano di perdere il lock nei
# break o nelle transizioni di tempo dove la confidence cala per ~1 s.
UNLOCK_STREAK = 3

# --- Stabilizzazione del BPM ---
# Mediana delle ultime stime: robusta a una singola stima errata di ottava,
# 5 valori (2.5 s) seguono comunque un cambio di tempo in pochi secondi.
BPM_MEDIAN_WINDOW = 5

# --- Aggancio di fase ---
# Guadagno parziale della correzione verso il picco osservato: 0.35 converge
# in 2-3 beat senza inseguire i picchi spuri.
PHASE_GAIN = 0.35
# Semiampiezza della finestra di accettazione della correzione, come
# frazione del periodo: i picchi oltre +-15% dalla previsione sono ignorati
# (probabili levare o ghost note).
PHASE_WINDOW_FRACTION = 0.15
# Flux minimo perche' un massimo locale conti come picco per la fase: filtra
# il rumore di fondo (quasi zero su segnale stazionario), i transienti veri
# arrivano ben sopra.
PEAK_MIN_FLUX = 0.2

# --- Suddivisioni del beat (ottavi/sedicesimi) ---
# Numero di suddivisioni del periodo: la fase attraversa k/DIVISIONS del
# periodo. Gli ottavi (2 per beat) sono un sottoinsieme dei sedicesimi (4 per
# beat): l'attraversamento di 1/2 e' anche un attraversamento di 2/4, quindi
# ogni ottavo e' anche un sedicesimo, e il beat (fase 0) e' entrambi.
EIGHTH_DIVISIONS = 2
SIXTEENTH_DIVISIONS = 4


@dataclass(frozen=True)
class BeatInfo:
    """Stato del beat tracking per un blocco.

    Attributes:
        bpm: BPM stimato; 0.0 se non lockato.
        locked: True se il tracker ha una stima affidabile.
        beat_phase: Fase in [0, 1): 0.0 = sul beat previsto, cresce
            linearmente fino al prossimo beat.
        is_beat: True solo nel blocco piu' vicino al beat previsto
            (uno per beat, solo se locked).
        confidence: Qualita' della stima di tempo in [0, 1].
        is_eighth: True nel blocco in cui la fase attraversa un ottavo del beat
            (k/2 del periodo: 2 eventi per beat). is_beat implica is_eighth.
        is_sixteenth: True nel blocco in cui la fase attraversa un sedicesimo
            (k/4 del periodo: 4 eventi per beat). is_beat e is_eighth implicano
            is_sixteenth (ogni ottavo e' anche un sedicesimo).
    """

    bpm: float
    locked: bool
    beat_phase: float
    is_beat: bool
    confidence: float
    is_eighth: bool = False
    is_sixteenth: bool = False


class BeatTracker:
    """Stima tempo e fase dei beat da un flusso di flux per blocco.

    Mantiene una storia del flux (~HISTORY_SECONDS) e ogni
    ~TEMPO_INTERVAL_SECONDS ne calcola l'autocorrelazione (mean-removed) sui
    lag corrispondenti a BPM_MIN-BPM_MAX, pesata da un prior log-normale
    centrato su PRIOR_CENTER_BPM per disambiguare le ottave; il picco viene
    raffinato con interpolazione parabolica e il BPM stabilizzato con la
    mediana delle ultime stime. Il lock scatta dopo LOCK_STREAK stime
    consecutive sopra soglia e cade dopo UNLOCK_STREAK stime sotto isteresi.

    La fase avanza di un periodo per beat; i picchi locali di flux vicini
    alla previsione (+-PHASE_WINDOW_FRACTION del periodo) correggono la
    previsione con guadagno parziale PHASE_GAIN.

    Attributes:
        block_rate_hz: Frequenza dei blocchi in Hz (chiamate a `update` al
            secondo).
    """

    def __init__(self, block_rate_hz: float) -> None:
        """Inizializza il tracker per un dato block rate.

        Args:
            block_rate_hz: Frequenza dei blocchi in Hz (es. 46.875 per
                blocchi da 1024 campioni a 48 kHz). Deve essere positiva.

        Raises:
            ValueError: Se `block_rate_hz` non e' positiva.
        """
        if block_rate_hz <= 0.0:
            raise ValueError(f"block_rate_hz deve essere positiva, ricevuto {block_rate_hz}")
        self.block_rate_hz: float = block_rate_hz

        history_blocks: int = int(round(HISTORY_SECONDS * block_rate_hz))
        self._history: deque[float] = deque(maxlen=history_blocks)
        self._min_history_blocks: int = int(round(MIN_HISTORY_SECONDS * block_rate_hz))
        self._estimate_interval_blocks: int = max(
            1, int(round(TEMPO_INTERVAL_SECONDS * block_rate_hz))
        )
        self._blocks_since_estimate: int = 0

        # Indice del blocco corrente (aggiornato a inizio update).
        self._block_index: int = -1

        # Stato della stima di tempo.
        self._bpm_history: deque[float] = deque(maxlen=BPM_MEDIAN_WINDOW)
        self._bpm: float = 0.0
        self._confidence: float = 0.0
        self._locked: bool = False
        self._good_streak: int = 0
        self._bad_streak: int = 0

        # Stato della fase: previsione del prossimo beat (in blocchi,
        # frazionaria) e rilevatore di picchi locali di flux.
        self._next_beat_block: float = 0.0
        self._prev_flux: float = 0.0
        self._prev_prev_flux: float = 0.0
        self._last_peak_block: int | None = None
        # Fase del beat del blocco precedente, per rilevare gli attraversamenti
        # delle suddivisioni (ottavi/sedicesimi).
        self._prev_beat_phase: float = 0.0

    def reset(self) -> None:
        """Azzera storia, lock e fase: riparte da zero come appena costruito.

        Da chiamare al cambio brano (il BPM vecchio e la storia del flux non
        valgono piu' per il brano nuovo). Le costanti derivate dal block rate
        restano: solo lo STATO viene svuotato.
        """
        self._history.clear()
        self._blocks_since_estimate = 0
        self._block_index = -1
        self._bpm_history.clear()
        self._bpm = 0.0
        self._confidence = 0.0
        self._locked = False
        self._good_streak = 0
        self._bad_streak = 0
        self._next_beat_block = 0.0
        self._prev_flux = 0.0
        self._prev_prev_flux = 0.0
        self._last_peak_block = None
        self._prev_beat_phase = 0.0

    def update(self, flux: float) -> BeatInfo:
        """Consuma il flux di un blocco e restituisce lo stato corrente.

        Va chiamata una volta per blocco. Costo O(1) ammortizzato: la
        ristima del tempo avviene solo ogni ~TEMPO_INTERVAL_SECONDS.

        Args:
            flux: Inviluppo di onset del blocco corrente in [0, 1].

        Returns:
            Lo stato del beat tracking per questo blocco.
        """
        self._block_index += 1
        self._history.append(flux)

        peak_block: int | None = self._detect_peak(flux)
        if peak_block is not None:
            self._last_peak_block = peak_block

        # Ristima periodica del tempo (solo con storia sufficiente).
        self._blocks_since_estimate += 1
        if (
            len(self._history) >= self._min_history_blocks
            and self._blocks_since_estimate >= self._estimate_interval_blocks
        ):
            self._blocks_since_estimate = 0
            self._run_estimate()

        is_beat: bool = False
        is_eighth: bool = False
        is_sixteenth: bool = False
        beat_phase: float = 0.0
        if self._locked:
            period: float = self.block_rate_hz * 60.0 / self._bpm
            if peak_block is not None:
                self._correct_phase(peak_block, period)
            # Emissione: is_beat nel blocco piu' vicino alla previsione.
            if self._block_index >= self._next_beat_block - 0.5:
                is_beat = True
                self._next_beat_block += period
            beat_phase = 1.0 - (self._next_beat_block - self._block_index) / period
            beat_phase = min(max(beat_phase, 0.0), 1.0 - PARABOLIC_EPS)
            # Suddivisioni: sul beat (fase ~0, appena wrappata) valgono entrambe;
            # altrimenti si guarda quali confini k/n la fase ha attraversato tra
            # il blocco precedente e questo.
            if is_beat:
                is_eighth = True
                is_sixteenth = True
            else:
                is_eighth = self._crossed_subdivision(
                    self._prev_beat_phase, beat_phase, EIGHTH_DIVISIONS
                )
                is_sixteenth = self._crossed_subdivision(
                    self._prev_beat_phase, beat_phase, SIXTEENTH_DIVISIONS
                )
        self._prev_beat_phase = beat_phase

        return BeatInfo(
            bpm=self._bpm if self._locked else 0.0,
            locked=self._locked,
            beat_phase=beat_phase,
            is_beat=is_beat,
            confidence=self._confidence,
            is_eighth=is_eighth,
            is_sixteenth=is_sixteenth,
        )

    @staticmethod
    def _crossed_subdivision(prev_phase: float, phase: float, divisions: int) -> bool:
        """True se la fase ha attraversato un confine di suddivisione interno.

        I confini interni sono j/divisions per j = 1..divisions-1 (lo 0 e' il
        beat, gestito a parte). Il caso di wrap (prev > phase, cioe' un beat
        appena emesso) non passa di qui: la fase non retrocede mai tra due
        blocchi senza un beat, quindi basta il controllo prev < confine <= phase.

        Args:
            prev_phase: Fase del beat del blocco precedente in [0, 1).
            phase: Fase del beat del blocco corrente in [0, 1).
            divisions: Numero di suddivisioni del periodo (2 = ottavi, 4 = sedic.).

        Returns:
            True se un confine interno cade in (prev_phase, phase].
        """
        for j in range(1, divisions):
            boundary: float = j / divisions
            if prev_phase < boundary <= phase:
                return True
        return False

    def _detect_peak(self, flux: float) -> int | None:
        """Rileva un massimo locale di flux (con un blocco di latenza).

        Il picco e' al blocco precedente quando il suo flux supera sia il
        blocco prima sia quello corrente ed e' sopra PEAK_MIN_FLUX; la
        disuguaglianza stretta sul passato evita picchi su flux costante.

        Args:
            flux: Flux del blocco corrente.

        Returns:
            Indice del blocco del picco, oppure None.
        """
        peak: int | None = None
        if (
            self._prev_flux >= PEAK_MIN_FLUX
            and self._prev_flux > self._prev_prev_flux
            and self._prev_flux >= flux
        ):
            peak = self._block_index - 1
        self._prev_prev_flux = self._prev_flux
        self._prev_flux = flux
        return peak

    def _run_estimate(self) -> None:
        """Esegue una ristima del tempo e aggiorna lock, BPM e confidence."""
        bpm, confidence = self._estimate_tempo()
        self._confidence = confidence

        if bpm > 0.0:
            self._bpm_history.append(bpm)

        if confidence >= LOCK_CONFIDENCE:
            self._good_streak += 1
            self._bad_streak = 0
        elif confidence < UNLOCK_CONFIDENCE:
            self._bad_streak += 1
            self._good_streak = 0
        else:
            # Zona di isteresi: nessuno dei due streak avanza.
            self._good_streak = 0

        if not self._locked and self._good_streak >= LOCK_STREAK:
            self._locked = True
            self._bpm = float(np.median(self._bpm_history))
            self._init_phase()
        elif self._locked and self._bad_streak >= UNLOCK_STREAK:
            self._locked = False
            self._bpm = 0.0
            self._good_streak = 0
        elif self._locked and self._bpm_history:
            self._bpm = float(np.median(self._bpm_history))

    def _estimate_tempo(self) -> tuple[float, float]:
        """Stima il BPM dall'autocorrelazione della storia di flux.

        Returns:
            Coppia (bpm, confidence); (0.0, 0.0) se il segnale non e'
            stimabile (silenzio, flux costante, storia degenere).
        """
        x: np.ndarray = np.asarray(self._history, dtype=np.float64)
        x = x - x.mean()
        if float(x.std()) < FLUX_STD_FLOOR:
            return 0.0, 0.0

        n: int = len(x)
        acf: np.ndarray = np.correlate(x, x, mode="full")[n - 1 :]
        acf0: float = float(acf[0])
        if acf0 <= 0.0:
            return 0.0, 0.0
        acf_norm: np.ndarray = acf / acf0

        min_lag: int = max(1, int(np.floor(self.block_rate_hz * 60.0 / BPM_MAX)))
        max_lag: int = min(int(np.ceil(self.block_rate_hz * 60.0 / BPM_MIN)), n - 2)
        if max_lag <= min_lag + 1:
            return 0.0, 0.0

        # Punteggio: acf normalizzata pesata dal prior log-normale sui BPM.
        lags: np.ndarray = np.arange(min_lag, max_lag + 1)
        lag_bpms: np.ndarray = 60.0 * self.block_rate_hz / lags
        log_octaves: np.ndarray = np.log2(lag_bpms / PRIOR_CENTER_BPM)
        prior: np.ndarray = np.exp(-0.5 * (log_octaves / PRIOR_SIGMA_OCTAVES) ** 2)
        scores: np.ndarray = acf_norm[lags] * prior
        best_lag: int = int(lags[int(np.argmax(scores))])

        # Anti-ottava: se il mezzo lag e' periodico quanto il picco, il
        # tempo vero e' il doppio (purche' resti entro BPM_MAX). Il confronto
        # con OCTAVE_ACF_RATIO usa il valore DEbiased acf_db(lag) = acf[lag] *
        # n / (n - lag): l'acf biased (acf/acf[0]) sottopesa i lag lunghi (meno
        # prodotti sovrapposti), quindi favorirebbe il tempo piu' veloce (mezzo
        # lag) nel rapporto. Il debias e' calcolato da acf_norm (il fattore
        # acf[0] si semplifica nel rapporto, quindi confronto identico a quello
        # su acf grezza). La SELEZIONE del lag e la confidence restano sul biased
        # (tarati cosi'); OCTAVE_ACF_RATIO non cambia.
        half_lag: float = best_lag / 2.0
        if half_lag >= self.block_rate_hz * 60.0 / BPM_MAX:
            lo: int = max(1, int(np.floor(half_lag)) - 1)
            hi: int = min(n - 1, int(np.ceil(half_lag)) + 1)
            half_window: np.ndarray = acf_norm[lo : hi + 1]
            # Debias solo per la decisione (n - lag > 0 garantito: hi <= n - 1 e
            # best_lag <= max_lag <= n - 2).
            half_lags: np.ndarray = np.arange(lo, hi + 1)
            half_db: np.ndarray = acf_norm[half_lags] * n / (n - half_lags)
            best_db: float = float(acf_norm[best_lag]) * n / (n - best_lag)
            if float(np.max(half_db)) >= OCTAVE_ACF_RATIO * best_db:
                best_lag = lo + int(np.argmax(half_window))

        refined_lag: float = self._refine_peak(acf_norm, best_lag)
        # Guardia: l'interpolazione parabolica puo' spingere il lag sotto min_lag
        # (fino a best_lag - 1); con block size grandi min_lag puo' valere 1 e un
        # refined_lag ~0 darebbe divisione per zero / BPM infinito. Il pavimento
        # a min_lag (lag di BPM_MAX) tiene il BPM entro il range valido.
        refined_lag = max(refined_lag, float(min_lag))
        bpm: float = 60.0 * self.block_rate_hz / refined_lag
        bpm = min(max(bpm, BPM_MIN), BPM_MAX)
        confidence: float = min(max(float(acf_norm[best_lag]), 0.0), 1.0)
        return bpm, confidence

    @staticmethod
    def _refine_peak(acf_norm: np.ndarray, lag: int) -> float:
        """Raffina la posizione del picco acf con interpolazione parabolica.

        Args:
            acf_norm: Autocorrelazione normalizzata (lag 0..N-1).
            lag: Lag intero del picco.

        Returns:
            Lag frazionario raffinato (uguale a `lag` se non raffinabile).
        """
        if lag < 1 or lag >= len(acf_norm) - 1:
            return float(lag)
        y0: float = float(acf_norm[lag - 1])
        y1: float = float(acf_norm[lag])
        y2: float = float(acf_norm[lag + 1])
        denom: float = y0 - 2.0 * y1 + y2
        if abs(denom) < PARABOLIC_EPS:
            return float(lag)
        delta: float = 0.5 * (y0 - y2) / denom
        if abs(delta) > 1.0:
            return float(lag)
        return lag + delta

    def _init_phase(self) -> None:
        """Inizializza la previsione del prossimo beat al momento del lock.

        Ancora la griglia all'ultimo picco di flux osservato (se presente) e
        avanza di periodi interi fino a superare il blocco corrente, cosi'
        il primo `is_beat` cade sul prossimo beat della griglia reale.
        """
        period: float = self.block_rate_hz * 60.0 / self._bpm
        if self._last_peak_block is not None:
            self._next_beat_block = self._last_peak_block + period
        else:
            self._next_beat_block = self._block_index + period
        while self._next_beat_block <= self._block_index + 0.5:
            self._next_beat_block += period

    def _correct_phase(self, peak_block: int, period: float) -> None:
        """Corregge la previsione del beat verso un picco di flux osservato.

        L'errore e' calcolato modulo periodo (il picco puo' riferirsi al
        beat appena emesso o al prossimo) e applicato solo se entro la
        finestra di accettazione, con guadagno parziale.

        Args:
            peak_block: Indice del blocco del picco osservato.
            period: Periodo corrente del beat in blocchi.
        """
        error: float = peak_block - self._next_beat_block
        error -= period * round(error / period)
        if abs(error) <= PHASE_WINDOW_FRACTION * period:
            self._next_beat_block += PHASE_GAIN * error
