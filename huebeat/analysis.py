"""Analisi audio per huebeat.

Modulo di puro calcolo (nessun logging, nessun effetto collaterale). Riceve
blocchi audio mono float32 e ne estrae feature normalizzate in [0, 1], pronte
per pilotare l'intensita' delle luci.

Interfaccia pubblica: `FeatureAnalyzer`, che estrae l'energia di tre bande
(bass/mid/high) piu' un rilevatore di onset via spectral flux, restituite come
`AudioFeatures`.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from huebeat.chroma import ChromaAnalyzer
from huebeat.hpss import HpssAnalyzer
from huebeat.voice import VoiceDetector

# --- Hop tra finestre sovrapposte (overlap del 50%) ---
# Divisore del block_size per l'hop: hop = block_size // HOP_DIVISOR. Con 2 le
# finestre avanzano di meta' blocco (overlap 50%), raddoppiando il frame rate
# delle luci a parita' di risoluzione FFT (a 48 kHz/1024: da ~46.9 a ~93.75 fps).
# Le catture avanzano il loro ring buffer di questo hop e il cli ne ricava il
# block_rate (= sample_rate / hop); l'analyzer riceve l'hop gia' calcolato come
# parametro (default = block_size, cioe' blocchi disgiunti = comportamento
# storico). Gli unici buffer che ACCUMULANO campioni tra blocchi (il buffer
# esteso del kick qui, il buffer ACF in voice.py) devono percio' appendere solo
# la coda nuova (block[-hop:]); le FFT per-blocco usano sempre la finestra intera.
HOP_DIVISOR = 2

# --- Banda di frequenza dei bassi (Hz) ---
BASS_LOW_HZ = 20.0
BASS_HIGH_HZ = 150.0

# --- AGC a inseguimento di picco ---
# Tempo perche' il picco memorizzato scenda a ~37% (1/e) in assenza di segnale.
PEAK_DECAY_SECONDS = 3.0
# Pavimento del picco: evita divisioni per zero e impedisce all'AGC di
# amplificare il rumore di fondo fino a saturare.
PEAK_FLOOR = 1e-6
# AGC per banda con pavimento incrociato: ogni banda e' normalizzata sul SUO
# picco, ma il denominatore non scende sotto questa frazione del picco
# condiviso (la banda dominante). Cosi' una banda con contenuto vero ma piu'
# debole (i medi/alti sotto una cassa potente) vive sulla propria dinamica e
# arriva a 1.0 sui SUOI picchi, mentre una banda quasi vuota (leakage di un
# tono puro: ~0.4% del picco) resta bassa invece di auto-normalizzarsi a
# saturazione. Misurato su musica reale: con la sola scala
# condivisa i canali non-bassi vivevano a value 0.10-0.23, quasi spenti.
BAND_PEAK_FLOOR_SHARE = 0.1

# --- Envelope smoothing asimmetrico sull'output ---
# Salita quasi immediata, discesa piu' dolce.
ATTACK = 0.9
RELEASE = 0.25

# --- Clamp dell'output ---
OUTPUT_MIN = 0.0
OUTPUT_MAX = 1.0


# --- Bande di frequenza per FeatureAnalyzer (Hz) ---
# Estremi inclusivi; ogni banda seleziona i bin rFFT con freq nell'intervallo.
MID_LOW_HZ = 150.0
MID_HIGH_HZ = 2000.0
HIGH_LOW_HZ = 2000.0
HIGH_HIGH_HZ = 8000.0

# --- Onset detection via spectral flux ---
# Limite superiore dei bin considerati per il flux (in Hz).
FLUX_MAX_HZ = 8000.0
# Moltiplicatore applicato alla mediana della storia recente del flux per
# ricavare la soglia adattiva. ACCOPPIATO allo smoothing del flux: con lo
# smoothing corto (2 blocchi) la distribuzione del flux su segnale stazionario
# e' piu' larga e serve un moltiplicatore piu' alto per non generare falsi
# onset sul tono continuo (griglia misurata: 1.5 -> 33 falsi, 1.8 -> 13,
# 2.0 -> 0 con detection 100% fino a 10 Hz; 2.5 -> 0 falsi ma perde beat
# deboli su tono gia' a 8 Hz). Se cambia lo smoothing, rifare la griglia.
FLUX_THRESHOLD_MULTIPLIER = 2.0
# Durata della storia di flux tenuta per la mediana adattiva. Convertita in
# numero di blocchi nel costruttore (round(secondi*block_rate)), cosi' e'
# invariante al sample rate: a 48 kHz con blocchi da 1024 campioni da' 43
# blocchi (0.92*46.875=43.125 -> 43), il valore tarato sul campo.
FLUX_HISTORY_SECONDS = 0.92
# Minimo numero di valori in storia prima di emettere un onset: sotto questa
# soglia la mediana non e' ancora affidabile e onset_strength resta 0.
FLUX_MIN_HISTORY = 5
# Pavimento numerico del flux: evita divisioni per zero nella normalizzazione.
FLUX_FLOOR = 1e-6

# --- Kick detector (onset sulla sola banda bassi) ---
# Refrattarieta' dedicata, allineata al broadband: 0.06 s (~3 blocchi) regge
# i bassline run veloci (sul campo, a 0.1 s i cambi rapidi nei bassi si
# perdevano); resta sopra la finestra di smoothing (2 blocchi ~ 43 ms).
BASS_ONSET_MIN_GAP_SECONDS = 0.06
# La FFT del blocco singolo (1024 a 48 kHz) ha bin da ~47 Hz: un sub/808 a
# 25-40 Hz cade PRIMA del primo bin utile e arriva al flux solo come leakage
# parziale. Il kick detector usa quindi una finestra di BASS_FFT_BLOCKS
# blocchi (bin da ~23 Hz con 2), al costo di ~21 ms di smearing extra solo su
# questo percorso. Le ampiezze vengono riscalate di 1/BASS_FFT_BLOCKS per
# restare confrontabili col flux broadband (gate a quota).
BASS_FFT_BLOCKS = 2
# Gate di significativita' a RAMPA sulla quota di flux bassi/broadband dello
# stesso blocco. La soglia a mediana e' solo RELATIVA e su un fondo
# quasi-zero scatta su qualsiasi briciola di leakage dalle altre bande
# (misurato: burst a 1 kHz -> 44-49 falsi kick senza gate). Ma un muro secco
# alto uccide i kick veri nel mix denso: misurato su musica reale,
# con soglia 0.25 morivano 134 candidati su 166 (mediana delle quote
# reali: 0.15, con colpi a forza piena tra 0.12 e 0.20). Quindi: sotto
# BASS_SHARE_ZERO nessun onset (e niente refrattarieta' consumata), sopra
# BASS_SHARE_FULL forza piena, in mezzo la forza scala linearmente.
BASS_SHARE_ZERO = 0.08
BASS_SHARE_FULL = 0.20
# Gap refrattario minimo tra due onset consecutivi (secondi). Vincolo: deve
# restare SOPRA la finestra di smoothing del flux (2 blocchi ~ 43 ms),
# altrimenti lo stesso transiente, spalmato dalla media mobile, triggera due
# volte (verificato con smoothing a 4: a 0.06 s i click doppiavano). Con
# smoothing a 2 blocchi, 0.06 s (= 3 blocchi) rispetta il vincolo e supporta
# beat fino a ~14 Hz.
ONSET_MIN_GAP_SECONDS = 0.06
# Durata della media mobile applicata al flux prima della soglia. Convertita in
# numero di blocchi nel costruttore (round(secondi*block_rate)), invariante al
# sample rate: a 48 kHz/1024 da' 2 blocchi (0.043*46.875=2.0156 -> 2), il
# valore tarato. Riduce la varianza relativa del flux su segnale stazionario
# (dove il leakage variabile della finestra genera micro-fluttuazioni) senza
# schiacciare i transienti, che restano ordini di grandezza sopra il fondo. Una
# semplice trasformazione monotona (log, normalizzazione) non basta: la soglia
# usa la mediana, invariante per trasformazioni monotone, quindi serve un
# filtro temporale che cambi la forma della distribuzione del flux.
# VINCOLO sul valore: la finestra determina il duty cycle del flux smussato
# con beat fitti; a 4 blocchi, sopra ~7 beat/s piu' di meta' dei blocchi era
# "alta", la mediana inglobava i picchi e il rilevamento crollava a zero
# (misurato: 0% a 8-10 Hz). Con 2 blocchi il duty resta sotto il 50% fino a
# ~11 Hz. Se si rialza, riverificare lo sweep dei rate oltre ai test base.
FLUX_SMOOTHING_SECONDS = 0.043

# --- Sub-bassi (20-60 Hz, dal buffer esteso del kick) ---
# Energia sub-bassi dal buffer esteso del kick (2048 campioni, bin ~23 Hz):
# distingue il sub/808 TENUTO dal mid-bass e dal transiente di cassa. Lo spettro
# usato per il sub e' FINESTRATO (Hann) per contenere il leakage delle sidelobe
# nel range strettissimo dei sub (il sub e' tonale/tenuto, la finestra e'
# appropriata); il percorso del kick detector resta non finestrato e intatto.
SUB_LOW_HZ = 20.0
SUB_HIGH_HZ = 60.0
# AGC dedicato al sub: picco a inseguimento (stesso schema delle bande) col
# pavimento incrociato sull'energia della banda bassi 20-150 Hz (riusa la
# maschera del kick), cosi' un sub assente non si auto-normalizza a saturazione.
SUB_PEAK_DECAY_SECONDS = 3.0
SUB_PEAK_FLOOR_SHARE = 0.1
# Envelope asimmetrico del sub: attack moderato, RELEASE LENTA. Il sub e' tenuto
# (non transiente): una release lenta lo mantiene acceso tra un colpo e l'altro
# invece di seguirne l'inviluppo istantaneo. Costanti di tempo in secondi,
# convertite in coefficienti per blocco (invarianti al sample rate e all'hop).
SUB_ATTACK_SECONDS = 0.08
SUB_RELEASE_SECONDS = 0.5

# --- Warmth (temperatura spettrale, per il bend caldo/freddo della palette) ---
# Centroide spettrale mappato in [-1, 1]: a WARMTH_CENTROID_WARM_HZ o sotto il
# suono e' "caldo" (+1, bassi/scuro), a WARMTH_CENTROID_COLD_HZ o sopra e'
# "freddo" (-1, brillante), lineare in mezzo. Range scelto sul centroide
# tipico di un mix pieno (~1-3 kHz).
WARMTH_CENTROID_WARM_HZ = 800.0
WARMTH_CENTROID_COLD_HZ = 3000.0
# Tempo caratteristico dello smoothing (secondi): il warmth e' un carattere
# della sezione, non deve inseguire i transienti; si assesta in ~10-15 s.
WARMTH_SMOOTHING_SECONDS = 10.0
# Somma spettrale minima per considerare definito il centroide: sotto
# (silenzio) il warmth decade dolcemente verso il neutro 0.
WARMTH_SPECTRUM_FLOOR = 1e-6

# --- Energia di sezione (per la dinamica della palette) ---
# Misura lenta di quanto e' "carica" la sezione corrente del brano, calcolata
# sull'energia GREZZA del blocco (pre-AGC): l'AGC delle bande si riadatta in
# ~PEAK_DECAY_SECONDS e dopo pochi secondi farebbe sembrare piena anche una
# sezione quieta lunga. Nota: dipende dal volume di sistema, un cambio di
# volume a meta' brano viene letto come cambio di sezione (accettato).
# Tempo caratteristico dello smoothing dell'energia grezza (secondi).
SECTION_SMOOTHING_SECONDS = 2.0
# Tempo perche' il picco lento di riferimento scenda a ~37% (secondi).
SECTION_PEAK_DECAY_SECONDS = 60.0
# Pavimento del picco lento: evita divisioni per zero col silenzio.
SECTION_PEAK_FLOOR = 1e-9


@dataclass
class AudioFeatures:
    """Feature audio estratte da un singolo blocco.

    Attributes:
        bass: Energia normalizzata della banda bassi in [0, 1].
        mid: Energia normalizzata della banda medi in [0, 1].
        high: Energia normalizzata della banda alti in [0, 1].
        onset_strength: Intensita' dell'onset nel blocco corrente in [0, 1];
            0.0 significa nessun onset rilevato.
        bass_onset_strength: Come onset_strength ma calcolato sul flux della
            SOLA banda bassi (20-150 Hz): il "kick detector". Fa pompare i
            canali a dominanza bassi sul colpo di cassa e sui cambi della
            linea di basso, che il rilevatore broadband diluisce.
        high_onset_strength: Come onset_strength ma calcolato sul flux della
            SOLA banda alti (2-8 kHz): rileva snare e hi-hat (colpi in gamma
            alta) che il rilevatore broadband diluisce. Pensato per i canali
            col ruolo high/treble; 0.0 significa nessun onset alto rilevato.
        section_energy: Energia di sezione in [0, 1]: energia grezza del blocco
            lisciata (~2 s) e normalizzata su un picco lento (~60 s). Distingue
            sezioni quiete e drop di un brano; 0.0 col silenzio.
        flux: Inviluppo di onset continuo in [0, 1]: spectral flux smussato,
            normalizzato sul picco del flux. A differenza di onset_strength
            (impulsivo, con soglia e refrattarieta') e' un segnale denso a ogni
            blocco, adatto al beat tracking.
        warmth: Temperatura spettrale in [-1, 1] (centroide lisciato ~10 s):
            +1 = suono scuro/caldo (bassi dominanti), -1 = brillante/freddo,
            0 = neutro o silenzio. Guida il bend caldo/freddo della palette.
        voice_level: Presenza vocale stimata in [0, 1] (euristica tier 1:
            energia in banda vocale x armonicita'). Pilota i canali col ruolo
            "voice".
        harmony_hue: Hue in [0, 1) della tonalita' dominante (chroma mappato sul
            circolo delle quinte), oppure None se la tonalita' non e' definita
            (silenzio, profilo piatto, energia/dominanza sotto soglia).
        harmony_strength: Dominanza lisciata della classe di altezza piu' forte
            in [0, 1] (quanto la tonalita' emerge); 0.0 sul silenzio.
        percussive_level: Livello dell'energia PERCUSSIVA (transienti larghi in
            frequenza) in [0, 1] dalla separazione HPSS; 0.0 sul silenzio.
        harmonic_level: Livello dell'energia ARMONICA (toni tenuti, stretti in
            frequenza) in [0, 1] dalla separazione HPSS; 0.0 sul silenzio.
        sub_level: Energia sub-bassi 20-60 Hz in [0, 1] con AGC a release lenta
            (il sub e' tenuto): distingue il sub/808 tenuto dal mid-bass e dal
            transiente di cassa; 0.0 sul silenzio.
    """

    bass: float
    mid: float
    high: float
    onset_strength: float
    section_energy: float = 0.0
    flux: float = 0.0
    bass_onset_strength: float = 0.0
    high_onset_strength: float = 0.0
    warmth: float = 0.0
    voice_level: float = 0.0
    harmony_hue: float | None = None
    harmony_strength: float = 0.0
    percussive_level: float = 0.0
    harmonic_level: float = 0.0
    sub_level: float = 0.0


class _FluxOnsetDetector:
    """Rilevatore di onset via spectral flux su un sottoinsieme di bin rFFT.

    Incapsula lo stato del rilevamento (spettro precedente, media mobile,
    storia per la soglia adattiva, picco, refrattarieta') cosi' da poterne
    avere piu' istanze indipendenti: una broadband (punch/beat tracking) e
    una sulla sola banda bassi (kick detector). Le costanti FLUX_* e la
    logica sono condivise; cambia solo il sottoinsieme di bin in ingresso.
    """

    def __init__(
        self,
        refractory_blocks: int,
        peak_decay_per_block: float,
        smoothing_blocks: int,
        history_blocks: int,
    ) -> None:
        """Inizializza lo stato del rilevatore.

        Args:
            refractory_blocks: Blocchi minimi tra due onset consecutivi.
            peak_decay_per_block: Fattore di decadimento del picco per blocco.
            smoothing_blocks: Lunghezza della media mobile del flux, in blocchi
                (derivata da FLUX_SMOOTHING_SECONDS dal chiamante).
            history_blocks: Lunghezza della storia del flux per la mediana
                adattiva, in blocchi (derivata da FLUX_HISTORY_SECONDS).
        """
        self._prev: np.ndarray | None = None
        self._window: deque[float] = deque(maxlen=smoothing_blocks)
        self._history: deque[float] = deque(maxlen=history_blocks)
        self._peak: float = FLUX_FLOOR
        self._refractory_blocks: int = refractory_blocks
        self._blocks_since: int = refractory_blocks
        self._peak_decay: float = peak_decay_per_block

    def process(self, magnitudes: np.ndarray, min_flux: float = 0.0) -> tuple[float, float, float]:
        """Aggiorna il rilevatore con le ampiezze del blocco corrente.

        Spectral flux half-wave rectified rispetto al blocco precedente,
        media mobile breve, soglia adattiva sulla mediana della storia,
        normalizzazione sul picco e refrattarieta' temporale.

        Args:
            magnitudes: Ampiezze rFFT (non finestrate) dei bin di interesse.
            min_flux: Gate assoluto: sotto questo valore di flux smussato
                nessun onset (0.0 = disattivato). Valutato PRIMA della
                refrattarieta', cosi' un trigger gateato non consuma il gap.

        Returns:
            Tripla `(onset_strength, flux_normalizzato, flux_smussato)`: i
            primi due in [0, 1] (strength impulsivo, 0.0 se nessun onset;
            inviluppo continuo), il terzo in unita' grezze (per gate esterni).
        """
        # Al primo blocco non c'e' spettro precedente: flux = 0.
        if self._prev is None:
            raw_flux: float = 0.0
        else:
            diff: np.ndarray = magnitudes - self._prev
            raw_flux = float(np.sum(np.maximum(diff, 0.0)))
        self._prev = magnitudes

        # Media mobile del flux grezzo: riduce la varianza relativa del flux su
        # segnale stazionario senza abbattere i transienti.
        self._window.append(raw_flux)
        flux: float = float(np.mean(self._window))

        # Aggiorna il picco del flux con lo stesso decay dell'AGC.
        self._peak = max(self._peak * self._peak_decay, flux, FLUX_FLOOR)

        # Avanza il contatore di refrattarieta' prima di valutare l'onset.
        self._blocks_since += 1

        # Soglia adattiva: valida solo con storia sufficiente. Registra il flux
        # in storia dopo aver letto la mediana, cosi' la soglia del blocco
        # corrente dipende solo dai valori passati.
        strength: float = 0.0
        if len(self._history) >= FLUX_MIN_HISTORY and flux >= min_flux:
            median: float = float(np.median(self._history))
            threshold: float = FLUX_THRESHOLD_MULTIPLIER * median + FLUX_FLOOR
            if flux > threshold and self._blocks_since >= self._refractory_blocks:
                denom: float = max(self._peak - threshold, FLUX_FLOOR)
                strength = min(max((flux - threshold) / denom, 0.0), 1.0)
                if strength > 0.0:
                    self._blocks_since = 0

        self._history.append(flux)
        # Inviluppo continuo (per il beat tracking): flux smussato normalizzato
        # sul suo picco (il picco ha gia' il pavimento FLUX_FLOOR).
        flux_normalized: float = min(max(flux / self._peak, 0.0), 1.0)
        return strength, flux_normalized, flux


class FeatureAnalyzer:
    """Estrae bande di energia e onset da un flusso audio a blocchi.

    Per ogni blocco applica finestra di Hann e rFFT, poi:

    - misura l'energia RMS in tre bande (bass/mid/high), ognuna normalizzata da
      un AGC a inseguimento di picco e lisciata con un envelope asimmetrico
      (attack/release);
    - calcola lo spectral flux half-wave rectified rispetto al blocco precedente
      e ne deriva un `onset_strength` con soglia adattiva (mediana della storia
      recente), normalizzazione sul picco del flux e refrattarieta' temporale.

    Attributes:
        sample_rate: Frequenza di campionamento in Hz.
        block_size: Numero di campioni per blocco.
    """

    def __init__(self, sample_rate: int, block_size: int, hop: int | None = None) -> None:
        """Inizializza l'analizzatore e precomputa finestra, maschere e decay.

        Args:
            sample_rate: Frequenza di campionamento del segnale in Hz.
            block_size: Numero di campioni attesi in ogni blocco passato a
                `process` (larghezza della finestra FFT).
            hop: Passo di avanzamento tra due blocchi consecutivi, in campioni.
                None (default) equivale a `block_size`: blocchi disgiunti, cioe'
                il comportamento storico. Con `block_size // 2` le finestre si
                sovrappongono al 50% e il block rate raddoppia; i buffer che
                accumulano campioni tra blocchi (kick, ACF vocale) appendono
                allora solo la coda nuova `block[-hop:]`, mentre le FFT per
                blocco continuano a usare la finestra intera.
        """
        self.sample_rate: int = sample_rate
        self.block_size: int = block_size
        # Hop effettivo: default block_size (blocchi disgiunti = comportamento
        # storico). block_rate e durata del blocco derivano dall'hop, cosi' tutte
        # le costanti temporali restano invarianti in secondi reali.
        self._hop: int = block_size if hop is None else hop

        # Finestra di Hann precomputata (una sola allocazione).
        self._window: np.ndarray = np.hanning(block_size).astype(np.float32)

        # Frequenza di ogni bin rFFT e maschere di banda precomputate.
        freqs: np.ndarray = np.fft.rfftfreq(block_size, d=1.0 / sample_rate)
        self._bass_mask: np.ndarray = (freqs >= BASS_LOW_HZ) & (freqs <= BASS_HIGH_HZ)
        self._mid_mask: np.ndarray = (freqs >= MID_LOW_HZ) & (freqs <= MID_HIGH_HZ)
        self._high_mask: np.ndarray = (freqs >= HIGH_LOW_HZ) & (freqs <= HIGH_HIGH_HZ)
        # Maschera dei bin usati per il flux (fino a FLUX_MAX_HZ).
        self._flux_mask: np.ndarray = freqs <= FLUX_MAX_HZ

        # Fattore di decadimento del picco per blocco, condiviso da tutti gli
        # AGC (bande e flux): decay = exp(-durata_blocco / PEAK_DECAY_SECONDS).
        # La durata del blocco e' il passo dell'hop (intervallo reale tra due
        # chiamate a process), non la larghezza della finestra: cosi' il decay
        # per secondo resta invariante quando l'hop dimezza (rate doppio).
        block_duration: float = self._hop / sample_rate
        self._peak_decay_per_block: float = float(np.exp(-block_duration / PEAK_DECAY_SECONDS))

        # Freeze dell'AGC: durante un breakdown i picchi di banda
        # NON devono decadere, altrimenti l'AGC si riadatta al quasi-silenzio
        # (PEAK_DECAY_SECONDS = 3 s) e al drop le bande saturano subito,
        # appiattendo l'esplosione. Il flag congela solo il DECAY dei picchi
        # banda (i picchi possono ancora salire); onset e section_energy
        # restano vivi (servono proprio a rilevare breakdown e drop).
        self._agc_hold: bool = False

        # AGC delle bande: picco per banda + picco condiviso come pavimento
        # incrociato (vedi BAND_PEAK_FLOOR_SHARE), con envelope asimmetrico
        # indipendente per ciascuna banda.
        self._band_peak: float = PEAK_FLOOR
        self._bass_peak: float = PEAK_FLOOR
        self._mid_peak: float = PEAK_FLOOR
        self._high_peak: float = PEAK_FLOOR
        self._bass_out: float = OUTPUT_MIN
        self._mid_out: float = OUTPUT_MIN
        self._high_out: float = OUTPUT_MIN

        # Presenza vocale (euristica tier 1, vedi huebeat.voice). Riceve lo
        # stesso hop: il suo buffer ACF deve appendere solo la coda nuova.
        self._voice = VoiceDetector(sample_rate, block_size, self._hop)

        # Warmth: centroide spettrale lisciato lentamente (vedi WARMTH_*).
        self._warmth: float = 0.0
        self._warmth_coeff: float = 1.0 - float(np.exp(-block_duration / WARMTH_SMOOTHING_SECONDS))
        self._centroid_freqs: np.ndarray = freqs[self._flux_mask]

        # Energia di sezione: smoothing esponenziale dell'energia grezza del
        # blocco e picco lento di riferimento (vedi costanti SECTION_*).
        self._section_smooth: float = 0.0
        self._section_peak: float = SECTION_PEAK_FLOOR
        self._section_smooth_coeff: float = 1.0 - float(
            np.exp(-block_duration / SECTION_SMOOTHING_SECONDS)
        )
        self._section_peak_decay: float = float(
            np.exp(-block_duration / SECTION_PEAK_DECAY_SECONDS)
        )

        # Finestra di smoothing e storia del flux derivate dal tempo (invarianti
        # al sample rate E all'hop): round(secondi*block_rate) con block_rate =
        # sample_rate/hop. A 48 kHz/1024 senza hop danno 2 e 43 (i valori
        # tarati); con hop 512 (block_rate 93.75) danno 4 e 86 (stessa durata in
        # secondi). maxlen deve restare >= 1 anche a block rate estremi.
        block_rate: float = sample_rate / self._hop
        flux_smoothing_blocks: int = max(1, int(round(FLUX_SMOOTHING_SECONDS * block_rate)))
        flux_history_blocks: int = max(1, int(round(FLUX_HISTORY_SECONDS * block_rate)))

        # Rilevatori di onset su spettro NON finestrato (vedi process): uno
        # broadband (punch globale + inviluppo per il beat tracking), uno sulla
        # sola banda bassi (kick detector per i canali a dominanza bassi) e uno
        # sulla banda alti (snare/hi-hat per i canali high/treble). Refrattarieta'
        # convertita in numero di blocchi da saltare. L'onset alti parte dalle
        # STESSE costanti del broadband (refrattarieta' e smoothing/storia).
        refractory_blocks: int = int(round(ONSET_MIN_GAP_SECONDS / block_duration))
        bass_refractory_blocks: int = int(round(BASS_ONSET_MIN_GAP_SECONDS / block_duration))
        self._onset_broadband = _FluxOnsetDetector(
            refractory_blocks,
            self._peak_decay_per_block,
            flux_smoothing_blocks,
            flux_history_blocks,
        )
        self._onset_bass = _FluxOnsetDetector(
            bass_refractory_blocks,
            self._peak_decay_per_block,
            flux_smoothing_blocks,
            flux_history_blocks,
        )
        self._onset_high = _FluxOnsetDetector(
            refractory_blocks,
            self._peak_decay_per_block,
            flux_smoothing_blocks,
            flux_history_blocks,
        )

        # Finestra estesa per il kick detector (vedi BASS_FFT_BLOCKS): ring
        # buffer degli ultimi blocchi e maschera 20-150 Hz sui bin fitti.
        bass_fft_size: int = block_size * BASS_FFT_BLOCKS
        self._bass_buffer: np.ndarray = np.zeros(bass_fft_size, dtype=np.float32)
        bass_freqs: np.ndarray = np.fft.rfftfreq(bass_fft_size, d=1.0 / sample_rate)
        self._bass_flux_mask: np.ndarray = (bass_freqs >= BASS_LOW_HZ) & (
            bass_freqs <= BASS_HIGH_HZ
        )

        # Sub-bassi (20-60 Hz) dallo STESSO buffer esteso del kick, ma su spettro
        # FINESTRATO (Hann sul buffer 2048) per contenere il leakage; il kick
        # resta non finestrato. La maschera di riferimento del pavimento
        # incrociato e' quella 20-150 Hz del kick (self._bass_flux_mask).
        self._bass_window: np.ndarray = np.hanning(bass_fft_size).astype(np.float32)
        self._sub_mask: np.ndarray = (bass_freqs >= SUB_LOW_HZ) & (bass_freqs <= SUB_HIGH_HZ)
        self._sub_peak: float = PEAK_FLOOR
        self._sub_ref_peak: float = PEAK_FLOOR
        self._sub_out: float = OUTPUT_MIN
        self._sub_peak_decay: float = float(np.exp(-block_duration / SUB_PEAK_DECAY_SECONDS))
        self._sub_attack: float = 1.0 - float(np.exp(-block_duration / SUB_ATTACK_SECONDS))
        self._sub_release: float = 1.0 - float(np.exp(-block_duration / SUB_RELEASE_SECONDS))

        # Chroma (tonalita' dominante -> hue) e HPSS lite (armonico/percussivo):
        # riusano le magnitudo FINESTRATE gia' calcolate in process (nessuna FFT
        # nuova); ricevono lo stesso hop per le loro costanti di tempo.
        self._chroma = ChromaAnalyzer(sample_rate, block_size, self._hop)
        self._hpss = HpssAnalyzer(sample_rate, block_size, self._hop)

    def set_agc_hold(self, hold: bool) -> None:
        """Congela (o scongela) il decadimento dei picchi AGC delle bande.

        Usato dal motore durante i breakdown rilevati: i picchi non
        decadono verso il quasi-silenzio, cosi' al drop successivo le bande
        rientrano con l'headroom della sezione piena invece di saturare al
        primo colpo. I picchi possono comunque SALIRE; onset, flux e
        section_energy non sono toccati. Assegnazione atomica: sicuro da
        chiamare dal thread del loop.

        Args:
            hold: True per congelare il decay dei picchi banda.
        """
        self._agc_hold = hold

    def process(self, block: "np.ndarray") -> AudioFeatures:
        """Elabora un blocco audio e restituisce le feature estratte.

        Args:
            block: Array float32 mono con valori in [-1, 1] e lunghezza pari a
                `block_size`.

        Returns:
            Le feature del blocco: energie di banda e intensita' dell'onset,
            tutte in [0.0, 1.0].
        """
        # 1. rFFT finestrata (Hann) per le bande. Per l'onset si usa invece uno
        # spettro non finestrato (vedi _process_onset): la finestra di Hann
        # azzera i campioni ai bordi del blocco, rendendo quasi invisibili i
        # transienti che vi cadono, il che e' controproducente per il flux.
        spectrum: np.ndarray = np.fft.rfft(block * self._window)
        magnitudes: np.ndarray = np.abs(spectrum)

        # 2. Energia RMS delle tre bande e picco AGC condiviso.
        bass_energy: float = self._band_energy(magnitudes, self._bass_mask)
        mid_energy: float = self._band_energy(magnitudes, self._mid_mask)
        high_energy: float = self._band_energy(magnitudes, self._high_mask)
        band_decay: float = 1.0 if self._agc_hold else self._peak_decay_per_block
        self._band_peak = max(
            self._band_peak * band_decay,
            bass_energy,
            mid_energy,
            high_energy,
            PEAK_FLOOR,
        )

        # Normalizzazione per banda (pavimento incrociato) + envelope.
        bass: float = self._smooth_band(bass_energy, "_bass_peak", "_bass_out")
        mid: float = self._smooth_band(mid_energy, "_mid_peak", "_mid_out")
        high: float = self._smooth_band(high_energy, "_high_peak", "_high_out")

        # 3. Onset via spectral flux half-wave rectified su spettro non
        # finestrato: broadband (punch/beat) + banda bassi (kick detector).
        # Il kick detector e' gateato sulla frazione del flux broadband dello
        # stesso blocco: senza gate il leakage spettrale dei transienti di
        # altre bande dentro i 3 bin bassi supera la soglia a mediana, che su
        # un fondo quasi-zero e' solo relativa (misurato: burst a 1 kHz ->
        # 44 falsi kick in 10 s).
        onset_magnitudes: np.ndarray = np.abs(np.fft.rfft(block))
        onset_strength, flux, broad_flux_raw = self._onset_broadband.process(
            onset_magnitudes[self._flux_mask]
        )
        # Kick detector sulla finestra estesa (ring buffer degli ultimi
        # BASS_FFT_BLOCKS blocchi): bin abbastanza fitti da vedere i sub sotto
        # i ~47 Hz. Ampiezze riscalate per il gate a quota (vedi costante).
        # Il ring buffer AVANZA di hop e appende SOLO la coda nuova del blocco
        # (block[-hop:]): con l'overlap i blocchi consecutivi condividono
        # campioni, appendere l'intero blocco li duplicherebbe. Con hop =
        # block_size (default) equivale a shiftare di un blocco intero.
        hop = self._hop
        self._bass_buffer[:-hop] = self._bass_buffer[hop:]
        self._bass_buffer[-hop:] = block[-hop:]
        bass_magnitudes: np.ndarray = (
            np.abs(np.fft.rfft(self._bass_buffer))[self._bass_flux_mask] / BASS_FFT_BLOCKS
        )
        bass_onset_strength, _bass_flux, bass_flux_raw = self._onset_bass.process(
            bass_magnitudes,
            min_flux=BASS_SHARE_ZERO * broad_flux_raw,
        )
        if bass_onset_strength > 0.0:
            # Rampa sulla quota bassi/broadband: kick nel mix denso attenuato
            # ma visibile, leakage debole quasi azzerato (vedi BASS_SHARE_*).
            share: float = bass_flux_raw / max(broad_flux_raw, FLUX_FLOOR)
            ramp: float = min(
                max((share - BASS_SHARE_ZERO) / (BASS_SHARE_FULL - BASS_SHARE_ZERO), 0.0), 1.0
            )
            bass_onset_strength *= ramp

        # Onset sulla sola banda alti (snare/hi-hat): stesse costanti del
        # broadband, alimentato dai bin gia' mascherati della banda high sullo
        # spettro NON finestrato. Nessun gate: la banda alti ha contenuto vero
        # (non e' quasi-vuota come i 3 bin del kick), quindi la soglia a mediana
        # non scatta sul leakage.
        high_onset_strength, _high_flux, _high_flux_raw = self._onset_high.process(
            onset_magnitudes[self._high_mask]
        )

        # 4. Sub-bassi 20-60 Hz dal buffer esteso del kick (gia' aggiornato qui
        # sopra), su spettro finestrato, con AGC a release lenta.
        sub_level: float = self._process_sub()

        # 5. Energia di sezione dall'energia grezza del blocco (pre-AGC).
        section_energy: float = self._process_section_energy(block)

        # 6. Warmth: centroide spettrale (finestrato, <= FLUX_MAX_HZ) lisciato.
        warmth: float = self._process_warmth(magnitudes)

        # 7. Presenza vocale (energia banda vocale x armonicita'). Le magnitudo
        # finestrate (Hann) sono gia' calcolate qui sopra con la STESSA finestra
        # e block_size del VoiceDetector: gliele passiamo per l'energia in banda
        # (il percorso ACF resta interno al detector, sul suo buffer lungo).
        voice_level: float = self._voice.process(block, magnitudes=magnitudes)

        # 8. Chroma (tonalita' -> hue) e HPSS lite (armonico/percussivo): entrambi
        # sulle magnitudo FINESTRATE gia' calcolate (nessuna FFT nuova).
        harmony_hue, harmony_strength = self._chroma.process(magnitudes)
        percussive_level, harmonic_level = self._hpss.process(magnitudes)

        return AudioFeatures(
            bass=bass,
            mid=mid,
            high=high,
            onset_strength=onset_strength,
            section_energy=section_energy,
            flux=flux,
            bass_onset_strength=bass_onset_strength,
            high_onset_strength=high_onset_strength,
            warmth=warmth,
            voice_level=voice_level,
            harmony_hue=harmony_hue,
            harmony_strength=harmony_strength,
            percussive_level=percussive_level,
            harmonic_level=harmonic_level,
            sub_level=sub_level,
        )

    def _process_warmth(self, magnitudes: np.ndarray) -> float:
        """Aggiorna e restituisce la temperatura spettrale del blocco.

        Centroide spettrale sulle ampiezze finestrate fino a FLUX_MAX_HZ,
        mappato linearmente in [-1, 1] tra WARMTH_CENTROID_WARM_HZ (+1) e
        WARMTH_CENTROID_COLD_HZ (-1), poi lisciato con costante di tempo
        WARMTH_SMOOTHING_SECONDS. Col silenzio il target e' 0 (neutro).

        Args:
            magnitudes: Ampiezze rFFT finestrate del blocco corrente.

        Returns:
            Warmth corrente in [-1, 1].
        """
        spectrum = magnitudes[self._flux_mask]
        total = float(np.sum(spectrum))
        if total > WARMTH_SPECTRUM_FLOOR:
            centroid = float(np.sum(self._centroid_freqs * spectrum) / total)
            span = WARMTH_CENTROID_COLD_HZ - WARMTH_CENTROID_WARM_HZ
            position = (centroid - WARMTH_CENTROID_WARM_HZ) / span
            raw = 1.0 - 2.0 * min(max(position, 0.0), 1.0)
        else:
            raw = 0.0
        self._warmth += (raw - self._warmth) * self._warmth_coeff
        return float(min(max(self._warmth, -1.0), 1.0))

    def _process_section_energy(self, block: "np.ndarray") -> float:
        """Aggiorna e restituisce l'energia di sezione del blocco corrente.

        Liscia l'energia media del blocco (tempo caratteristico
        `SECTION_SMOOTHING_SECONDS`) e la normalizza su un picco a decadimento
        lento (`SECTION_PEAK_DECAY_SECONDS`): il rapporto distingue le sezioni
        quiete dai drop senza che la normalizzazione "dimentichi" troppo in
        fretta come fa l'AGC delle bande.

        Args:
            block: Blocco audio mono float32 in [-1, 1].

        Returns:
            Energia di sezione in [0, 1].
        """
        raw_energy: float = float(np.mean(np.square(block)))
        self._section_smooth += (raw_energy - self._section_smooth) * self._section_smooth_coeff
        self._section_peak = max(
            self._section_peak * self._section_peak_decay,
            self._section_smooth,
            SECTION_PEAK_FLOOR,
        )
        return float(min(max(self._section_smooth / self._section_peak, 0.0), 1.0))

    def _process_sub(self) -> float:
        """Aggiorna e restituisce il livello dei sub-bassi (20-60 Hz).

        Calcola l'energia RMS 20-60 Hz sul buffer esteso del kick FINESTRATO
        (Hann sul buffer 2048, per contenere il leakage nel range strettissimo),
        la normalizza con un AGC a inseguimento di picco col pavimento incrociato
        sull'energia della banda bassi 20-150 Hz (cosi' un sub assente resta
        basso invece di auto-normalizzarsi) e la liscia con un envelope a release
        lenta (il sub e' tenuto, non transiente). Il buffer e' gia' stato
        aggiornato dal percorso del kick in `process`.

        Returns:
            Livello dei sub-bassi in [0, 1]; 0.0 sul silenzio.
        """
        windowed: np.ndarray = np.abs(np.fft.rfft(self._bass_buffer * self._bass_window))
        sub_energy: float = self._band_energy(windowed, self._sub_mask)
        ref_energy: float = self._band_energy(windowed, self._bass_flux_mask)
        self._sub_peak = max(self._sub_peak * self._sub_peak_decay, sub_energy, PEAK_FLOOR)
        self._sub_ref_peak = max(self._sub_ref_peak * self._sub_peak_decay, ref_energy, PEAK_FLOOR)
        denom: float = max(self._sub_peak, SUB_PEAK_FLOOR_SHARE * self._sub_ref_peak)
        normalized: float = min(sub_energy / denom, 1.0)
        coeff: float = self._sub_attack if normalized > self._sub_out else self._sub_release
        self._sub_out += (normalized - self._sub_out) * coeff
        return float(min(max(self._sub_out, OUTPUT_MIN), OUTPUT_MAX))

    def _band_energy(self, magnitudes: np.ndarray, mask: np.ndarray) -> float:
        """Calcola l'energia RMS delle ampiezze rFFT in una banda.

        Args:
            magnitudes: Ampiezze rFFT del blocco corrente.
            mask: Maschera booleana dei bin appartenenti alla banda.

        Returns:
            RMS delle ampiezze dei bin selezionati.
        """
        band_mags: np.ndarray = magnitudes[mask]
        return float(np.sqrt(np.mean(band_mags**2)))

    def _smooth_band(self, energy: float, peak_attr: str, out_attr: str) -> float:
        """Normalizza l'energia di una banda e la liscia con l'envelope.

        Ogni banda insegue il PROPRIO picco (arriva a 1.0 sui suoi massimi,
        anche se un'altra banda domina lo spettro), ma il denominatore non
        scende sotto BAND_PEAK_FLOOR_SHARE volte il picco condiviso: le bande
        quasi vuote (es. il leakage spettrale di un tono puro) restano basse
        invece di auto-normalizzarsi a saturazione. Lo smoothing
        attack/release usa stato per-banda indipendente.

        Args:
            energy: Energia RMS della banda per il blocco corrente.
            peak_attr: Nome dell'attributo che tiene il picco della banda.
            out_attr: Nome dell'attributo che tiene l'output lisciato.

        Returns:
            Livello normalizzato e lisciato della banda, clampato in [0, 1].
        """
        band_decay: float = 1.0 if self._agc_hold else self._peak_decay_per_block
        peak: float = max(getattr(self, peak_attr) * band_decay, energy, PEAK_FLOOR)
        setattr(self, peak_attr, peak)
        normalized: float = energy / max(peak, BAND_PEAK_FLOOR_SHARE * self._band_peak)

        out: float = getattr(self, out_attr)
        coeff: float = ATTACK if normalized > out else RELEASE
        out += (normalized - out) * coeff
        setattr(self, out_attr, out)

        return float(min(max(out, OUTPUT_MIN), OUTPUT_MAX))
