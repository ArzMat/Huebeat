"""Rilevatore di struttura build-up / breakdown / drop per huebeat.

Modulo di puro calcolo (nessun logging, nessun effetto collaterale). Consuma
per ogni blocco le feature audio (`AudioFeatures`) e lo stato del beat
(`BeatInfo`) e riconosce i momenti "da concerto" dei generi EDM/nightcore:

- BUILD: un riser, cioe' tensione in salita per qualche secondo (la densita'
  degli onset accelera e il basso viene tolto o resta stabile);
- BREAKDOWN: un crollo di energia sostenuto (la sezione va molto sotto il suo
  massimo recente);
- DROP: l'esplosione, una risalita violenta di energia/bassi con onset forte,
  che arriva da un build carico o da un breakdown.

Interfaccia pubblica: `StructureTracker`, che a ogni blocco restituisce uno
`StructureState` con fase, intensita' del build e un flag one-shot `drop_fired`
(True solo nel primo blocco in cui il drop scatta), pensato per pilotare una
coreografia al motore (buio di un beat + slam).

Filosofia (come beat.py): PRECISIONE > RECALL. Un drop mancato costa poco, un
falso blackout a meta' ritornello e' inaccettabile. Quindi ogni transizione ha
soglie con isteresi, dwell minimi e un refrattario tra i drop; su musica
STABILE (energia e densita' costanti) la fase resta "neutral" per sempre.

Nota sul warmth. In `analysis.py` warmth vale +1 = caldo/scuro (bassi
dominanti) e -1 = freddo/brillante. I riser reali di EDM/nightcore diventano
piu' BRILLANTI (sweep di rumore verso l'alto, rullate, basso rimosso -> il
centroide sale), quindi durante un riser il warmth SCENDE verso il freddo. Per
questo il build non e' gateato sul warmth (segno ambiguo/opposto rispetto a
"sale nei riser"), ma usa la coppia sign-robusta densita' onset in salita +
basso in calo/stabile; un warmth in calo (piu' brillante) fa solo accumulare
tensione piu' in fretta come conferma.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huebeat.analysis import AudioFeatures
    from huebeat.beat import BeatInfo

# --- Densita' di onset (finestra scorrevole) ---
# Finestra su cui si conta la densita' degli onset (onset al secondo). ~2 s:
# abbastanza lunga da lisciare la spaziatura dei colpi, abbastanza corta da
# seguire l'accelerazione di una rullata di riser.
DENSITY_WINDOW_SEC = 2.0
# onset_strength (broadband o kick) sopra cui il blocco conta come un evento.
# Basso ma non zero: onset_strength e' impulsivo in [0, 1], gli onset veri
# arrivano ben sopra, il fondo resta sotto.
ONSET_EVENT_MIN = 0.1

# --- BUILD ---
# Ritardo su cui si misura il trend (valore ora - valore di ~1 s fa) di
# densita', bass e warmth. 1 s da' reattivita' senza inseguire il singolo colpo.
BUILD_TREND_SEC = 1.0
# Aumento minimo della densita' (onset/s sul ritardo di trend) per dichiarare
# la densita' "in salita". 0.75 sta SOPRA le oscillazioni di un beat periodico
# stabile (un evento che entra/esce da una finestra di 2 s muove la densita' di
# ~0.5 onset/s), cosi' la musica stabile non accumula mai tensione; un riser
# reale (densita' da ~2 a ~10 onset/s) lo supera con margine.
DENSITY_RISE_EPS = 0.75
# Il basso, sul ritardo di trend, non deve salire in modo netto: deve essere in
# calo o stabile (il riser tipico TOGLIE i bassi prima del drop). Piccola
# tolleranza positiva per ammettere un basso "stabile" con rumore.
BASS_RISE_MAX = 0.02
# Calo di warmth (centroide che si sposta sul brillante) che conferma il riser.
WARMTH_FALL_EPS = 0.01
# Passo di accumulo per blocco quando il trend e' in salita, piu' il bonus se
# anche il warmth conferma il riser brillante.
BUILD_RISE_STEP = 1.0
BUILD_WARMTH_BONUS = 0.5
# Decadimento dell'accumulo per blocco quando il trend muore: piu' rapido della
# salita (2x) cosi' "decade se il trend muore" senza restare appeso.
BUILD_DECAY = 2.0
# Accumulo netto di salita (in secondi equivalenti) per ENTRARE in build: ~2 s
# di trend positivo persistente (vincolo del task).
BUILD_ENTER_SEC = 2.0
# Accumulo di saturazione: build_intensity arriva a 1.0 dopo ~un riser pieno.
BUILD_ACCUM_MAX_SEC = 4.0
# Sotto questo accumulo si esce dal build (isteresi: EXIT < ENTER -> niente
# sfarfallio di fase attorno alla soglia).
BUILD_EXIT_SEC = 1.0
# Intensity minima perche' un build possa esplodere in un drop. Con ENTER=2 s e
# ACCUM_MAX=4 s, appena entrato in build l'intensity vale gia' ~0.5, quindi
# qualunque build stabilizzato puo' droppare (0.35 lascia margine).
DROP_BUILD_INTENSITY_MIN = 0.35

# --- BREAKDOWN ---
# Finestra del massimo recente di section_energy (~10 s): riferimento contro cui
# si misura il crollo.
SECTION_ROLLMAX_SEC = 10.0
# section_energy sotto questa frazione del massimo recente = crollo di energia.
# 0.35 e' un vero "crollo" (una sezione quieta o un vuoto pre-drop), non un
# semplice calo di dinamica.
BREAKDOWN_ENERGY_FRACTION = 0.35
# Durata minima del crollo sostenuto per dichiarare breakdown (>= ~0.7 s): un
# singolo buco tra due colpi non basta.
BREAKDOWN_MIN_SEC = 0.7
# Il massimo recente deve superare questa soglia perche' abbia senso parlare di
# crollo: distingue un breakdown (si veniva da energia vera) dal silenzio, dove
# il rapporto e' indefinito. Col silenzio la fase resta neutral, mai breakdown.
SECTION_ACTIVE_MIN = 0.08

# --- DROP ---
# Finestra su cui si misura il "salto" di energia/bassi (< ~0.5 s: una risalita
# violenta, non una salita graduale).
DROP_RISE_SEC = 0.5
# Salto minimo di section_energy o di bass sul ritardo di risalita per un drop.
DROP_SECTION_JUMP = 0.35
DROP_BASS_JUMP = 0.35
# Onset (broadband o kick) minimo sul blocco del drop: l'esplosione ha sempre un
# colpo forte, non e' solo una rampa di energia.
DROP_ONSET_MIN = 0.3
# Gap refrattario minimo tra due drop (>= 8 s): dopo uno scatto, il salto di
# energia resta alto per parecchi blocchi e senza refrattario ri-scatterebbe a
# ogni blocco; e due esplosioni ravvicinate sono un solo evento coreografico.
REFRACTORY_SEC = 8.0
# Durata della fase "drop" dopo lo scatto (dwell): tiene la fase stabile per la
# coreografia prima di tornare a valutare build/breakdown.
DROP_DWELL_SEC = 1.0


@dataclass(frozen=True)
class StructureState:
    """Stato strutturale del brano per un blocco.

    Attributes:
        phase: Fase corrente: "neutral" | "build" | "breakdown" | "drop".
        build_intensity: Tensione accumulata in [0, 1]; 0.0 fuori dal build.
        drop_fired: True SOLO nel primo blocco in cui un drop scatta (one-shot);
            resta False nei blocchi successivi dello stesso drop e durante il
            refrattario.
    """

    phase: str
    build_intensity: float
    drop_fired: bool


class StructureTracker:
    """Macchina a stati build / breakdown / drop da feature audio per blocco.

    Mantiene finestre scorrevoli su densita' di onset, bass, warmth e
    section_energy e, a ogni `update`, decide la fase con precedenza
    drop > breakdown > build > neutral. Il build accumula tensione finche' la
    densita' sale e il basso non sale (isteresi ENTER/EXIT); il breakdown
    scatta su un crollo di energia sostenuto; il drop richiede un salto violento
    di energia/bassi con onset forte, partendo da un build carico o da un
    breakdown, con `drop_fired` one-shot e refrattario tra i drop.

    Attributes:
        block_rate_hz: Frequenza dei blocchi in Hz (chiamate a `update` al
            secondo).
    """

    def __init__(self, block_rate_hz: float) -> None:
        """Inizializza il tracker per un dato block rate.

        Args:
            block_rate_hz: Frequenza dei blocchi in Hz (es. 46.875 per blocchi
                da 1024 campioni a 48 kHz). Deve essere positiva.

        Raises:
            ValueError: Se `block_rate_hz` non e' positiva.
        """
        if block_rate_hz <= 0.0:
            raise ValueError(f"block_rate_hz deve essere positiva, ricevuto {block_rate_hz}")
        self.block_rate_hz: float = block_rate_hz

        # Conteggi in blocchi derivati dalle costanti in secondi.
        density_window_blocks: int = self._blocks(DENSITY_WINDOW_SEC)
        trend_blocks: int = self._blocks(BUILD_TREND_SEC)
        rollmax_blocks: int = self._blocks(SECTION_ROLLMAX_SEC)
        drop_rise_blocks: int = self._blocks(DROP_RISE_SEC)

        self._build_enter: float = float(self._blocks(BUILD_ENTER_SEC))
        self._build_exit: float = float(self._blocks(BUILD_EXIT_SEC))
        self._build_accum_max: float = float(self._blocks(BUILD_ACCUM_MAX_SEC))
        self._breakdown_min_blocks: int = self._blocks(BREAKDOWN_MIN_SEC)
        self._refractory_blocks: int = self._blocks(REFRACTORY_SEC)
        self._drop_dwell_blocks: int = self._blocks(DROP_DWELL_SEC)
        # Warm-up: prima che densita' e suo riferimento di trend siano riempiti,
        # la densita' cresce solo perche' la finestra si sta riempiendo (falso
        # trend positivo). Fino ad allora si resta neutral, senza accumulare.
        self._warmup_blocks: int = density_window_blocks + trend_blocks

        # Finestre scorrevoli e storie di trend.
        self._onset_window: deque[int] = deque(maxlen=density_window_blocks)
        self._density_hist: deque[float] = deque(maxlen=trend_blocks + 1)
        self._bass_trend_hist: deque[float] = deque(maxlen=trend_blocks + 1)
        self._warmth_trend_hist: deque[float] = deque(maxlen=trend_blocks + 1)
        self._section_short: deque[float] = deque(maxlen=drop_rise_blocks + 1)
        self._bass_short: deque[float] = deque(maxlen=drop_rise_blocks + 1)
        self._section_rollmax: deque[float] = deque(maxlen=rollmax_blocks)

        # Stato della macchina.
        self._block_index: int = -1
        self._phase: str = "neutral"
        self._build_accum: float = 0.0
        self._breakdown_low_blocks: int = 0
        self._blocks_since_drop: int = self._refractory_blocks
        self._drop_hold: int = 0

    def reset(self) -> None:
        """Azzera finestre, accumulatori e fase: riparte come appena costruito.

        Da chiamare al cambio brano: senza reset il rolling-max e il refrattario
        del brano precedente restano (intro del brano nuovo letta come breakdown,
        primo drop soppresso dal refrattario ereditato). Le finestre scorrevoli
        vengono svuotate (mantengono il loro maxlen), gli accumulatori tornano a
        zero, il refrattario riparte "scaduto" (drop possibile subito) e la fase
        torna a neutral. Le costanti derivate dal block rate restano.
        """
        self._onset_window.clear()
        self._density_hist.clear()
        self._bass_trend_hist.clear()
        self._warmth_trend_hist.clear()
        self._section_short.clear()
        self._bass_short.clear()
        self._section_rollmax.clear()
        self._block_index = -1
        self._phase = "neutral"
        self._build_accum = 0.0
        self._breakdown_low_blocks = 0
        self._blocks_since_drop = self._refractory_blocks
        self._drop_hold = 0

    def update(self, features: AudioFeatures, beat: BeatInfo) -> StructureState:
        """Consuma le feature di un blocco e restituisce lo stato strutturale.

        Va chiamata una volta per blocco. Il parametro `beat` fa parte
        dell'interfaccia (il motore lavora a livello di beat per la coreografia);
        la decisione di fase qui non ne dipende, cosi' resta robusta anche
        quando il beat non e' lockato.

        Args:
            features: Feature audio del blocco corrente.
            beat: Stato del beat tracking del blocco corrente.

        Returns:
            Lo stato strutturale per questo blocco.
        """
        del beat  # riservato al motore per l'allineamento coreografico
        self._block_index += 1
        self._blocks_since_drop += 1

        # --- Aggiornamento dei segnali interni ---
        onset_now: float = max(features.onset_strength, features.bass_onset_strength)
        self._onset_window.append(1 if onset_now > ONSET_EVENT_MIN else 0)
        density: float = sum(self._onset_window) / DENSITY_WINDOW_SEC

        self._density_hist.append(density)
        self._bass_trend_hist.append(features.bass)
        self._warmth_trend_hist.append(features.warmth)
        density_trend: float = density - self._density_hist[0]
        bass_trend: float = features.bass - self._bass_trend_hist[0]
        warmth_trend: float = features.warmth - self._warmth_trend_hist[0]

        self._section_short.append(features.section_energy)
        self._bass_short.append(features.bass)
        section_jump: float = features.section_energy - self._section_short[0]
        bass_jump: float = features.bass - self._bass_short[0]

        self._section_rollmax.append(features.section_energy)
        section_rollmax: float = max(self._section_rollmax)

        # --- Accumulo del build (aggiornato sempre, anche se la fase e' altra) ---
        warmed_up: bool = self._block_index >= self._warmup_blocks
        rising: bool = warmed_up and density_trend > DENSITY_RISE_EPS and bass_trend < BASS_RISE_MAX
        if rising:
            step: float = BUILD_RISE_STEP
            if warmth_trend < -WARMTH_FALL_EPS:
                step += BUILD_WARMTH_BONUS
            self._build_accum = min(self._build_accum + step, self._build_accum_max)
        else:
            self._build_accum = max(self._build_accum - BUILD_DECAY, 0.0)
        build_intensity: float = min(max(self._build_accum / self._build_accum_max, 0.0), 1.0)

        # --- Contatore del crollo di energia (breakdown) ---
        energy_low: bool = (
            section_rollmax >= SECTION_ACTIVE_MIN
            and features.section_energy < BREAKDOWN_ENERGY_FRACTION * section_rollmax
        )
        self._breakdown_low_blocks = self._breakdown_low_blocks + 1 if energy_low else 0
        breakdown_ready: bool = self._breakdown_low_blocks >= self._breakdown_min_blocks

        # --- Dwell del drop: tiene la fase "drop" stabile dopo lo scatto ---
        if self._drop_hold > 0:
            self._drop_hold -= 1
            self._phase = "drop"
            return StructureState(phase="drop", build_intensity=0.0, drop_fired=False)

        # --- DROP (priorita' massima): salto violento + onset forte, da build
        # carico o da breakdown, fuori dal refrattario ---
        can_drop: bool = self._phase == "breakdown" or (
            self._phase == "build" and build_intensity >= DROP_BUILD_INTENSITY_MIN
        )
        jumped: bool = section_jump >= DROP_SECTION_JUMP or bass_jump >= DROP_BASS_JUMP
        if (
            can_drop
            and jumped
            and onset_now >= DROP_ONSET_MIN
            and self._blocks_since_drop >= self._refractory_blocks
        ):
            self._blocks_since_drop = 0
            self._drop_hold = self._drop_dwell_blocks
            self._build_accum = 0.0
            self._breakdown_low_blocks = 0
            self._phase = "drop"
            return StructureState(phase="drop", build_intensity=0.0, drop_fired=True)

        # --- BREAKDOWN ---
        if breakdown_ready:
            self._phase = "breakdown"
            return StructureState(phase="breakdown", build_intensity=0.0, drop_fired=False)

        # --- BUILD (con isteresi ENTER/EXIT) ---
        if self._build_accum >= self._build_enter or (
            self._phase == "build" and self._build_accum >= self._build_exit
        ):
            self._phase = "build"
            return StructureState(phase="build", build_intensity=build_intensity, drop_fired=False)

        # --- NEUTRAL ---
        self._phase = "neutral"
        return StructureState(phase="neutral", build_intensity=0.0, drop_fired=False)

    def _blocks(self, seconds: float) -> int:
        """Converte una durata in secondi nel numero di blocchi (minimo 1).

        Args:
            seconds: Durata in secondi.

        Returns:
            Numero di blocchi corrispondente al block rate, almeno 1.
        """
        return max(1, int(round(seconds * self.block_rate_hz)))
