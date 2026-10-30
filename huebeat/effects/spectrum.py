"""Effetto "spectrum": spettro spaziale per-segmento con punch sui colpi.

L'effetto storico di huebeat: sinistra=bassi, centro=medi, destra=alti, con
un flash desaturante ("punch") caricato dagli onset. E' il default.
"""

import colorsys
from typing import TYPE_CHECKING

from huebeat import tuning
from huebeat.analysis import AudioFeatures
from huebeat.effects.base import (
    FRAME_RATE_HZ,
    ChannelInfo,
    palette_channel_base,
    path_layout,
    spatial_layout,
)

if TYPE_CHECKING:
    from huebeat.beat import BeatInfo

# --- Punch (flash sui colpi) ---
# NOTA tuning runtime: PUNCH_HALF_LIFE_SECONDS, LEVEL_GAIN, PUNCH_GAIN e
# BASS_CONTRAST_GAMMA sono i DEFAULT storici; i valori vivi arrivano da
# huebeat.tuning.live (regolabili a caldo dal frontend) e coincidono con
# questi in assenza di override.
# Tempo di dimezzamento del punch dopo un onset, in secondi.
PUNCH_HALF_LIFE_SECONDS = 0.25
# Tetto del punch accumulato (i colpi ravvicinati si sommano fino a qui).
MAX_PUNCH = 1.0
# Quanto il punch riduce la saturazione (il colpo "sbianca" il colore).
PUNCH_DESATURATION = 0.7
# Saturazione minima raggiungibile durante un punch.
MIN_SATURATION = 0.3

# --- Guadagni di luminosita' ---
# Contributo del livello di banda al value HSV.
LEVEL_GAIN = 0.85
# Contributo del punch al value HSV.
PUNCH_GAIN = 0.6

# --- Contrasto dei bassi ---
# L'AGC condiviso normalizza sul picco della banda dominante, che e' quasi
# sempre quella dei bassi: il loro livello vive schiacciato vicino a 1.0 e i
# cambi si vedono poco. La gamma riespande l'escursione visiva (con 2.0:
# 0.7 -> 0.49, 0.85 -> 0.72, 1.0 -> 1.0) senza toccare l'AGC, che serve alla
# scala relativa tra bande.
BASS_CONTRAST_GAMMA = 2.0

# Pesi di banda fissi per i ruoli forzati: (bass, mid, high). "rest" e'
# "tutto tranne i bassi": medi e alti mediati, per dividere le luci in due
# gruppi bassi / non-bassi. I ruoli "voice", "drums" e "wash" sono gestiti a
# parte: non mixano le bande, il livello arriva da una feature dedicata.
ROLE_WEIGHTS = {
    "bass": (1.0, 0.0, 0.0),
    "mid": (0.0, 1.0, 0.0),
    "high": (0.0, 0.0, 1.0),
    "rest": (0.0, 0.5, 0.5),
}

# Ruoli che pilotano il livello da una feature dedicata invece che dal mix di
# bande (pesi a zero): voice = presenza vocale, drums = componente percussiva
# HPSS, wash = componente armonica HPSS.
DIRECT_LEVEL_ROLES = ("voice", "drums", "wash")

# --- Ruolo "wash" (tappeto armonico) ---
# Mezza vita dello smoothing extra dolce del livello wash: il tappeto segue
# l'inviluppo armonico senza sfarfallii sui colpi (nessun punch su questo ruolo).
WASH_LEVEL_HALF_LIFE_SECONDS = 0.5

# --- Sub glow ---
# Quota del sub tenuto che fa da fondo di luce sui canali con peso bassi: la
# componente bass del livello non scende sotto SUB_FLOOR_SHARE * sub_level,
# cosi' un sub sostenuto tiene accesa la luce anche quando il colpo manca.
SUB_FLOOR_SHARE = 0.5

# --- Flash predittivo sulla griglia del beat ---
# Quota del contributo di griglia che arriva anche ai canali non-bassi: il
# beat previsto e' tipicamente la cassa, quindi i canali bassi lo ricevono
# pieno e gli altri attenuato.
GRID_PUNCH_NON_BASS_SHARE = 0.5


class SpectrumPunchEffect:
    """Effetto spettro spaziale con flash desaturante sui colpi.

    I canali sono ordinati per x crescente e ricevono una posizione
    normalizzata p in [0, 1] per rango: a sinistra dominano i bassi, al centro
    i medi, a destra gli alti. Ogni canale ha un gradiente di hue fisso lungo
    x; gli onset caricano un "punch" per canale che decade esponenzialmente e
    che alza la luminosita' riducendo la saturazione (flash verso il bianco).
    La sorgente del punch e' pesata: i canali a dominanza bassi seguono il
    kick detector (bass_onset_strength), gli altri l'onset broadband; il
    livello bassi ha inoltre una gamma di contrasto (BASS_CONTRAST_GAMMA)
    che riespande l'escursione schiacciata dall'AGC condiviso.

    I canali con `role` forzato ("bass" | "mid" | "high" | "rest") usano il peso
    di banda del ruolo invece della distribuzione spaziale; i ruoli a livello
    diretto ("voice" | "drums" | "wash") prendono il livello da una feature
    dedicata (presenza vocale, componente percussiva/armonica HPSS): drums
    riceve il punch pieno dal broadband, wash nessun punch e uno smoothing extra
    dolce del livello. Il COLORE di ogni ruolo resta quello della sua posizione.

    Il knob `mirror` interpola con continuita' la posizione di ogni canale tra la
    mappa spaziale di rango e la simmetria a U (i due capi del percorso = bassi,
    il fondo = alti); posizione e colore seguono lo stesso morph. Sui canali con
    peso bassi il livello include un fondo dal sub tenuto (SUB_FLOOR_SHARE).

    Lo stato interno si limita ai punch per canale, al livello wash lisciato e
    alla palette corrente.
    """

    def __init__(self, channels: list[ChannelInfo]) -> None:
        """Prepara l'effetto: ordina i canali e precomputa posizioni e decay.

        Args:
            channels: Canali dell'area con le rispettive posizioni.
        """
        self._channels, self._positions = spatial_layout(channels)
        # Posizione simmetrica a U per ogni canale (allineata a self._channels):
        # dalla posizione lungo il PERCORSO fisico (path_layout) si ricava
        # 1 - |2*p - 1|, cosi' i due CAPI del percorso (vicini all'ascoltatore)
        # valgono 0 (bassi) e il fondo vale 1 (alti). Usata dal morph "mirror".
        _, path_positions = path_layout(self._channels)
        path_by_id = {
            channel.channel_id: path_pos
            for channel, path_pos in zip(self._channels, path_positions, strict=True)
        }
        self._sym_positions: list[float] = [
            1.0 - abs(2.0 * path_by_id[channel.channel_id] - 1.0) for channel in self._channels
        ]
        # Morph mirror corrente e posizioni effettive derivate (ricalcolate in
        # render solo quando il knob cambia, non a ogni frame).
        self._mirror: float = tuning.live.mirror
        self._effective_positions: list[float] = self._compute_effective_positions(self._mirror)
        # Posizione [0, 1] usata per COLORARE ogni canale (id -> posizione),
        # esposta per lo slam del drop; segue il morph mirror come le bande.
        self.positions: dict[int, float] = self._positions_by_id()
        count = len(self._channels)
        # Punch per canale, caricato dagli onset e decaduto a ogni frame.
        self._punches: list[float] = [0.0] * count
        # Livello lisciato per i canali "wash" (tappeto armonico): smoothing
        # extra dolce, un valore per canale (gli altri ruoli lo ignorano).
        self._wash_levels: list[float] = [0.0] * count
        # Fattore di ritenzione per frame dello smoothing wash (quota del valore
        # vecchio mantenuta): mezza vita WASH_LEVEL_HALF_LIFE_SECONDS.
        self._wash_retain: float = 0.5 ** (1.0 / (WASH_LEVEL_HALF_LIFE_SECONDS * FRAME_RATE_HZ))
        # Half-life corrente e fattore di decadimento per frame derivato;
        # ricalcolato in render solo quando il knob cambia (cache).
        self._decay_half_life: float = tuning.live.punch_half_life_seconds
        self._punch_decay_per_frame: float = 0.5 ** (1.0 / (self._decay_half_life * FRAME_RATE_HZ))
        # Palette corrente per la colorazione dei canali; None = modalita'
        # default (gradiente di hue fisso). Aggiornata da set_palette.
        self._palette: list[tuple[float, float, float]] | None = None
        # Immagine stereo corrente (oggetto con pan_bass/pan_mid/pan_high in
        # [-1, 1]) o None; aggiornata dal motore con set_stereo prima di ogni
        # render (assegnazione atomica, come la palette).
        self._stereo: object | None = None

    def _compute_effective_positions(self, mirror: float) -> list[float]:
        """Interpola posizione spaziale e simmetrica a U secondo il knob mirror.

        Per ogni canale pos_eff = (1 - mirror) * spaziale + mirror * simmetrica.
        Con mirror 0 restituisce le posizioni spaziali di rango (identico a
        oggi); con mirror 1 quelle simmetriche a U.

        Args:
            mirror: Peso del morph in [0, 1] (0 = spaziale, 1 = simmetrico).

        Returns:
            Lista di posizioni effettive in [0, 1], allineata a self._channels.
        """
        return [
            (1.0 - mirror) * spatial + mirror * sym
            for spatial, sym in zip(self._positions, self._sym_positions, strict=True)
        ]

    def _positions_by_id(self) -> dict[int, float]:
        """Mappa channel_id -> posizione effettiva corrente.

        Returns:
            Dict con la posizione effettiva (dopo il morph mirror) per canale.
        """
        return {
            channel.channel_id: position
            for channel, position in zip(self._channels, self._effective_positions, strict=True)
        }

    def set_palette(self, palette: list[tuple[float, float, float]] | None) -> None:
        """Imposta la palette usata per colorare i canali (o torna al default.)

        In modalita' palette hue e saturazione base di ogni canale derivano dai
        colori RGB della palette (interpolati lungo l'asse spaziale) invece che
        dal gradiente di hue fisso; passando None (o una lista vuota) l'effetto
        torna alla colorazione di default, numericamente identica a prima.

        Thread-safety: questo metodo e' pensato per essere chiamato da un thread
        diverso da quello che esegue `render` (es. un poller di "now playing").
        L'aggiornamento e' una singola riassegnazione di attributo, operazione
        atomica in CPython: `render` legge il riferimento una sola volta per
        frame e vede sempre o la palette vecchia o quella nuova, mai uno stato
        intermedio. Non servono lock.

        Args:
            palette: Lista di colori `(r, g, b)` con componenti in [0, 1], usati
                come stop di un gradiente lungo l'asse spaziale; None o lista
                vuota per la modalita' default.
        """
        self._palette = palette

    def set_stereo(self, image: object | None) -> None:
        """Aggiorna l'immagine stereo usata per spostare la luce sull'asse.

        Args:
            image: Oggetto con attributi `pan_bass`, `pan_mid`, `pan_high`
                in [-1, 1] (es. StereoImage), oppure None per disattivare.
        """
        self._stereo = image

    def render(
        self, features: AudioFeatures, beat: BeatInfo | None = None
    ) -> list[tuple[int, tuple[float, float, float]]]:
        """Produce i colori di un frame a partire dalle feature audio.

        Per ogni canale mixa le tre bande secondo la posizione spaziale (o il
        ruolo forzato), aggiorna il punch (caricato dall'onset e decaduto di un
        frame) e converte hue/saturazione/valore in RGB. Hue e saturazione base
        derivano dal gradiente fisso (default) o dalla palette corrente.

        Args:
            features: Feature audio del blocco corrente, tutte in [0, 1].
            beat: Stato del beat tracker; questo effetto non lo usa.

        Returns:
            Lista di `(channel_id, (r, g, b))` per tutti i canali, con
            componenti float in [0, 1].
        """
        # Letta una sola volta per frame: set_palette puo' rimpiazzarla da un
        # altro thread, ma il riferimento locale resta stabile per tutto il loop.
        palette = self._palette
        # Knob runtime: riferimento locale per il frame (le scritture dal
        # frontend sono riassegnazioni atomiche di attributi float).
        knobs = tuning.live
        if knobs.punch_half_life_seconds != self._decay_half_life:
            self._decay_half_life = knobs.punch_half_life_seconds
            self._punch_decay_per_frame = 0.5 ** (1.0 / (self._decay_half_life * FRAME_RATE_HZ))

        # Morph "mirror": ricalcola le posizioni effettive (bande + colore) solo
        # quando il knob cambia, non a ogni frame. self.positions e' riassegnata
        # in blocco (lettura atomica dal loop di run per lo slam del drop).
        if knobs.mirror != self._mirror:
            self._mirror = knobs.mirror
            self._effective_positions = self._compute_effective_positions(self._mirror)
            self.positions = self._positions_by_id()

        # Livello bassi con contrasto riespanso (vedi BASS_CONTRAST_GAMMA).
        bass_level: float = features.bass**knobs.bass_contrast_gamma

        # Sub tenuto (HPSS): fondo di luce sui canali con peso bassi (sub glow).
        # getattr col default 0.0: campo assente = identico a oggi (max con 0).
        sub_level: float = getattr(features, "sub_level", 0.0)

        # Flash predittivo: sul blocco del beat PREVISTO dalla griglia (il
        # tracker anticipa, l'onset rilevato arriva 2-3 blocchi dopo) si
        # aggiunge un punch pesato da knob e confidence. Copre anche i beat
        # il cui attacco affoga nel mix. 0 sul knob = disattivato. Il punch e'
        # scalato dall'energia di sezione (continuo, nessuna soglia nuova): nei
        # breakdown/bridge quieti il flash sulla griglia svanisce anche col
        # tracker ancora agganciato, e rientra al ritorno della cassa.
        grid_punch: float = 0.0
        if beat is not None and beat.locked and beat.is_beat and knobs.beat_predict > 0.0:
            grid_punch = knobs.beat_predict * beat.confidence * features.section_energy

        # Immagine stereo del frame (riferimento locale, come la palette).
        stereo = self._stereo
        stereo_gain: float = knobs.stereo_gain if stereo is not None else 0.0

        # Onset sugli alti (snare/hi-hat): sorgente di punch per i canali a
        # dominanza alti, che l'onset broadband diluisce. getattr col default
        # 0.0 cosi' il codice regge anche se AudioFeatures non espone (ancora)
        # il campo: con 0.0 il max sotto non cambia nulla (identico a oggi).
        high_onset_strength: float = getattr(features, "high_onset_strength", 0.0)

        entries: list[tuple[int, tuple[float, float, float]]] = []
        for index, (channel, position) in enumerate(
            zip(self._channels, self._effective_positions, strict=True)
        ):
            # Pesi di banda: dal ruolo forzato se presente, altrimenti dalla
            # posizione (sinistra=bassi, centro=medi, destra=alti, transizioni
            # lineari). I ruoli a livello diretto (voice/drums/wash) non mixano
            # le bande: pesi a zero, il livello arriva da una feature dedicata.
            role = channel.role
            if role in DIRECT_LEVEL_ROLES:
                weight_bass, weight_mid, weight_high = 0.0, 0.0, 0.0
            elif role in ROLE_WEIGHTS:
                weight_bass, weight_mid, weight_high = ROLE_WEIGHTS[role]
            else:
                weight_bass = max(0.0, 1.0 - 2.0 * position)
                weight_high = max(0.0, 2.0 * position - 1.0)
                weight_mid = 1.0 - weight_bass - weight_high

            # Punch additivo: i colpi ravvicinati si sommano (il flash cresce)
            # invece di inseguire il massimo, che inghiottirebbe il secondo
            # colpo di un doppio beat. La sorgente del punch e' pesata per
            # canale: i canali a dominanza bassi pompano sul kick detector
            # (banda 20-150 Hz), gli altri sull'onset broadband. Il decay e'
            # applicato prima del calcolo del colore, cosi' il colore riflette
            # gia' lo stato del frame. Il ruolo "wash" (tappeto) non riceve
            # punch: il livello resta liscio.
            if role == "wash":
                punch_input: float = 0.0
            else:
                punch_input = (
                    weight_bass * features.bass_onset_strength
                    + (1.0 - weight_bass) * features.onset_strength
                )
                # I canali a dominanza alti prendono anche il colpo del rullante/
                # charleston: max, non somma, cosi' un high_onset_strength a 0 (o
                # assente) lascia il punch invariato. Con weight_bass 0 (voice,
                # drums) il punch e' l'onset broadband pieno.
                punch_input = max(punch_input, weight_high * high_onset_strength)
                if grid_punch > 0.0:
                    punch_input += grid_punch * (
                        weight_bass + GRID_PUNCH_NON_BASS_SHARE * (1.0 - weight_bass)
                    )
            if punch_input > 0.0:
                self._punches[index] = min(MAX_PUNCH, self._punches[index] + punch_input)
            self._punches[index] *= self._punch_decay_per_frame
            punch: float = self._punches[index]

            if role == "voice":
                level: float = features.voice_level
            elif role == "drums":
                # Componente percussiva HPSS: livello diretto (punch pieno dal
                # broadband gia' calcolato sopra, come i canali bass sul colpo).
                level = getattr(features, "percussive_level", 0.0)
            elif role == "wash":
                # Componente armonica HPSS con smoothing extra dolce (EMA per
                # canale): un tappeto che segue l'inviluppo senza sfarfallii.
                target = getattr(features, "harmonic_level", 0.0)
                smoothed = self._wash_levels[index] * self._wash_retain + target * (
                    1.0 - self._wash_retain
                )
                self._wash_levels[index] = smoothed
                level = smoothed
            else:
                # Sub glow: la componente bass non scende sotto un fondo dato dal
                # sub tenuto (max con SUB_FLOOR_SHARE * sub_level) sui canali con
                # peso bassi. Con sub_level 0 (assente) resta bass_level.
                effective_bass = bass_level
                if weight_bass > 0.0:
                    effective_bass = max(bass_level, SUB_FLOOR_SHARE * sub_level)
                level = (
                    effective_bass * weight_bass
                    + features.mid * weight_mid
                    + features.high * weight_high
                )
                if stereo_gain > 0.0:
                    # Pan stereo: il suono spostato a destra rinforza i canali
                    # di destra e attenua quelli di sinistra (e viceversa).
                    # L'asse del canale e' la posizione di rango in [-1, 1].
                    channel_axis = 2.0 * position - 1.0
                    channel_pan = (
                        weight_bass * stereo.pan_bass  # type: ignore[attr-defined]
                        + weight_mid * stereo.pan_mid  # type: ignore[attr-defined]
                        + weight_high * stereo.pan_high  # type: ignore[attr-defined]
                    )
                    level *= max(0.0, 1.0 + stereo_gain * channel_pan * channel_axis)

            # Hue, saturazione e scala di valore: gradiente di hue fisso
            # (default) oppure colore campionato dalla palette lungo l'asse.
            hue, sat_base, value_scale = palette_channel_base(palette, position)

            # Colore HSV: il punch desatura (verso sat_base ridotta) e accende.
            # Con sat_base = 1.0 e value_scale = 1.0 la formula coincide con la
            # versione precedente (modalita' default invariata).
            saturation: float = min(
                max(sat_base * (1.0 - PUNCH_DESATURATION * punch), MIN_SATURATION * sat_base),
                1.0,
            )
            # Headroom del punch (B7): la base scala con (1 - punch_gain*punch)
            # prima di sommare il punch, cosi' nei ritornelli pieni (level ~ 1)
            # ogni colpo da' uno scatto visibile invece di clampare a 1. Con
            # punch = 0 il fattore e' 1 e la formula coincide con quella di prima.
            punch_term: float = knobs.punch_gain * punch
            base_term: float = knobs.level_gain * level * (1.0 - min(1.0, punch_term))
            value: float = value_scale * (base_term + punch_term)
            # Macro-dinamica (B4): scurisce le sezioni quiete del brano in
            # proporzione a dynamics_gain; a 0 (default) il fattore e' 1.0.
            value *= 1.0 - knobs.dynamics_gain * (1.0 - features.section_energy)
            value = min(max(value, 0.0), 1.0)

            entries.append((channel.channel_id, colorsys.hsv_to_rgb(hue, saturation, value)))
        return entries
