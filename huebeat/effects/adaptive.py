"""Effetto "adaptive": segue il carattere del brano a tre regimi.

Wrapper che sceglie momento per momento tra pulse (calma), spectrum (normale) e
waterfall (raffica) leggendo due segnali: l'energia di sezione e la densita' di
colpi su una finestra scorrevole. La calma e la raffica hanno isteresi (due
soglie ciascuna) e ogni regime deve durare una permanenza minima prima di
poter cambiare, cosi' l'effetto non oscilla di continuo ai confini.
"""

from collections import deque
from typing import TYPE_CHECKING

from huebeat import tuning
from huebeat.analysis import AudioFeatures
from huebeat.effects.base import FRAME_RATE_HZ, ChannelInfo
from huebeat.effects.pulse import PulseEffect
from huebeat.effects.spectrum import SpectrumPunchEffect
from huebeat.effects.waterfall import WaterfallEffect

if TYPE_CHECKING:
    from huebeat.beat import BeatInfo

# --- Finestra per la densita' di colpi ---
# Durata (in secondi) della finestra scorrevole su cui si stima la densita' di
# colpi in eventi/s. 1.5 s: abbastanza lunga da stabilizzare la stima, corta
# abbastanza da reagire all'arrivo di un drop.
DENSITY_WINDOW_SECONDS = 1.5
# La finestra in numero di frame (deque booleana), derivata dal frame rate.
DENSITY_WINDOW_FRAMES = round(DENSITY_WINDOW_SECONDS * FRAME_RATE_HZ)

# --- Isteresi sulla calma (regime "pulse") ---
# NOTA tuning runtime: CALM_*, BURST_* e MIN_DWELL_SECONDS sono i DEFAULT
# storici; i valori vivi arrivano da huebeat.tuning.live (soglie regolabili a
# caldo dal frontend, isteresi derivata con divario fisso).
# Sotto o pari a questa energia di sezione la sezione e' "calma" -> pulse.
CALM_ENERGY_ON = 0.35
# Si esce dalla calma solo quando l'energia risale a questa soglia: banda morta
# tra le due che evita rimbalzi al confine.
CALM_ENERGY_OFF = 0.45

# --- Isteresi sulla raffica (regime "waterfall") ---
# A densita' di colpi (eventi/s) pari o superiore a questa la sezione e' una
# "raffica" -> waterfall.
BURST_DENSITY_ON = 7.0
# Si esce dalla raffica solo quando la densita' scende a questa soglia: banda
# morta che evita rimbalzi al confine.
BURST_DENSITY_OFF = 5.0

# --- Permanenza minima nel regime corrente ---
# Secondi minimi in un regime prima di poter cambiare: evita commutazioni
# rapide quando i segnali oscillano intorno alle soglie.
MIN_DWELL_SECONDS = 2.0

# Nomi dei tre regimi esposti da `mode`.
MODE_PULSE = "pulse"
MODE_SPECTRUM = "spectrum"
MODE_WATERFALL = "waterfall"


class AdaptiveEffect:
    """Sceglie tra pulse (calma), spectrum (normale) e waterfall (raffica).

    Istanzia internamente `PulseEffect`, `SpectrumPunchEffect` e
    `WaterfallEffect` sugli stessi canali e delega il rendering a quello attivo;
    `set_palette` inoltra a tutti e tre. Il regime parte da "spectrum".

    A ogni frame calcola due segnali:

    - energia: `features.section_energy`;
    - densita' di colpi: frazione di frame recenti con un onset (kick o
      broadband) su una finestra scorrevole di DENSITY_WINDOW_FRAMES frame,
      convertita in eventi/s.

    Il regime OBIETTIVO segue questa priorita':

    1. "pulse" se la sezione e' calma: section_energy <= CALM_ENERGY_ON; si
       esce dalla calma solo con section_energy >= CALM_ENERGY_OFF (isteresi);
    2. "waterfall" se e' una raffica: densita' >= BURST_DENSITY_ON; si esce
       dalla raffica solo con densita' <= BURST_DENSITY_OFF (isteresi);
    3. altrimenti "spectrum".

    Il cambio effettivo avviene solo se il regime corrente dura gia' da almeno
    MIN_DWELL_SECONDS (permanenza minima), cosi' i segnali che ballano intorno
    alle soglie non fanno oscillare l'effetto.
    """

    def __init__(self, channels: list[ChannelInfo]) -> None:
        """Prepara i tre effetti interni e lo stato di commutazione.

        Args:
            channels: Canali dell'area con posizioni (e ruoli opzionali).
        """
        self._pulse = PulseEffect(channels)
        self._spectrum = SpectrumPunchEffect(channels)
        self._waterfall = WaterfallEffect(channels)
        # Regime corrente ("pulse" | "spectrum" | "waterfall"); parte da spectrum.
        self._mode: str = MODE_SPECTRUM
        # Frame trascorsi nel regime corrente (per la permanenza minima).
        self._frames_in_mode: int = 0
        # Storia booleana degli ultimi frame: True dove c'era un onset (kick o
        # broadband). Usata per stimare la densita' di colpi in eventi/s.
        self._onset_history: deque[bool] = deque(maxlen=DENSITY_WINDOW_FRAMES)

    @property
    def mode(self) -> str:
        """Regime attualmente attivo.

        Returns:
            "pulse", "spectrum" oppure "waterfall".
        """
        return self._mode

    @property
    def positions(self) -> dict[int, float]:
        """Mappa canale -> posizione [0, 1] dell'effetto interno ATTIVO.

        I regimi usano layout diversi (pulse/spectrum spaziale, waterfall lungo
        il percorso): la mappa segue quindi il regime corrente, cosi' lo slam
        del drop colora con le stesse posizioni dell'effetto che sta rendendo.

        Returns:
            Dict channel_id -> posizione [0, 1] dell'effetto attivo.
        """
        if self._mode == MODE_PULSE:
            return self._pulse.positions
        if self._mode == MODE_WATERFALL:
            return self._waterfall.positions
        return self._spectrum.positions

    def set_palette(self, palette: list[tuple[float, float, float]] | None) -> None:
        """Inoltra la palette a tutti e tre gli effetti interni.

        Thread-safety: come per gli effetti sottostanti, l'aggiornamento in
        ciascuno e' una riassegnazione atomica; qui e' solo un triplo inoltro.

        Args:
            palette: Lista di colori `(r, g, b)` con componenti in [0, 1]; None o
                lista vuota per la modalita' default.
        """
        self._pulse.set_palette(palette)
        self._spectrum.set_palette(palette)
        self._waterfall.set_palette(palette)

    def set_stereo(self, image: object | None) -> None:
        """Inoltra l'immagine stereo all'effetto spectrum interno.

        Args:
            image: Oggetto con pan per banda (vedi SpectrumPunchEffect.set_stereo),
                oppure None per disattivare.
        """
        self._spectrum.set_stereo(image)

    def _density(self) -> float:
        """Stima la densita' di colpi corrente in eventi al secondo.

        Returns:
            Colpi al secondo sulla finestra scorrevole: conteggio dei frame con
            onset diviso per la durata (in secondi) dei frame effettivamente in
            storia. 0.0 se la storia e' ancora vuota.
        """
        window = len(self._onset_history)
        if window == 0:
            return 0.0
        hits = sum(self._onset_history)
        return hits / (window / FRAME_RATE_HZ)

    def render(
        self, features: AudioFeatures, beat: "BeatInfo | None" = None
    ) -> list[tuple[int, tuple[float, float, float]]]:
        """Aggiorna il regime e delega il rendering all'effetto attivo.

        Registra l'onset del frame nella storia, ricalcola la densita' di colpi,
        determina il regime obiettivo (calma / raffica / normale con isteresi) e
        commuta solo dopo la permanenza minima; poi rende con l'effetto scelto
        passando features e beat.

        Args:
            features: Feature audio del blocco corrente, tutte in [0, 1].
            beat: Stato del beat tracker, inoltrato all'effetto attivo.

        Returns:
            Lista di `(channel_id, (r, g, b))` prodotta dall'effetto attivo.
        """
        # Registra se in questo frame c'e' stato un colpo (kick o broadband) e
        # aggiorna la stima di densita'.
        onset = features.bass_onset_strength > 0.0 or features.onset_strength > 0.0
        self._onset_history.append(onset)
        density = self._density()

        # Regime obiettivo per priorita' (calma > raffica > normale), con
        # isteresi a due soglie sia sulla calma che sulla raffica: nella banda
        # morta di ciascuna resta il regime corrente se gia' in quel regime.
        # Soglie lette dai knob runtime (riferimento locale per il frame).
        knobs = tuning.live
        energy = features.section_energy
        if energy <= knobs.calm_energy_on:
            target = MODE_PULSE
        elif self._mode == MODE_PULSE and energy < knobs.calm_energy_off:
            # Ancora nella banda morta della calma: non esce da pulse.
            target = MODE_PULSE
        elif density >= knobs.burst_density_on:
            target = MODE_WATERFALL
        elif self._mode == MODE_WATERFALL and density > knobs.burst_density_off:
            # Ancora nella banda morta della raffica: non esce da waterfall.
            target = MODE_WATERFALL
        else:
            target = MODE_SPECTRUM

        # Commuta solo dopo la permanenza minima nel regime corrente.
        min_dwell_frames = int(knobs.min_dwell_seconds * FRAME_RATE_HZ)
        if target != self._mode and self._frames_in_mode >= min_dwell_frames:
            self._mode = target
            self._frames_in_mode = 0
        else:
            self._frames_in_mode += 1

        if self._mode == MODE_PULSE:
            active = self._pulse
        elif self._mode == MODE_WATERFALL:
            active = self._waterfall
        else:
            active = self._spectrum
        return active.render(features, beat)
