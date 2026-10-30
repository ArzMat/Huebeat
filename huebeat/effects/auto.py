"""Effetto "auto": sceglie tra spectrum e pulse in base all'energia di sezione.

Wrapper che alterna due effetti a seconda di quanto e' "carica" la sezione del
brano: spectrum (spettro spaziale reattivo) nelle sezioni cariche, pulse
(respiro uniforme) in quelle quiete. Il passaggio ha isteresi (due soglie) e una
permanenza minima, cosi' non oscilla di continuo al confine.
"""

from typing import TYPE_CHECKING

from huebeat import tuning
from huebeat.analysis import AudioFeatures
from huebeat.effects.base import FRAME_RATE_HZ, ChannelInfo
from huebeat.effects.pulse import PulseEffect
from huebeat.effects.spectrum import SpectrumPunchEffect

if TYPE_CHECKING:
    from huebeat.beat import BeatInfo

# --- Soglie e permanenza dai knob adaptive (tuning.live) ---
# auto condivide con adaptive la stessa isteresi sull'energia di sezione e la
# stessa permanenza minima, lette a caldo dai knob invece che da costanti fisse:
# sopra calm_energy_off la sezione e' "carica" -> spectrum, a/sotto
# calm_energy_on e' "quieta" -> pulse, in mezzo nessun cambio (banda morta); il
# cambio effettivo scatta solo dopo min_dwell_seconds nel regime corrente.
# (Storicamente erano 0.55 / 0.45 / 8.0 s hardcoded.)

# Nomi delle due modalita' esposte da `mode`.
MODE_SPECTRUM = "spectrum"
MODE_PULSE = "pulse"


class AutoEffect:
    """Sceglie tra spectrum (sezioni cariche) e pulse (sezioni quiete).

    Istanzia internamente `SpectrumPunchEffect` e `PulseEffect` sugli stessi
    canali e delega il rendering a quello attivo. La scelta si basa su
    `features.section_energy` con isteresi, letta dai knob adaptive live
    (`tuning.live`): sopra `calm_energy_off` punta a spectrum, a/sotto
    `calm_energy_on` punta a pulse, in mezzo resta invariata. Il cambio
    effettivo avviene solo se la modalita' corrente dura gia' da almeno
    `min_dwell_seconds` (permanenza minima), cosi' i confini di sezione non
    fanno oscillare l'effetto.

    La palette impostata con `set_palette` viene inoltrata a entrambi gli effetti
    interni, cosi' il colore resta coerente a qualsiasi commutazione.
    """

    def __init__(self, channels: list[ChannelInfo]) -> None:
        """Prepara i due effetti interni e lo stato di commutazione.

        Args:
            channels: Canali dell'area con posizioni (e ruoli opzionali).
        """
        self._spectrum = SpectrumPunchEffect(channels)
        self._pulse = PulseEffect(channels)
        # Modalita' corrente ("spectrum" | "pulse"); si parte da spectrum.
        self._mode: str = MODE_SPECTRUM
        # Frame trascorsi nella modalita' corrente (per la permanenza minima).
        self._frames_in_mode: int = 0

    @property
    def mode(self) -> str:
        """Modalita' attualmente attiva.

        Returns:
            "spectrum" oppure "pulse".
        """
        return self._mode

    @property
    def positions(self) -> dict[int, float]:
        """Mappa canale -> posizione [0, 1] dell'effetto interno ATTIVO.

        spectrum e pulse condividono lo stesso layout spaziale, quindi la mappa
        e' la stessa nei due regimi; segue comunque l'effetto attivo per
        coerenza con lo slam del drop.

        Returns:
            Dict channel_id -> posizione [0, 1] dell'effetto attivo.
        """
        return self._spectrum.positions if self._mode == MODE_SPECTRUM else self._pulse.positions

    def set_palette(self, palette: list[tuple[float, float, float]] | None) -> None:
        """Inoltra la palette a entrambi gli effetti interni.

        Thread-safety: come per gli effetti sottostanti, l'aggiornamento e' una
        riassegnazione atomica in ciascuno; qui e' solo un doppio inoltro.

        Args:
            palette: Lista di colori `(r, g, b)` con componenti in [0, 1]; None o
                lista vuota per la modalita' default.
        """
        self._spectrum.set_palette(palette)
        self._pulse.set_palette(palette)

    def set_stereo(self, image: object | None) -> None:
        """Inoltra l'immagine stereo all'effetto spectrum interno.

        Args:
            image: Oggetto con pan per banda (vedi SpectrumPunchEffect.set_stereo),
                oppure None per disattivare.
        """
        self._spectrum.set_stereo(image)

    def render(
        self, features: AudioFeatures, beat: BeatInfo | None = None
    ) -> list[tuple[int, tuple[float, float, float]]]:
        """Aggiorna la modalita' e delega il rendering all'effetto attivo.

        Determina la modalita' obiettivo dall'energia di sezione (con isteresi) e
        commuta solo se la modalita' corrente ha superato la permanenza minima;
        poi rende con l'effetto selezionato passando features e beat.

        Args:
            features: Feature audio del blocco corrente, tutte in [0, 1].
            beat: Stato del beat tracker, inoltrato all'effetto attivo.

        Returns:
            Lista di `(channel_id, (r, g, b))` prodotta dall'effetto attivo.
        """
        # Modalita' obiettivo dall'energia di sezione (isteresi a due soglie
        # lette dai knob adaptive live): nella banda morta target resta la
        # modalita' corrente.
        knobs = tuning.live
        energy = features.section_energy
        if energy >= knobs.calm_energy_off:
            target = MODE_SPECTRUM
        elif energy <= knobs.calm_energy_on:
            target = MODE_PULSE
        else:
            target = self._mode

        # Commuta solo dopo la permanenza minima nella modalita' corrente,
        # anch'essa dal knob (ricalcolata a ogni frame, costa un moltiplicatore).
        min_dwell_frames = int(knobs.min_dwell_seconds * FRAME_RATE_HZ)
        if target != self._mode and self._frames_in_mode >= min_dwell_frames:
            self._mode = target
            self._frames_in_mode = 0
        else:
            self._frames_in_mode += 1

        active = self._spectrum if self._mode == MODE_SPECTRUM else self._pulse
        return active.render(features, beat)
