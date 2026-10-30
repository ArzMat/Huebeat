"""Effetto "pulse": tutta l'area respira insieme sul beat grid.

L'intera area pulsa in modo uniforme sul battito quando il beat tracker e'
lockato (envelope agganciata alla fase del beat); se il tracker non e' lockato
(o assente) ripiega su un accumulatore di punch caricato dagli onset, con la
stessa matematica di decadimento dell'effetto spectrum. Il gradiente di colore
lungo l'asse resta, ma la LUMINOSITA' e' identica per tutti i canali.
"""

import colorsys
import math
from typing import TYPE_CHECKING

from huebeat import tuning
from huebeat.analysis import AudioFeatures
from huebeat.effects.base import (
    FRAME_RATE_HZ,
    ChannelInfo,
    palette_channel_base,
    spatial_layout,
)

if TYPE_CHECKING:
    from huebeat.beat import BeatInfo

# --- Envelope sul beat (tracker lockato) ---
# Costante di decadimento dell'envelope lungo la fase del beat: envelope =
# exp(-beat_phase / PULSE_DECAY_PHASE). Picco sul beat (phase 0), coda che si
# spegne prima del beat successivo.
PULSE_DECAY_PHASE = 0.3

# --- Fallback sugli onset (tracker non lockato) ---
# Tempo di dimezzamento del punch accumulato dopo un onset, in secondi. Identico
# alla matematica del punch di spectrum.
PUNCH_HALF_LIFE_SECONDS = 0.25
# Tetto del punch accumulato (onset ravvicinati si sommano fino a qui).
MAX_PUNCH = 1.0

# --- Luminosita' ---
# Quota di luminosita' sempre presente col livello (l'area non si spegne del
# tutto tra un pulse e l'altro finche' c'e' segnale).
PULSE_FLOOR = 0.35
# Contributo complessivo del pulse al value HSV.
PULSE_GAIN = 1.0


class PulseEffect:
    """Tutta l'area pulsa insieme sul beat grid, con fallback sugli onset.

    I canali sono ordinati per x crescente e ricevono una posizione
    normalizzata p in [0, 1] per rango, usata solo per il COLORE (il gradiente
    di hue fisso o la palette restano). La LUMINOSITA' e' invece uniforme: e'
    lo stesso value per tutti i canali, cosi' l'intera area "respira" insieme.

    Sorgente dell'envelope:

    - se `beat` e' presente e lockato, envelope = exp(-beta_phase /
      PULSE_DECAY_PHASE): massima sul beat, decade lungo l'intervallo;
    - altrimenti un accumulatore di punch caricato dagli onset (additivo con
      tetto 1.0 e dimezzamento PUNCH_HALF_LIFE_SECONDS), identico a spectrum.

    Il `role` del canale (`ChannelInfo.role`) e' ignorato: l'effetto non mixa le
    bande per ruolo, la luminosita' viene dal livello complessivo.

    Lo stato interno si limita al punch accumulato e alla palette corrente.
    """

    def __init__(self, channels: list[ChannelInfo]) -> None:
        """Prepara l'effetto: ordina i canali e precomputa posizioni e decay.

        Args:
            channels: Canali dell'area con le rispettive posizioni.
        """
        self._channels, self._positions = spatial_layout(channels)
        # Posizione [0, 1] usata per COLORARE ogni canale (id -> posizione),
        # esposta per lo slam del drop.
        self.positions: dict[int, float] = {
            channel.channel_id: position
            for channel, position in zip(self._channels, self._positions, strict=True)
        }
        # Punch accumulato dagli onset (fallback quando il beat non e' lockato).
        self._punch: float = 0.0
        # Half-life corrente e fattore di decadimento per frame derivato;
        # ricalcolato in render solo quando il knob "Coda punch" cambia (cache),
        # come spectrum. Al default (PUNCH_HALF_LIFE_SECONDS) e' identico a prima.
        self._decay_half_life: float = tuning.live.punch_half_life_seconds
        self._punch_decay_per_frame: float = 0.5 ** (1.0 / (self._decay_half_life * FRAME_RATE_HZ))
        # Palette corrente per la colorazione dei canali; None = modalita'
        # default (gradiente di hue fisso). Aggiornata da set_palette.
        self._palette: list[tuple[float, float, float]] | None = None

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

    def render(
        self, features: AudioFeatures, beat: BeatInfo | None = None
    ) -> list[tuple[int, tuple[float, float, float]]]:
        """Produce i colori di un frame a partire dalle feature audio e dal beat.

        Calcola l'envelope (dalla fase del beat se lockato, altrimenti dal punch
        accumulato sugli onset), la combina col livello complessivo in una
        luminosita' uniforme e colora ogni canale con la sua posizione spaziale.

        Args:
            features: Feature audio del blocco corrente, tutte in [0, 1].
            beat: Stato del beat tracker; se lockato guida l'envelope, altrimenti
                si ripiega sugli onset. Puo' essere None.

        Returns:
            Lista di `(channel_id, (r, g, b))` per tutti i canali, con
            componenti float in [0, 1].
        """
        # Letta una sola volta per frame: set_palette puo' rimpiazzarla da un
        # altro thread, ma il riferimento locale resta stabile per tutto il loop.
        palette = self._palette

        # Knob runtime: ricalcola il decay solo se la coda punch e' cambiata
        # (come spectrum), cosi' il knob "Coda punch" agisce a caldo anche qui.
        knobs = tuning.live
        if knobs.punch_half_life_seconds != self._decay_half_life:
            self._decay_half_life = knobs.punch_half_life_seconds
            self._punch_decay_per_frame = 0.5 ** (1.0 / (self._decay_half_life * FRAME_RATE_HZ))

        # Punch sempre aggiornato: cosi' quando il beat perde il lock il
        # fallback riparte da uno stato coerente col segnale recente.
        if features.onset_strength > 0.0:
            self._punch = min(MAX_PUNCH, self._punch + features.onset_strength)
        self._punch *= self._punch_decay_per_frame

        # Envelope: dalla fase del beat se lockato, altrimenti dal punch.
        if beat is not None and beat.locked:
            envelope = math.exp(-beat.beat_phase / PULSE_DECAY_PHASE)
        else:
            envelope = self._punch

        # Livello complessivo (media delle tre bande) modulato dall'envelope,
        # con un pavimento sempre presente finche' c'e' segnale.
        overall = (features.bass + features.mid + features.high) / 3.0
        value_raw = overall * (PULSE_FLOOR + (1.0 - PULSE_FLOOR) * envelope)

        entries: list[tuple[int, tuple[float, float, float]]] = []
        for channel, position in zip(self._channels, self._positions, strict=True):
            # Colore dalla posizione (gradiente o palette); luminosita' uniforme.
            hue, sat_base, value_scale = palette_channel_base(palette, position)
            value = min(max(value_scale * value_raw * PULSE_GAIN, 0.0), 1.0)
            entries.append((channel.channel_id, colorsys.hsv_to_rgb(hue, sat_base, value)))
        return entries
