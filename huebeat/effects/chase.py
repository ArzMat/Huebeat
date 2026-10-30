"""Effetto "chase": flash che corre lungo la striscia a ogni onset.

A ogni onset spawna un "pacchetto" di luce che viaggia lungo l'asse spaziale,
in direzione alternata (uno verso destra, il successivo verso sinistra), sopra
un fondo che "respira" col livello complessivo. Piu' pacchetti attivi si
sommano: il risultato e' una scia di flash che attraversa i canali.
"""

import colorsys
import math
from typing import TYPE_CHECKING

from huebeat.analysis import AudioFeatures
from huebeat.effects.base import (
    FRAME_RATE_HZ,
    ChannelInfo,
    palette_channel_base,
    path_layout,
)

if TYPE_CHECKING:
    from huebeat.beat import BeatInfo

# --- Pacchetti che corrono lungo la striscia ---
# Tempo (secondi) che un pacchetto impiega a percorrere l'intera striscia [0, 1].
CHASE_TRAVEL_SECONDS = 0.6
# Larghezza spaziale del flash (sigma della gaussiana) in unita' di posizione.
CHASE_WIDTH = 0.15

# --- Guadagni di luminosita' ---
# Contributo del fondo "che respira" al value HSV, scala del livello complessivo.
BACKGROUND_GAIN = 0.3
# Contributo della somma dei pacchetti al value HSV.
CHASE_GAIN = 1.0


class ChaseEffect:
    """Flash che corre lungo la striscia a ogni onset, su fondo che respira.

    I canali sono ordinati lungo il percorso fisico dell'impianto
    (path_layout: una striscia a U viene percorsa come una U) e ricevono una
    posizione normalizzata p in [0, 1] sulla distanza cumulativa. A ogni
    onset nasce un pacchetto di luce
    con forza pari a `onset_strength`, in direzione alternata rispetto al
    precedente; ogni frame i pacchetti avanzano e muoiono usciti dall'area. Il
    contributo di un pacchetto a un canale e' una gaussiana centrata sulla
    posizione del pacchetto; i pacchetti si sommano (clamp a 1). Sotto scorre un
    fondo proporzionale al livello complessivo (media di bass/mid/high).

    Il colore di ogni canale viene dalla posizione spaziale (gradiente di hue
    fisso o palette): il flash accende la luminosita' ma NON desatura.

    Il `role` del canale (`ChannelInfo.role`) e' ignorato: l'effetto e' puramente
    spaziale e non mixa le bande per ruolo. Anche `beat` e' ignorato.

    Lo stato interno si limita ai pacchetti attivi, alla direzione del prossimo
    spawn e alla palette corrente.
    """

    def __init__(self, channels: list[ChannelInfo]) -> None:
        """Prepara l'effetto: ordina i canali e precomputa posizioni e passo.

        Args:
            channels: Canali dell'area con le rispettive posizioni.
        """
        self._channels, self._positions = path_layout(channels)
        # Posizione [0, 1] lungo il percorso usata per COLORARE ogni canale
        # (id -> posizione), esposta per lo slam del drop: qui e' l'ordine del
        # cammino fisico, non l'x, quindi lo slam deve leggere questa mappa.
        self.positions: dict[int, float] = {
            channel.channel_id: position
            for channel, position in zip(self._channels, self._positions, strict=True)
        }
        # Pacchetti attivi come triple (posizione, forza, direzione). La
        # direzione (+1 verso destra, -1 verso sinistra) e' fissata allo spawn e
        # non cambia: la posizione vive in [0 - margine, 1 + margine], oltre il
        # pacchetto e' morto.
        self._packets: list[tuple[float, float, float]] = []
        # Passo di avanzamento per frame (frazione della striscia).
        self._step_per_frame: float = 1.0 / (CHASE_TRAVEL_SECONDS * FRAME_RATE_HZ)
        # Margine oltre i bordi entro cui un pacchetto resta vivo (entra/esce
        # con la coda gaussiana ancora visibile).
        self._margin: float = CHASE_WIDTH * 2.0
        # True se il prossimo pacchetto parte da sinistra (posizione iniziale
        # negativa, avanza verso destra); si alterna a ogni spawn.
        self._next_from_left: bool = True
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
        self, features: AudioFeatures, beat: "BeatInfo | None" = None
    ) -> list[tuple[int, tuple[float, float, float]]]:
        """Produce i colori di un frame a partire dalle feature audio.

        Spawna un pacchetto se c'e' un onset, avanza tutti i pacchetti attivi,
        elimina quelli usciti dall'area, poi per ogni canale somma i contributi
        gaussiani dei pacchetti e li aggiunge al fondo che respira col livello
        complessivo. Il colore viene dalla posizione (gradiente o palette).

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

        # Spawn di un nuovo pacchetto sull'onset, in direzione alternata: chi
        # parte da sinistra (-margine) va verso destra (+1), il successivo parte
        # da destra (1 + margine) e va verso sinistra (-1).
        if features.onset_strength > 0.0:
            if self._next_from_left:
                start, direction = -self._margin, 1.0
            else:
                start, direction = 1.0 + self._margin, -1.0
            self._packets.append((start, features.onset_strength, direction))
            self._next_from_left = not self._next_from_left

        # Avanza i pacchetti nella loro direzione fissa e tieni solo quelli
        # ancora dentro [0 - margine, 1 + margine].
        advanced: list[tuple[float, float, float]] = []
        for position, strength, direction in self._packets:
            new_position = position + direction * self._step_per_frame
            if -self._margin <= new_position <= 1.0 + self._margin:
                advanced.append((new_position, strength, direction))
        self._packets = advanced

        # Fondo che respira col livello complessivo (media delle tre bande).
        overall = (features.bass + features.mid + features.high) / 3.0
        background = BACKGROUND_GAIN * overall

        entries: list[tuple[int, tuple[float, float, float]]] = []
        for channel, position in zip(self._channels, self._positions, strict=True):
            # Somma dei contributi gaussiani dei pacchetti a questo canale.
            packet_sum = 0.0
            for packet_position, strength, _direction in self._packets:
                delta = (position - packet_position) / CHASE_WIDTH
                packet_sum += strength * math.exp(-(delta * delta))
            packet_sum = min(packet_sum, 1.0)

            # Colore dalla posizione: il flash accende ma non desatura.
            hue, sat_base, value_scale = palette_channel_base(palette, position)
            value = min(max(value_scale * (background + CHASE_GAIN * packet_sum), 0.0), 1.0)

            entries.append((channel.channel_id, colorsys.hsv_to_rgb(hue, sat_base, value)))
        return entries
