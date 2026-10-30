"""Effetto "waterfall": la striscia scorre, ogni colpo viaggia nel tempo.

Ogni frame scrive l'intensita' del colpo all'ingresso (il canale piu' a
sinistra) e la storia scorre verso destra: cosi' il tempo si traduce in spazio
e ogni beat resta visibile come una banda distinta lungo la striscia, anche
nei drop fittissimi dove i colpi si accavallano. La luminosita' di ogni canale
e' il massimo della finestra di storia che gli compete, quindi anche un
impulso di un solo frame resta visibile per tutta la finestra mentre attraversa
il canale.
"""

import colorsys
from collections import deque
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

# --- Velocita' di scorrimento ---
# Tempo perche' un colpo attraversi l'intera striscia da sinistra a destra, in
# secondi. Con n canali definisce quanti frame di storia copre ogni canale
# (vedi WATERFALL_TRAVERSE_SECONDS nello __init__): piu' basso = scorrimento
# piu' veloce, bande piu' strette. Usato quando il beat tracker NON e' lockato.
WATERFALL_TRAVERSE_SECONDS = 0.7
# Col beat tracker lockato lo scorrimento si aggancia al tempo: l'attraversata
# dura questo numero di beat (2 beat: a 120 BPM = 1.0 s), clampata tra i due
# limiti qui sotto per non degenerare su tempi estremi.
WATERFALL_BEATS_PER_TRAVERSE = 2.0
WATERFALL_TRAVERSE_MIN_SECONDS = 0.4
WATERFALL_TRAVERSE_MAX_SECONDS = 1.4

# --- Fondo (la striscia non e' mai nera durante la musica) ---
# Frazione dell'energia complessiva (media delle tre bande) sempre presente
# come luminosita' di fondo, cosi' tra un colpo e l'altro la striscia resta
# leggermente accesa invece di spegnersi del tutto.
WATERFALL_BACKGROUND_GAIN = 0.2

# --- Guadagno del colpo ---
# Peso del contributo della banda di storia (il colpo che scorre) sul value
# HSV, sopra il fondo. 1.0: un colpo pieno porta il canale a piena scala.
WATERFALL_HIT_GAIN = 1.0


class WaterfallEffect:
    """La striscia scorre: ogni colpo si scrive a sinistra e viaggia a destra.

    I canali sono ordinati lungo il percorso fisico dell'impianto
    (path_layout: il ferro di cavallo viene percorso come una U, non
    appiattito sull'asse x): l'ingresso e' il capo sinistro del percorso e
    lo scorrimento lo segue per intero. A ogni frame
    l'intensita' del colpo (il massimo tra kick detector e onset broadband)
    viene spinta in testa a una deque di storia; la luminosita' del canale
    i-esimo da sinistra e' il MASSIMO della storia nella finestra
    [i * step, (i + 1) * step), dove step e' il numero di frame di storia per
    canale. Il massimo (invece della media) fa sopravvivere anche un impulso di
    un solo frame per tutta la finestra mentre attraversa il canale, cosi' ogni
    beat resta una banda distinta.

    Sopra al colpo c'e' un fondo pari a WATERFALL_BACKGROUND_GAIN volte
    l'energia complessiva, perche' la striscia non sia mai nera durante la
    musica. Il colore di ogni canale viene dalla sua posizione spaziale
    (gradiente di hue fisso o palette); non c'e' desaturazione.

    Il `role` del canale e' IGNORATO (l'effetto non mixa le bande per ruolo).
    `beat` invece aggancia lo scorrimento al tempo: col tracker lockato
    l'attraversata dura WATERFALL_BEATS_PER_TRAVERSE beat (clampata tra i
    limiti MIN/MAX), senza lock si usa il tempo fisso di default.

    Lo stato interno si limita alla deque di storia e alla palette corrente.
    """

    def __init__(self, channels: list[ChannelInfo]) -> None:
        """Prepara l'effetto: ordina i canali e dimensiona la storia.

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
        count = len(self._channels)
        # Frame di storia coperti da ogni canale: l'intera striscia si
        # attraversa in WATERFALL_TRAVERSE_SECONDS, spalmati sui count canali.
        # Almeno 1 frame per canale anche con molti canali o traverse corto.
        self._step: int = max(1, round(WATERFALL_TRAVERSE_SECONDS * FRAME_RATE_HZ / count))
        # Storia dei valori di hit, piu' recente in testa (appendleft),
        # dimensionata sul traverse MASSIMO: col beat sync lo step cambia a
        # runtime e la finestra dell'ultimo canale arriva a step * count.
        max_step: int = max(1, round(WATERFALL_TRAVERSE_MAX_SECONDS * FRAME_RATE_HZ / count))
        self._history: deque[float] = deque(maxlen=max_step * count)
        # Palette corrente per la colorazione dei canali; None = modalita'
        # default (gradiente di hue fisso). Aggiornata da set_palette.
        self._palette: list[tuple[float, float, float]] | None = None

    def set_palette(self, palette: list[tuple[float, float, float]] | None) -> None:
        """Imposta la palette usata per colorare i canali (o torna al default.)

        In modalita' palette hue e saturazione base di ogni canale derivano dai
        colori RGB della palette (interpolati lungo l'asse spaziale) invece che
        dal gradiente di hue fisso; passando None (o una lista vuota) l'effetto
        torna alla colorazione di default.

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
        """Produce i colori di un frame facendo scorrere la storia dei colpi.

        Spinge il colpo del frame corrente in testa alla storia, poi per ogni
        canale prende il massimo della sua finestra di storia come intensita'
        del colpo che lo attraversa, ci somma il fondo e colora il canale con la
        sua posizione spaziale.

        Args:
            features: Feature audio del blocco corrente, tutte in [0, 1].
            beat: Stato del beat tracker; questo effetto lo ignora.

        Returns:
            Lista di `(channel_id, (r, g, b))` per tutti i canali, con
            componenti float in [0, 1].
        """
        # Letta una sola volta per frame: set_palette puo' rimpiazzarla da un
        # altro thread, ma il riferimento locale resta stabile per tutto il loop.
        palette = self._palette

        # Scorrimento agganciato al tempo: col beat tracker lockato
        # l'attraversata dura WATERFALL_BEATS_PER_TRAVERSE beat (clampata),
        # altrimenti il default a tempo fisso.
        if beat is not None and getattr(beat, "locked", False) and beat.bpm > 0.0:
            traverse = min(
                max(
                    WATERFALL_BEATS_PER_TRAVERSE * 60.0 / beat.bpm,
                    WATERFALL_TRAVERSE_MIN_SECONDS,
                ),
                WATERFALL_TRAVERSE_MAX_SECONDS,
            )
        else:
            traverse = WATERFALL_TRAVERSE_SECONDS
        self._step = max(1, round(traverse * FRAME_RATE_HZ / len(self._channels)))

        # Intensita' del colpo del frame: il piu' forte tra kick detector e
        # onset broadband, spinto in testa alla storia (piu' recente davanti).
        hit: float = max(features.bass_onset_strength, features.onset_strength)
        self._history.appendleft(hit)

        # Fondo comune: frazione dell'energia complessiva (media delle bande),
        # cosi' la striscia non e' mai nera finche' c'e' musica.
        overall: float = (features.bass + features.mid + features.high) / 3.0
        background: float = WATERFALL_BACKGROUND_GAIN * overall

        step = self._step
        # Materializzata una volta: lo slicing per finestra e' su una lista, la
        # deque non supporta lo slicing diretto.
        history = list(self._history)
        history_len = len(history)

        entries: list[tuple[int, tuple[float, float, float]]] = []
        for index, (channel, position) in enumerate(
            zip(self._channels, self._positions, strict=True)
        ):
            # Finestra di storia del canale i-esimo da sinistra: [start, end).
            # Il canale 0 (ingresso) vede la storia piu' recente, l'ultimo la
            # piu' vecchia -> il colpo appare a sinistra e scorre a destra.
            start = index * step
            end = start + step
            # Massimo della finestra: un impulso di un frame resta visibile per
            # tutta la finestra mentre la attraversa. Storia piu' corta della
            # finestra (start oltre i dati disponibili) -> banda ancora spenta.
            if start >= history_len:
                band = 0.0
            else:
                band = max(history[start:end])

            # Colore dalla posizione (gradiente o palette); nessuna desaturazione.
            hue, sat_base, value_scale = palette_channel_base(palette, position)
            value = min(max(value_scale * (background + WATERFALL_HIT_GAIN * band), 0.0), 1.0)
            entries.append((channel.channel_id, colorsys.hsv_to_rgb(hue, sat_base, value)))
        return entries
