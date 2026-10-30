"""Fondamenta condivise del package effetti.

Contiene cio' che ogni effetto usa: la posizione dei canali, il layout
spaziale normalizzato e il campionamento della palette lungo l'asse. Gli
effetti sono classi con la stessa interfaccia (duck typing):

- `set_palette(palette | None)`: aggiorna la palette corrente (thread-safe
  per assegnazione atomica).
- `render(features, beat=None) -> list[(channel_id, (r, g, b))]`: produce i
  colori di un frame; `beat` e' il `BeatInfo` del beat tracker, che gli
  effetti possono ignorare.
"""

import colorsys
import math
from dataclasses import dataclass

# --- Cadenza del loop di rendering ---
# Frame rate assunto per i decadimenti temporali: blocchi da 1024 campioni a
# 48 kHz (la cadenza reale del loop `run`, dettata dalla lettura audio).
FRAME_RATE_HZ = 47.0

# --- Gradiente di hue di default lungo l'asse x (modalita' senza palette) ---
# Hue del canale piu' a sinistra (0.85 = violetto).
BASE_HUE = 0.85
# Escursione di hue da sinistra a destra (violetto -> rosso -> arancio).
HUE_SPAN = 0.35

# --- Posizione normalizzata di fallback per aree a canale singolo ---
SINGLE_CHANNEL_POSITION = 0.5


@dataclass(frozen=True)
class ChannelInfo:
    """Posizione (e ruolo opzionale) di un canale dell'area Entertainment.

    Attributes:
        channel_id: Id del canale (0-255) come esposto dal Bridge.
        x: Coordinata x nell'area (-1 = sinistra, +1 = destra).
        y: Coordinata y nell'area.
        z: Coordinata z nell'area.
        role: Ruolo forzato del canale ("bass" | "mid" | "high") oppure None
            per la distribuzione spaziale automatica. Onorato dagli effetti
            che mixano le bande per canale (spectrum); gli altri lo ignorano.
    """

    channel_id: int
    x: float
    y: float
    z: float
    role: str | None = None


def spatial_layout(channels: list[ChannelInfo]) -> tuple[list[ChannelInfo], list[float]]:
    """Ordina i canali per x e assegna una posizione normalizzata per rango.

    Args:
        channels: Canali dell'area con le rispettive posizioni.

    Returns:
        Coppia `(canali_ordinati, posizioni)` con posizioni p_i = i / (n - 1)
        in [0, 1] (0.5 se il canale e' unico).
    """
    ordered = sorted(channels, key=lambda channel: channel.x)
    count = len(ordered)
    positions = [
        index / (count - 1) if count > 1 else SINGLE_CHANNEL_POSITION for index in range(count)
    ]
    return ordered, positions


def path_layout(channels: list[ChannelInfo]) -> tuple[list[ChannelInfo], list[float]]:
    """Ordina i canali lungo il PERCORSO fisico e li posiziona per distanza.

    Per gli effetti geometrici (chase, waterfall) l'ordine giusto non e' l'x
    crescente ma il cammino lungo la forma reale dell'impianto: una striscia
    a ferro di cavallo attorno a uno schermo deve essere percorsa su per un
    braccio, lungo la traversa e giu' per l'altro, non appiattita sull'asse
    orizzontale. Il percorso e' costruito greedy nearest-neighbor in 3D dal
    canale piu' a sinistra; le posizioni sono normalizzate sulla DISTANZA
    cumulativa (le luci lontane dalla striscia pesano per quanto distano,
    cosi' un'onda le raggiunge con il ritardo giusto).

    Args:
        channels: Canali dell'area con le rispettive posizioni.

    Returns:
        Coppia `(canali_ordinati, posizioni)` con posizioni in [0, 1]
        (0.5 se il canale e' unico).
    """
    count = len(channels)
    if count <= 1:
        return list(channels), [SINGLE_CHANNEL_POSITION] * count

    remaining = list(channels)
    current = min(remaining, key=lambda channel: (channel.x, channel.z, channel.y))
    remaining.remove(current)
    ordered = [current]
    distances = [0.0]
    while remaining:
        nearest = min(
            remaining,
            key=lambda channel: (
                (channel.x - current.x) ** 2
                + (channel.y - current.y) ** 2
                + (channel.z - current.z) ** 2
            ),
        )
        remaining.remove(nearest)
        step = math.sqrt(
            (nearest.x - current.x) ** 2
            + (nearest.y - current.y) ** 2
            + (nearest.z - current.z) ** 2
        )
        distances.append(distances[-1] + step)
        ordered.append(nearest)
        current = nearest

    total = distances[-1]
    if total <= 0.0:
        # Canali coincidenti: ripiega sul rango, come spatial_layout.
        positions = [index / (count - 1) for index in range(count)]
    else:
        positions = [distance / total for distance in distances]
    return ordered, positions


def palette_channel_base(
    palette: list[tuple[float, float, float]] | None, position: float
) -> tuple[float, float, float]:
    """Calcola hue, saturazione e scala di valore di un canale alla posizione data.

    In modalita' default (palette None o vuota) usa il gradiente di hue fisso
    lungo l'asse spaziale con saturazione piena e scala di valore 1.0. In
    modalita' palette il colore e' campionato lungo la palette (indice continuo
    f = p * (n - 1), interpolazione in HSV tra gli stop adiacenti: hue per la
    via angolare piu' corta, saturazione e valore lineari): hue e saturazione
    colorano il canale, il valore interpolato diventa una SCALA moltiplicativa
    della luminosita' (cosi' una palette scurita, es. dalle sezioni quiete,
    scurisce davvero le luci). Con un solo stop usa quel colore.

    Args:
        palette: Palette di colori `(r, g, b)` in [0, 1], o None/vuota per il
            default.
        position: Posizione normalizzata del canale in [0, 1].

    Returns:
        Tripla `(hue, sat_base, value_scale)` con componenti in [0, 1].
    """
    if not palette:
        hue = (BASE_HUE + HUE_SPAN * position) % 1.0
        return hue, 1.0, 1.0

    count = len(palette)
    if count == 1:
        return colorsys.rgb_to_hsv(*palette[0])

    # Indice continuo lungo gli stop: interpolazione in HSV tra il colore a
    # floor(f) e quello a ceil(f). L'hue segue la via angolare piu' corta sulla
    # ruota (stessa logica di _signed_hue_delta in palette.py), saturazione e
    # valore sono lineari: cosi' due stop lontani non passano dal grigio a meta'
    # strada come farebbe il lerp lineare in RGB. Con stop identici (fraction 0
    # o low == high) il risultato coincide con lo stop, quindi il default e la
    # palette a stop uguali restano invariati.
    continuous = position * (count - 1)
    low_index = int(math.floor(continuous))
    high_index = int(math.ceil(continuous))
    fraction = continuous - low_index
    low_hue, low_sat, low_val = colorsys.rgb_to_hsv(*palette[low_index])
    high_hue, high_sat, high_val = colorsys.rgb_to_hsv(*palette[high_index])
    signed_hue_delta = (high_hue - low_hue + 0.5) % 1.0 - 0.5
    hue = (low_hue + signed_hue_delta * fraction) % 1.0
    sat_base = low_sat + (high_sat - low_sat) * fraction
    value_scale = low_val + (high_val - low_val) * fraction
    return hue, sat_base, value_scale
