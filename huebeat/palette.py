"""Generatore di palette cromatiche per-brano per huebeat.

Modulo di puro calcolo (nessun logging, nessun effetto collaterale). Genera in
modo deterministico una palette di colori a partire dall'id della traccia
(sha256 dell'id, non dalla cover): la stessa traccia produce sempre la stessa
palette. I colori sono sempre saturi e gli accostamenti sono vincolati da uno
schema armonico (analogo, complementare, triadico, split-complementare) scelto
anch'esso in modo deterministico dall'hash.

Senza un brano Spotify, `generate_harmonic_palette` produce la stessa struttura a
partire da un hue di base ESPLICITO (dal circolo delle quinte del chroma) e da una
classe di altezza, cosi' la tonalita' della musica guida tinta e schema anche in
assenza di un id traccia.

La palette alimenta un gradiente spaziale interpolato in HSV tra stop adiacenti
(hue per la via angolare piu' corta, sat/valore lineari): per questo gli stop
vengono ordinati per hue crescente, cosi' che hue vicini tra stop adiacenti
diano un'interpolazione pulita, senza passare dal grigio.

Limite noto (v1): l'ordinamento per hue assoluto puo' separare un cluster a
cavallo del wrap 0/1 (es. due rossi a 0.98 e 0.02 finiscono agli estremi opposti
della palette invece di stare vicini). Accettato per la v1.

Oltre alla generazione, il modulo contiene `PaletteAnimator`: la dinamica frame
per frame della palette (crossfade al cambio brano, drift spaziale lento
sincronizzabile al beat, vividezza guidata dall'energia di sezione, bend
caldo/freddo guidato dall'audio e rotazione lenta dell'hue nel corso del brano),
da interporre tra il poller Spotify e `SpectrumPunchEffect.set_palette`.
"""

from __future__ import annotations

import colorsys
import hashlib
from typing import TYPE_CHECKING

from huebeat import tuning
from huebeat.effects import FRAME_RATE_HZ

if TYPE_CHECKING:
    # Import solo per il type hint di update: l'animatore accede a BeatInfo per
    # duck typing (bpm/locked), senza dipendere da huebeat.beat a runtime.
    from huebeat.beat import BeatInfo

# --- Dimensione della palette ---
# Numero di stop generati di default (tavolozza ricca; gli schemi hanno 5 offset
# e ciclano con i % len(scheme), i byte di saturazione/valore indicizzano col
# modulo, quindi 7 stop restano coerenti e piu' vari lungo la striscia).
DEFAULT_PALETTE_SIZE = 7

# --- Derivazione dall'hash ---
# Byte usati per la hue di base (big-endian) e divisore per portarla in [0, 1).
HUE_BYTES_START = 0
HUE_BYTES_END = 2
HUE_DIVISOR = 65536.0
# Byte che seleziona lo schema armonico, e numero di schemi disponibili.
SCHEME_BYTE_INDEX = 2
NUM_SCHEMES = 4
# Offset di partenza dei byte per saturazione e valore (poi si cicla sul digest).
SATURATION_BYTE_OFFSET = 3
VALUE_BYTE_OFFSET = 11
# Divisore per normalizzare un byte (0-255) in [0, 1].
BYTE_MAX = 255.0

# --- Range HSV (colori sempre saturi e luminosi) ---
SATURATION_MIN = 0.85
SATURATION_MAX = 1.0
VALUE_MIN = 0.7
VALUE_MAX = 1.0

# --- Schemi armonici ---
# Ogni schema e' una tupla di 5 offset di hue relativi alla hue di base (in giri
# di ruota, cioe' frazioni di 1.0). L'ordine in SCHEMES determina la mappatura
# da scheme_index (digest[SCHEME_BYTE_INDEX] % NUM_SCHEMES).
ANALOGO = (-0.12, -0.06, 0.0, 0.06, 0.12)
COMPLEMENTARE = (-0.06, 0.0, 0.06, 0.47, 0.53)
TRIADICO = (0.0, 0.03, 1 / 3, 1 / 3 + 0.03, 2 / 3)
SPLIT_COMPLEMENTARE = (-0.04, 0.0, 0.04, 0.42, 0.58)
SCHEMES = (ANALOGO, COMPLEMENTARE, TRIADICO, SPLIT_COMPLEMENTARE)

# Wrap della ruota dei colori (hue in [0, 1)).
HUE_WRAP = 1.0

# --- Warp anti-verde dell'hue di base ---
# L'hue di base grezzo e' uniforme in [0, 1): circa il 23% dei brani cade nella
# fascia verde/lime (GREEN_BAND_START..GREEN_BAND_END), poco gradita sulle luci
# e senza rimedio una volta generata. `_warp_hue` e' una biiezione monotona e
# continua di [0, 1) in [0, 1) che riduce la porzione di ingresso mappata sul
# verde a GREEN_COMPRESSION della sua larghezza naturale, ridistribuendo il
# resto sulle bande calde (rosso/arancio), magenta e blu. Sposta la PROBABILITA'
# che un brano nasca verde, non i colori verdi in se'. Derivazione dei punti di
# controllo: vedi il commento in fondo al modulo.
GREEN_BAND_START = 0.22
GREEN_BAND_END = 0.45
GREEN_COMPRESSION = 0.40

# Punti di controllo derivati (vedi in fondo al modulo). L'ingresso e' partito
# in tre tratti lineari: [0, _GREEN_IN_START) -> caldi, [_GREEN_IN_START,
# _GREEN_IN_END) -> verde compresso, [_GREEN_IN_END, 1) -> magenta/blu.
_GREEN_OUT_WIDTH = GREEN_BAND_END - GREEN_BAND_START
_GREEN_IN_WIDTH = _GREEN_OUT_WIDTH * GREEN_COMPRESSION
_FREED_WIDTH = _GREEN_OUT_WIDTH - _GREEN_IN_WIDTH
_LOWER_OUT_WIDTH = GREEN_BAND_START
_UPPER_OUT_WIDTH = HUE_WRAP - GREEN_BAND_END
_OUTER_OUT_WIDTH = _LOWER_OUT_WIDTH + _UPPER_OUT_WIDTH
# La larghezza liberata dalla compressione del verde e' redistribuita ai due
# tratti esterni in proporzione alla loro larghezza in USCITA, cosi' entrambi
# i poli (caldi e magenta/blu) guadagnano densita' senza discontinuita'.
_GREEN_IN_START = GREEN_BAND_START + _FREED_WIDTH * (_LOWER_OUT_WIDTH / _OUTER_OUT_WIDTH)
_GREEN_IN_END = _GREEN_IN_START + _GREEN_IN_WIDTH


def _warp_hue(hue: float) -> float:
    """Rimappa l'hue di base grezzo per rendere raro il verde, in modo monotono.

    Mappa piecewise-lineare, monotona crescente e continua, di [0, 1) in [0, 1):
    solo un tratto stretto di ingresso (`_GREEN_IN_START`..`_GREEN_IN_END`, largo
    GREEN_COMPRESSION volte la fascia verde) viene "steso" sull'intera fascia
    verde di uscita, mentre i due tratti esterni comprimono l'ingresso e quindi
    addensano l'uscita su caldi e magenta/blu. Un hue di base uniforme cade cosi'
    nel verde con probabilita' GREEN_COMPRESSION volte quella naturale.

    Args:
        hue: Hue di base grezzo in [0, 1).

    Returns:
        L'hue di base rimappato in [0, 1).
    """
    if hue < _GREEN_IN_START:
        return hue / _GREEN_IN_START * GREEN_BAND_START
    if hue < _GREEN_IN_END:
        return GREEN_BAND_START + (hue - _GREEN_IN_START) / _GREEN_IN_WIDTH * _GREEN_OUT_WIDTH
    return GREEN_BAND_END + (hue - _GREEN_IN_END) / (HUE_WRAP - _GREEN_IN_END) * _UPPER_OUT_WIDTH


def _build_palette(
    hue_base: float, scheme: tuple[float, ...], digest: bytes, num_colors: int
) -> list[tuple[float, float, float]]:
    """Compone gli stop da hue di base, schema armonico e byte di sat/valore.

    Cuore condiviso da `generate_palette` (hue di base dall'hash, warpato) e
    `generate_harmonic_palette` (hue di base esplicito, non warpato): applica gli
    offset dello schema all'hue di base e pesca saturazione e valore di ogni stop
    da byte distinti del digest (ciclando col modulo). Gli stop sono ordinati per
    hue crescente per alimentare il gradiente spaziale interpolato in HSV.

    Args:
        hue_base: Hue di base gia' pronto in [0, 1) (warp gia' applicato dal
            chiamante o deliberatamente omesso).
        scheme: Tupla di offset di hue relativi (frazioni di ruota).
        digest: Byte da cui derivare saturazione e valore di ogni stop.
        num_colors: Numero di stop da generare (>= 1).

    Returns:
        Lista di `num_colors` colori RGB (r, g, b in [0, 1]) ordinata per hue
        crescente.
    """
    colors: list[tuple[float, float, float]] = []
    for i in range(num_colors):
        offset = scheme[i % len(scheme)]
        hue = (hue_base + offset) % HUE_WRAP

        saturation_byte = digest[(SATURATION_BYTE_OFFSET + i) % len(digest)]
        saturation = SATURATION_MIN + (saturation_byte / BYTE_MAX) * (
            SATURATION_MAX - SATURATION_MIN
        )

        value_byte = digest[(VALUE_BYTE_OFFSET + i) % len(digest)]
        value = VALUE_MIN + (value_byte / BYTE_MAX) * (VALUE_MAX - VALUE_MIN)

        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))

    colors.sort(key=lambda rgb: colorsys.rgb_to_hsv(*rgb)[0])
    return colors


def generate_palette(
    track_id: str, num_colors: int = DEFAULT_PALETTE_SIZE
) -> list[tuple[float, float, float]]:
    """Genera una palette RGB deterministica dall'id di una traccia.

    La derivazione parte dallo sha256 dell'id: gli stessi input producono sempre
    lo stesso output, senza usare il modulo random. Hue di base, schema armonico,
    saturazione e valore di ogni stop sono estratti da byte distinti del digest.
    L'hue di base viene inoltre rimappato da `_warp_hue` per rendere raro il
    verde (la stessa mappa deterministica per ogni brano).
    I colori sono sempre saturi (saturazione >= SATURATION_MIN) e luminosi
    (valore >= VALUE_MIN). Gli stop finali sono ordinati per hue crescente per
    alimentare un gradiente spaziale interpolato in HSV tra stop adiacenti.

    Limite noto (v1): l'ordinamento per hue assoluto puo' separare un cluster a
    cavallo del wrap 0/1 (es. rossi a 0.98 e 0.02 finiscono agli estremi opposti).

    Args:
        track_id: Identificatore della traccia (qualsiasi stringa, es. l'id
            Spotify). Viene codificato in UTF-8 e passato a sha256.
        num_colors: Numero di colori da generare. Deve essere >= 1. Valori
            diversi da 5 funzionano ciclando sugli offset dello schema.

    Returns:
        Lista di `num_colors` colori RGB, ognuno una tupla (r, g, b) con
        componenti float in [0, 1], ordinata per hue crescente.

    Raises:
        ValueError: Se `num_colors` e' minore di 1.
    """
    if num_colors < 1:
        raise ValueError(f"num_colors deve essere >= 1, ricevuto {num_colors}")

    digest = hashlib.sha256(track_id.encode("utf-8")).digest()

    hue_base = int.from_bytes(digest[HUE_BYTES_START:HUE_BYTES_END], "big") / HUE_DIVISOR
    # Rimappa l'hue di base per rendere raro il verde (vedi _warp_hue): i brani
    # verde-dominanti crollano, il resto si sposta su caldi/magenta/blu.
    hue_base = _warp_hue(hue_base)
    scheme_index = digest[SCHEME_BYTE_INDEX] % NUM_SCHEMES
    scheme = SCHEMES[scheme_index]

    return _build_palette(hue_base, scheme, digest, num_colors)


def generate_harmonic_palette(
    base_hue: float, pitch_class: int, num_colors: int = DEFAULT_PALETTE_SIZE
) -> list[tuple[float, float, float]]:
    """Genera una palette armonica da un hue di base e una classe di altezza.

    Riusa la macchina degli schemi di `generate_palette` (via `_build_palette`)
    ma con hue di base ESPLICITO (`base_hue`, gia' sul circolo delle quinte,
    scelto dalla musica) e schema selezionato in modo deterministico dalla classe
    di altezza (`pitch_class % NUM_SCHEMES`): tonalita' diverse danno sia tinta
    sia struttura diverse. Saturazione e valore di ogni stop derivano da un digest
    deterministico della classe di altezza, cosi' la stessa tonalita' produce
    sempre la stessa palette.

    A differenza di `generate_palette`, il warp anti-verde NON viene applicato:
    `base_hue` e' una tinta scelta dalla tonalita' del brano (non un hash casuale
    da riequilibrare via probabilita'), quindi va rispettata cosi' com'e' anche
    quando cade nella fascia verde.

    Args:
        base_hue: Hue di base in [0, 1) dal circolo delle quinte (harmony_hue del
            chroma). Usato tale e quale (mod 1), senza warp anti-verde.
        pitch_class: Classe di altezza dominante in [0, 11] (convenzione MIR
            C=0..B=11): seleziona lo schema armonico e semina saturazione/valore.
        num_colors: Numero di stop da generare (>= 1).

    Returns:
        Lista di `num_colors` colori RGB (r, g, b in [0, 1]) ordinata per hue
        crescente.

    Raises:
        ValueError: Se `num_colors` e' minore di 1.
    """
    if num_colors < 1:
        raise ValueError(f"num_colors deve essere >= 1, ricevuto {num_colors}")

    scheme = SCHEMES[pitch_class % NUM_SCHEMES]
    # Digest deterministico dalla classe di altezza: da' a saturazione e valore
    # la stessa varieta' del percorso per-brano, restando stabile per la stessa
    # tonalita' (stesso pitch class -> stessa palette).
    digest = hashlib.sha256(str(pitch_class).encode("ascii")).digest()
    return _build_palette(base_hue % HUE_WRAP, scheme, digest, num_colors)


# --- Dinamica della palette (PaletteAnimator) ---
# Ogni componente e' disattivabile singolarmente (per il collaudo sul campo).
# NOTA tuning runtime: CROSSFADE_SECONDS, DRIFT_PERIOD_SECONDS e
# HUE_ROTATION_SECONDS_PER_TURN sono i DEFAULT storici; i valori vivi arrivano
# da huebeat.tuning.live (regolabili a caldo dal frontend).
CROSSFADE_ENABLED = True
DRIFT_ENABLED = True
ENERGY_ENABLED = True
HUE_ROTATION_ENABLED = True
WARMTH_ENABLED = True
HARMONY_ENABLED = True
# Durata del crossfade tra palette al cambio brano (secondi).
CROSSFADE_SECONDS = 2.0
# Periodo del drift: tempo per un giro completo della palette lungo la striscia.
DRIFT_PERIOD_SECONDS = 30.0
# Numero di beat per un giro completo della palette quando il beat e' lockato:
# lega il drift al tempo del brano invece che a un periodo fisso in secondi.
DRIFT_BEATS_PER_TURN = 16.0
# Avanzamento extra di fase per un onset a intensita' piena (frazione di giro).
DRIFT_ONSET_KICK = 0.01
# Vividezza a energia di sezione zero: scala minima di valore e saturazione
# degli stop (a energia piena la scala e' 1.0, colori pieni).
ENERGY_VALUE_FLOOR = 0.55
ENERGY_SATURATION_FLOOR = 0.75
# Hue shift massimo alle sezioni piu' energetiche (frazione di ruota).
ENERGY_HUE_SHIFT_MAX = 0.05
# Rotazione lenta dell'hue nel corso del brano: un giro completo della ruota in
# HUE_ROTATION_SECONDS_PER_TURN secondi (12 min: percettibile ma gentile, circa
# un quarto di giro in 3 min). Si azzera a ogni cambio brano.
HUE_ROTATION_SECONDS_PER_TURN = 720.0
# Bend caldo/freddo guidato dall'audio: hue di ogni stop attratto verso un polo
# ambra (caldo) o blu (freddo) al piu' per WARMTH_BEND_MAX del percorso, cosi'
# che l'identita' cromatica del brano resti riconoscibile.
WARM_POLE_HUE = 0.05
COLD_POLE_HUE = 0.58
WARMTH_BEND_MAX = 0.25
# Bend armonico: hue di ogni stop attratto verso l'hue del brano (dal circolo
# delle quinte, harmony_hue) al piu' per HARMONY_BEND_MAX del percorso, scalato
# dal knob harmony_gain e dalla forza armonica del frame. Stesso ordine di
# grandezza del bend caldo/freddo, cosi' l'identita' cromatica del brano resta.
HARMONY_BEND_MAX = 0.22

RGBColor = tuple[float, float, float]


def _lerp_hsv(start: RGBColor, end: RGBColor, fraction: float) -> RGBColor:
    """Interpola due colori RGB in HSV, hue per la via angolare piu' corta.

    Saturazione e valore sono interpolati linearmente; l'hue segue il verso
    piu' breve sulla ruota (via `_signed_hue_delta`), cosi' due stop lontani
    non passano dal grigio a meta' strada come farebbe il lerp lineare in RGB
    (era la causa delle bande slavate sugli schemi complementare/triadico/split).
    Con `start` uguale a `end` restituisce il colore invariato, senza artefatti
    di conversione.

    Args:
        start: Colore di partenza (componenti in [0, 1]).
        end: Colore di arrivo (componenti in [0, 1]).
        fraction: Posizione dell'interpolazione in [0, 1].

    Returns:
        Il colore interpolato.
    """
    if start == end:
        return start
    start_hue, start_sat, start_val = colorsys.rgb_to_hsv(*start)
    end_hue, end_sat, end_val = colorsys.rgb_to_hsv(*end)
    hue = (start_hue + _signed_hue_delta(start_hue, end_hue) * fraction) % HUE_WRAP
    saturation = start_sat + (end_sat - start_sat) * fraction
    value = start_val + (end_val - start_val) * fraction
    return colorsys.hsv_to_rgb(hue, saturation, value)


def _sample_cyclic(palette: list[RGBColor], position: float) -> RGBColor:
    """Campiona una palette trattata come gradiente CICLICO.

    Gli n stop sono i punti k/n del ciclo; tra uno stop e il successivo (incluso
    il raccordo ultimo -> primo) si interpola in HSV (hue per la via piu' corta).

    Args:
        palette: Stop della palette (non vuota).
        position: Posizione sul ciclo in [0, 1) (valori fuori vengono avvolti).

    Returns:
        Il colore campionato.
    """
    count = len(palette)
    scaled = (position % HUE_WRAP) * count
    index = int(scaled) % count
    fraction = scaled - int(scaled)
    return _lerp_hsv(palette[index], palette[(index + 1) % count], fraction)


def _resample(palette: list[RGBColor], count: int) -> list[RGBColor]:
    """Ricampiona una palette a `count` stop con interpolazione lineare NON ciclica.

    Serve al crossfade quando le due palette hanno numero di stop diverso.

    Args:
        palette: Stop di partenza (non vuota).
        count: Numero di stop desiderato (>= 1).

    Returns:
        Lista di `count` stop equispaziati lungo la palette.
    """
    if len(palette) == count:
        return list(palette)
    if len(palette) == 1 or count == 1:
        return [palette[0]] * count
    resampled: list[RGBColor] = []
    for index in range(count):
        continuous = index / (count - 1) * (len(palette) - 1)
        low = int(continuous)
        high = min(low + 1, len(palette) - 1)
        resampled.append(_lerp_hsv(palette[low], palette[high], continuous - low))
    return resampled


def _energy_adjust(color: RGBColor, section_energy: float) -> RGBColor:
    """Modula un colore con l'energia di sezione corrente.

    Sezioni quiete: colore piu' scuro e desaturato (scala verso i pavimenti
    ENERGY_VALUE_FLOOR / ENERGY_SATURATION_FLOOR). Sezioni cariche: colore
    pieno con un leggero hue shift proporzionale (fino a ENERGY_HUE_SHIFT_MAX).

    Args:
        color: Colore RGB di partenza (componenti in [0, 1]).
        section_energy: Energia di sezione in [0, 1].

    Returns:
        Il colore modulato.
    """
    hue, saturation, value = colorsys.rgb_to_hsv(*color)
    hue = (hue + ENERGY_HUE_SHIFT_MAX * section_energy) % HUE_WRAP
    saturation *= ENERGY_SATURATION_FLOOR + (1.0 - ENERGY_SATURATION_FLOOR) * section_energy
    value *= ENERGY_VALUE_FLOOR + (1.0 - ENERGY_VALUE_FLOOR) * section_energy
    return colorsys.hsv_to_rgb(hue, saturation, value)


def _signed_hue_delta(hue: float, target: float) -> float:
    """Distanza circolare CON SEGNO da un hue a un altro sulla ruota.

    Il segno indica il verso piu' breve lungo la ruota (in [0, 1)) per andare
    da `hue` a `target`: positivo in avanti, negativo indietro.

    Args:
        hue: Hue di partenza in [0, 1).
        target: Hue di arrivo in [0, 1).

    Returns:
        La distanza con segno in [-0.5, 0.5].
    """
    # (delta + 0.5) % 1 - 0.5 riporta la differenza grezza nell'intervallo
    # simmetrico [-0.5, 0.5), scegliendo cosi' il verso piu' corto.
    return (target - hue + 0.5) % HUE_WRAP - 0.5


def _warmth_bend(color: RGBColor, warmth: float) -> RGBColor:
    """Piega l'hue di un colore verso il polo caldo o freddo secondo l'audio.

    Con warmth positivo l'hue e' attratto verso WARM_POLE_HUE (ambra), con
    warmth negativo verso COLD_POLE_HUE (blu). Lo spostamento e' al piu'
    WARMTH_BEND_MAX del percorso verso il polo, scalato da |warmth|, cosi' che
    l'identita' cromatica del brano resti riconoscibile. Con warmth 0 il colore
    non cambia.

    Args:
        color: Colore RGB di partenza (componenti in [0, 1]).
        warmth: Bilanciamento caldo/freddo in [-1, 1] (positivo = caldo).

    Returns:
        Il colore con l'hue piegato verso il polo scelto.
    """
    if warmth == 0.0:
        return color
    hue, saturation, value = colorsys.rgb_to_hsv(*color)
    pole = WARM_POLE_HUE if warmth > 0.0 else COLD_POLE_HUE
    delta = _signed_hue_delta(hue, pole)
    hue = (hue + delta * abs(warmth) * WARMTH_BEND_MAX) % HUE_WRAP
    return colorsys.hsv_to_rgb(hue, saturation, value)


def _harmony_bend(color: RGBColor, harmony_hue: float, bend: float) -> RGBColor:
    """Piega l'hue di un colore verso l'hue armonico del brano.

    L'hue e' attratto verso `harmony_hue` (dal circolo delle quinte) lungo la via
    angolare piu' corta, di una frazione `bend` del percorso. Con `bend` 0 il
    colore non cambia; saturazione e valore restano intatti, cosi' l'identita'
    cromatica del brano resta riconoscibile.

    Args:
        color: Colore RGB di partenza (componenti in [0, 1]).
        harmony_hue: Hue armonico bersaglio in [0, 1) (circolo delle quinte).
        bend: Frazione del percorso verso il bersaglio in [0, 1] (0 = nessuna
            piega).

    Returns:
        Il colore con l'hue piegato verso l'hue armonico.
    """
    if bend == 0.0:
        return color
    hue, saturation, value = colorsys.rgb_to_hsv(*color)
    delta = _signed_hue_delta(hue, harmony_hue)
    hue = (hue + delta * bend) % HUE_WRAP
    return colorsys.hsv_to_rgb(hue, saturation, value)


def _rotate_hue(color: RGBColor, rotation: float) -> RGBColor:
    """Ruota additivamente l'hue di un colore lungo la ruota.

    Args:
        color: Colore RGB di partenza (componenti in [0, 1]).
        rotation: Rotazione da sommare all'hue (frazione di ruota, mod 1).

    Returns:
        Il colore con l'hue ruotato.
    """
    hue, saturation, value = colorsys.rgb_to_hsv(*color)
    hue = (hue + rotation) % HUE_WRAP
    return colorsys.hsv_to_rgb(hue, saturation, value)


class PaletteAnimator:
    """Dinamica frame per frame della palette: crossfade, drift, energia, bend, rotazione.

    Si interpone tra la sorgente della palette (il poller Spotify) e l'effetto:
    `set_target` riceve la palette del brano corrente (o None per la modalita'
    default), `update` va chiamata una volta per frame e restituisce la palette
    da passare a `SpectrumPunchEffect.set_palette`.

    Componenti (ognuna disattivabile con la sua costante *_ENABLED):

    - crossfade: al cambio di palette interpola in RGB dalla palette mostrata
      a quella nuova in `CROSSFADE_SECONDS` invece di scattare;
    - drift: la palette, trattata come gradiente ciclico, scorre lungo la
      striscia; se il beat e' lockato il passo e' legato al tempo del brano
      (`DRIFT_BEATS_PER_TURN` beat per giro), altrimenti a `DRIFT_PERIOD_SECONDS`;
      in entrambi i casi gli onset aggiungono uno scatto `DRIFT_ONSET_KICK`;
    - energia: vividezza (valore/saturazione) e leggero hue shift guidati
      dall'energia di sezione;
    - bend caldo/freddo: l'hue di ogni stop e' attratto verso un polo ambra o
      blu secondo il parametro `warmth` (identita' del brano preservata);
    - bend armonico: l'hue di ogni stop e' attratto verso l'hue del brano (dal
      circolo delle quinte, `harmony_hue`) scalato dal knob `harmony_gain` e
      dalla forza armonica; con harmony_hue None o knob 0 non fa nulla;
    - rotazione hue: l'intera palette ruota lentamente lungo la ruota nel corso
      del brano (un giro in `HUE_ROTATION_SECONDS_PER_TURN`), azzerata a ogni
      cambio brano.

    Ordine di applicazione per frame: crossfade -> drift -> energia -> bend
    caldo/freddo -> bend armonico -> rotazione hue.

    Da usare interamente nel thread del loop di rendering: non e' thread-safe
    (a differenza di `SpectrumPunchEffect.set_palette`, che lo e').
    """

    def __init__(self) -> None:
        """Inizializza l'animatore in modalita' default (nessuna palette)."""
        self._target: list[RGBColor] | None = None
        # Palette di partenza del crossfade in corso; None = nessun crossfade.
        self._from: list[RGBColor] | None = None
        # Progresso del crossfade in [0, 1]; 1.0 = completato. Il passo viene
        # calcolato all'avvio di ogni crossfade dal knob runtime.
        self._blend: float = 1.0
        self._blend_step: float = 1.0 / (CROSSFADE_SECONDS * FRAME_RATE_HZ)
        # Fase del drift in [0, 1); il passo per frame e' derivato a ogni
        # frame dal knob runtime (drift_period_seconds).
        self._phase: float = 0.0
        # Fase della rotazione lenta dell'hue in [0, 1); passo per frame
        # derivato a ogni frame dal knob runtime. Si azzera a ogni cambio
        # brano (set_target con palette diversa).
        self._hue_rotation: float = 0.0

    def set_target(self, palette: list[RGBColor] | None) -> None:
        """Imposta la palette di destinazione (o None per la modalita' default).

        Chiamabile a ogni frame: se la palette non e' cambiata non fa nulla.
        Al cambio verso una nuova palette avvia un crossfade dalla palette
        attualmente mostrata; verso/da None scatta subito (la modalita' default
        non e' una palette, non c'e' niente da interpolare).

        Args:
            palette: Stop della nuova palette, o None.
        """
        if palette is self._target:
            return
        if palette == self._target:
            # Palette uguale ma oggetto nuovo: aggiorna il riferimento e basta.
            self._target = palette
            return
        # Palette diversa: e' un nuovo brano, la rotazione lenta riparte da zero
        # cosi' ogni brano comincia dalla sua identita' cromatica.
        self._hue_rotation = 0.0
        crossfade_seconds = tuning.live.crossfade_seconds
        if (
            palette is None
            or self._target is None
            or not CROSSFADE_ENABLED
            or crossfade_seconds <= 0.0
        ):
            self._from = None
            self._blend = 1.0
            self._target = list(palette) if palette is not None else None
            return
        self._blend_step = 1.0 / (crossfade_seconds * FRAME_RATE_HZ)
        self._from = self._blended_base()
        self._blend = 0.0
        self._target = list(palette)

    def update(
        self,
        onset_strength: float,
        section_energy: float,
        warmth: float = 0.0,
        beat: BeatInfo | None = None,
        harmony_hue: float | None = None,
        harmony_strength: float = 0.0,
    ) -> list[RGBColor] | None:
        """Avanza la dinamica di un frame e restituisce la palette corrente.

        Ordine di applicazione: crossfade -> drift -> energia -> bend
        caldo/freddo -> bend armonico -> rotazione hue.

        Args:
            onset_strength: Intensita' dell'onset del frame in [0, 1].
            section_energy: Energia di sezione del frame in [0, 1].
            warmth: Bilanciamento caldo/freddo in [-1, 1] (positivo = brano
                caldo/scuro attratto verso l'ambra, negativo = brillante/freddo
                attratto verso il blu, 0 = nessun bend).
            beat: Info di beat del frame, o None. Se presente e lockato con
                bpm > 0, il drift avanza a tempo (DRIFT_BEATS_PER_TURN beat per
                giro) invece che a DRIFT_PERIOD_SECONDS. Accesso per duck
                typing (attributi bpm/locked).
            harmony_hue: Hue armonico del brano in [0, 1) dal circolo delle
                quinte, o None se non stimato. Con None (o knob/forza a 0) il
                percorso resta invariato.
            harmony_strength: Forza della stima armonica in [0, 1]: scala il
                bend insieme al knob harmony_gain.

        Returns:
            Gli stop della palette da mostrare in questo frame, o None se non
            c'e' una palette attiva (modalita' default).
        """
        if self._target is None:
            return None

        if self._blend < 1.0:
            self._blend = min(1.0, self._blend + self._blend_step)
        base = self._blended_base()

        if DRIFT_ENABLED:
            if beat is not None and beat.locked and beat.bpm > 0.0:
                # Passo legato al tempo: bpm/60 beat al secondo, un giro ogni
                # DRIFT_BEATS_PER_TURN beat.
                drift_step = (beat.bpm / 60.0) / (DRIFT_BEATS_PER_TURN * FRAME_RATE_HZ)
            else:
                drift_step = 1.0 / (tuning.live.drift_period_seconds * FRAME_RATE_HZ)
            self._phase = (self._phase + drift_step + DRIFT_ONSET_KICK * onset_strength) % HUE_WRAP
            count = len(base)
            base = [_sample_cyclic(base, index / count + self._phase) for index in range(count)]

        if ENERGY_ENABLED:
            base = [_energy_adjust(color, section_energy) for color in base]

        if WARMTH_ENABLED:
            base = [_warmth_bend(color, warmth) for color in base]

        if HARMONY_ENABLED and harmony_hue is not None:
            # Piega verso l'hue del brano scalando col knob e la forza armonica.
            # Con knob 0 o forza 0 il bend e' 0 e il percorso resta invariato.
            harmony_bend = tuning.live.harmony_gain * harmony_strength * HARMONY_BEND_MAX
            if harmony_bend > 0.0:
                base = [_harmony_bend(color, harmony_hue, harmony_bend) for color in base]

        if HUE_ROTATION_ENABLED:
            hue_rotation_step = 1.0 / (tuning.live.hue_rotation_seconds_per_turn * FRAME_RATE_HZ)
            self._hue_rotation = (self._hue_rotation + hue_rotation_step) % HUE_WRAP
            base = [_rotate_hue(color, self._hue_rotation) for color in base]

        return base

    def _blended_base(self) -> list[RGBColor]:
        """Restituisce la palette base corrente (dopo il solo crossfade).

        Returns:
            La palette target se non c'e' crossfade in corso, altrimenti
            l'interpolazione tra partenza e target al progresso corrente.
        """
        assert self._target is not None
        if self._from is None or self._blend >= 1.0:
            return list(self._target)
        count = max(len(self._from), len(self._target))
        from_stops = _resample(self._from, count)
        to_stops = _resample(self._target, count)
        return [
            _lerp_hsv(start, end, self._blend)
            for start, end in zip(from_stops, to_stops, strict=True)
        ]


# --- Derivazione della mappa anti-verde (_warp_hue) ---
# Obiettivo: un hue di base uniforme in [0, 1) cade nella fascia verde/lime
# [GREEN_BAND_START, GREEN_BAND_END] (larghezza _GREEN_OUT_WIDTH = 0.23, cioe'
# ~23% dei brani); vogliamo portarlo a GREEN_COMPRESSION (40%) di quel valore,
# ~9%, spostando il resto su caldi e magenta/blu, con una mappa che non crei
# salti (monotona e continua) ne' introduca nuove soglie.
#
# Idea: una biiezione lineare a tratti f: [0, 1) -> [0, 1). La probabilita' che
# f(hue) cada nel verde e' la larghezza della sua controimmagine. Basta quindi
# che solo un tratto di ingresso largo _GREEN_IN_WIDTH = GREEN_COMPRESSION *
# _GREEN_OUT_WIDTH venga mappato sull'intera fascia verde di uscita: quel tratto
# ha pendenza > 1 (stira), gli altri due pendenza < 1 (comprimono l'ingresso e
# addensano l'uscita sui poli caldi e magenta/blu).
#
# I tre tratti in USCITA sono fissi: [0, GREEN_BAND_START] (caldi),
# [GREEN_BAND_START, GREEN_BAND_END] (verde), [GREEN_BAND_END, 1] (magenta/blu).
# In INGRESSO il verde occupa solo _GREEN_IN_WIDTH; la larghezza liberata
# _FREED_WIDTH = _GREEN_OUT_WIDTH - _GREEN_IN_WIDTH si aggiunge ai due tratti
# esterni in proporzione alla loro larghezza di uscita (_LOWER_OUT_WIDTH e
# _UPPER_OUT_WIDTH), cosi' nessuno dei due poli e' privilegiato oltre il suo
# peso naturale e le pendenze dei due tratti esterni restano uguali tra loro
# (raccordo senza spigoli marcati). Da qui _GREEN_IN_START e _GREEN_IN_END.
# Con i valori attuali: verde in ingresso [~0.259, ~0.351] (largo 0.092) contro
# [0.22, 0.45] in uscita; P(verde) passa da 0.23 a 0.092.
