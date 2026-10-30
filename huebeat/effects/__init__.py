"""Package effetti di huebeat.

Espone i tipi condivisi, gli effetti disponibili e la factory `create_effect`
che costruisce l'effetto scelto in config. Interfaccia comune degli effetti
in `huebeat.effects.base`.
"""

from huebeat.effects.base import FRAME_RATE_HZ, ChannelInfo
from huebeat.effects.spectrum import SpectrumPunchEffect

# Nomi degli effetti selezionabili da config (chiave `effect`).
EFFECT_NAMES = ("spectrum", "chase", "pulse", "auto", "waterfall", "adaptive")


def create_effect(name: str, channels: list[ChannelInfo]):
    """Costruisce l'effetto con il nome dato.

    Gli import sono locali per non pagare (ne' rompere) i moduli degli effetti
    non usati all'import del package.

    Args:
        name: Uno dei nomi in `EFFECT_NAMES`.
        channels: Canali dell'area con posizioni (e ruoli opzionali).

    Returns:
        L'istanza dell'effetto richiesto.

    Raises:
        ValueError: Se `name` non e' un effetto conosciuto.
    """
    if name == "spectrum":
        return SpectrumPunchEffect(channels)
    if name == "chase":
        from huebeat.effects.chase import ChaseEffect

        return ChaseEffect(channels)
    if name == "pulse":
        from huebeat.effects.pulse import PulseEffect

        return PulseEffect(channels)
    if name == "auto":
        from huebeat.effects.auto import AutoEffect

        return AutoEffect(channels)
    if name == "waterfall":
        from huebeat.effects.waterfall import WaterfallEffect

        return WaterfallEffect(channels)
    if name == "adaptive":
        from huebeat.effects.adaptive import AdaptiveEffect

        return AdaptiveEffect(channels)
    raise ValueError(f"Effetto sconosciuto: {name!r}. Disponibili: {', '.join(EFFECT_NAMES)}.")


__all__ = [
    "EFFECT_NAMES",
    "FRAME_RATE_HZ",
    "ChannelInfo",
    "SpectrumPunchEffect",
    "create_effect",
]
