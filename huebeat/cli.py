"""CLI di huebeat: pairing del Bridge, test luci, sync musica-luci e app desktop.

Espone cinque sottocomandi (`pair`, `test-lights`, `run`, `spotify-login`,
`app`) via argparse. Gli import dei moduli `bridge`, `streaming` e `spotify` sono locali
alle funzioni dei sottocomandi: cosi' questo file resta importabile anche se
quei moduli non ci sono ancora (utile per test isolati della logica di update
del file .env, e perche' `spotify` puo' essere sviluppato in parallelo).

Convenzione output: `print` solo per l'output utente della CLI (istruzioni,
conferme interattive); `logging` per la diagnostica.
"""

import argparse
import colorsys
import contextlib
import logging
import signal
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from huebeat import paths
from huebeat.config import settings
from huebeat.effects.base import palette_channel_base
from huebeat.envfile import update_env_values
from huebeat.session import close_session, setup_session

logger = logging.getLogger(__name__)

# --- Costanti di comportamento ---
# Durata dello sweep di test dei canali, in secondi.
TEST_SWEEP_SECONDS = 10.0
# Velocita' angolare della rotazione di hue nel tempo (giri/secondo).
SWEEP_HUE_SPEED = 0.2
# Sfasamento di hue tra un canale e il successivo (frazione di ruota).
SWEEP_HUE_PER_CHANNEL = 0.15
# Saturazione piena per tutti i colori generati.
FULL_SATURATION = 1.0
# Valore (luminosita') pieno per lo sweep di test.
FULL_VALUE = 1.0

# --- Exit code ---
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_PAIRED = 2

# --- Gestione file .env (nella home di huebeat.paths) ---
ENV_USERNAME_KEY = "HUE_USERNAME"
ENV_CLIENTKEY_KEY = "HUE_CLIENTKEY"

# --- Pairing ---
PAIR_TIMEOUT_SECONDS = 60.0


def _update_env_file(path: Path, username: str, clientkey: str) -> None:
    """Aggiorna username e clientkey nel file .env preservando il resto.

    Delegato a `envfile.update_env_values`: firma mantenuta per cmd_pair (le due
    chiavi del Bridge sono il caso storico). La logica di preservazione di
    commenti e chiavi vive ora in `huebeat.envfile`.

    Args:
        path: Percorso del file .env.
        username: Valore da scrivere per HUE_USERNAME.
        clientkey: Valore da scrivere per HUE_CLIENTKEY.
    """
    update_env_values(path, {ENV_USERNAME_KEY: username, ENV_CLIENTKEY_KEY: clientkey})


def _require_paired() -> None:
    """Verifica che almeno un bridge sia accoppiato (store o .env legacy).

    Raises:
        SystemExit: Con codice EXIT_NOT_PAIRED se il pairing manca.
    """
    from huebeat.bridgestore import any_bridge_configured

    if not any_bridge_configured():
        print("Nessun bridge accoppiato. Esegui `python -m huebeat pair` o il wizard dell'app.")
        raise SystemExit(EXIT_NOT_PAIRED)


def cmd_pair(_args: argparse.Namespace) -> int:
    """Esegue il pairing col Bridge e salva i secrets nel file .env.

    Args:
        _args: Argomenti del sottocomando (non usati).

    Returns:
        Exit code del comando.
    """
    from huebeat.bridge import discover_bridge, pair

    host = settings.bridge_host or discover_bridge()

    print(f"Bridge trovato: {host}")
    print(f"Premi il pulsante rotondo sul Bridge entro {int(PAIR_TIMEOUT_SECONDS)} secondi...")

    username, clientkey = pair(host, timeout_s=PAIR_TIMEOUT_SECONDS)

    env_path = paths.data_dir() / paths.ENV_FILENAME
    _update_env_file(env_path, username, clientkey)

    print(f"Pairing riuscito. Secrets salvati in {env_path}.")
    print(
        f'Suggerimento: metti bridge_host = "{host}" in config.toml per saltare '
        "l'auto-discovery ai prossimi avvii."
    )
    return EXIT_OK


def cmd_spotify_login(_args: argparse.Namespace) -> int:
    """Esegue il login OAuth PKCE a Spotify e salva i token nella cwd.

    Richiede `spotify_client_id` in config (chiave SPOTIFY_CLIENT_ID nel .env).
    Avvia il flusso interattivo di `SpotifyAuth` e salva i token nel file
    `.spotify_tokens.json` accanto al .env, cosi' `run` potra' leggere il
    "now playing" e generarne la palette per-brano.

    Args:
        _args: Argomenti del sottocomando (non usati).

    Returns:
        Exit code del comando.
    """
    if settings.spotify_client_id is None:
        print("Client id Spotify mancante: aggiungi SPOTIFY_CLIENT_ID in .env e riprova.")
        return EXIT_NOT_PAIRED

    from huebeat.spotify import get_auth

    client_id = settings.spotify_client_id.get_secret_value()
    cache_path = paths.data_dir() / paths.SPOTIFY_TOKENS_FILENAME
    # Istanza condivisa per file di token (registro di modulo): poller e server
    # di controllo usano lo STESSO autenticatore sullo stesso file.
    auth = get_auth(client_id, cache_path)
    auth.interactive_login()

    print(f"Login Spotify riuscito. Token salvati in {cache_path}.")
    return EXIT_OK


def cmd_test_lights(_args: argparse.Namespace) -> int:
    """Fa uno sweep di colori sui canali dell'area per verificare le luci.

    Args:
        _args: Argomenti del sottocomando (non usati).

    Returns:
        Exit code del comando.
    """
    _require_paired()

    session = setup_session()
    print(
        f"Test area '{session.area_name}': {len(session.channels)} canali a "
        f"{settings.stream_fps} fps per {int(TEST_SWEEP_SECONDS)} secondi. Ctrl+C per fermare."
    )

    frame_interval = 1.0 / settings.stream_fps
    try:
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= TEST_SWEEP_SECONDS:
                break
            entries = _sweep_entries(session.channels, elapsed)
            session.stream.send(entries)  # type: ignore[attr-defined]
            time.sleep(frame_interval)
    except KeyboardInterrupt:
        print("\nInterrotto.")
    finally:
        close_session(session)

    return EXIT_OK


def _sweep_entries(
    channels: list[dict], elapsed: float
) -> list[tuple[int, tuple[float, float, float]]]:
    """Genera i colori di un frame dello sweep di test.

    Ogni canale ruota sulla ruota HSV con uno sfasamento proprio, cosi' i canali
    si distinguono a occhio.

    Args:
        channels: Dict grezzi dei canali da illuminare (campo ``channel_id``).
        elapsed: Secondi trascorsi dall'inizio dello sweep.

    Returns:
        Lista di `(channel_id, (r, g, b))` con componenti float in [0, 1].
    """
    entries: list[tuple[int, tuple[float, float, float]]] = []
    for index, channel in enumerate(channels):
        hue = (elapsed * SWEEP_HUE_SPEED + index * SWEEP_HUE_PER_CHANNEL) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, FULL_SATURATION, FULL_VALUE)
        entries.append((channel["channel_id"], rgb))
    return entries


def _maybe_start_spotify_poller() -> "object | None":
    """Avvia il poller Spotify se configurato e autenticato, altrimenti None.

    Se `spotify_client_id` non e' impostato, l'integrazione Spotify e' disattiva
    e la funzione ritorna None (comportamento storico invariato). Se e' impostato
    ma mancano i token, avvisa l'utente di eseguire `spotify-login` e prosegue
    con la palette di default (ritorna None) senza bloccare. Con token presenti
    crea e avvia un `NowPlayingPoller`.

    Returns:
        Il `NowPlayingPoller` avviato, oppure None se Spotify e' disattivo o non
        autenticato.
    """
    if settings.spotify_client_id is None:
        return None

    from huebeat.spotify import NowPlayingPoller, get_auth

    client_id = settings.spotify_client_id.get_secret_value()
    cache_path = paths.data_dir() / paths.SPOTIFY_TOKENS_FILENAME
    # Istanza condivisa (registro di modulo) col server di controllo: un solo
    # autenticatore per file di token, niente rinnovi paralleli sullo stesso file.
    auth = get_auth(client_id, cache_path)
    if not auth.has_tokens():
        print(
            "Spotify configurato ma non autenticato: esegui 'python -m huebeat "
            "spotify-login' per la palette per-brano. Proseguo con la palette di default."
        )
        return None

    poller = NowPlayingPoller(auth)
    poller.start()
    print("Spotify collegato: ogni brano avra' la sua palette generata.")
    return poller


# --- Palette armonica senza Spotify (guidata dal chroma) ---
# Secondi per cui il pitch class dominante candidato deve restare stabile prima
# di generare e applicare la palette armonica: isteresi anti-sfarfallio, la
# tonalita' e' un carattere della sezione, non insegue accordo per accordo.
HARMONIC_PALETTE_HOLD_SECONDS = 8.0
# Nomi delle 12 classi di altezza (convenzione MIR C=0..B=11, pc 0 = Do),
# indicizzati dal pitch class restituito da `_hue_to_pitch_class`, per il log.
_PITCH_CLASS_NAMES = (
    "Do",
    "Do#",
    "Re",
    "Re#",
    "Mi",
    "Fa",
    "Fa#",
    "Sol",
    "Sol#",
    "La",
    "La#",
    "Si",
)


def _hue_to_pitch_class(hue: float) -> int:
    """Inverte la mappa del circolo delle quinte: hue -> classe di altezza.

    `chroma.py` mappa la classe di altezza sull'hue con
    ``hue = ((pc * FIFTH_STEP_SEMITONES) mod NUM_PITCH_CLASSES) / NUM_PITCH_CLASSES``.
    Poiche' FIFTH_STEP_SEMITONES (7) e' inverso di se stesso modulo 12
    (7*7 = 49 = 1 mod 12), la stessa moltiplicazione per 7 inverte la mappa:
    ``pc = (round(hue * 12) * 7) mod 12``. Il round assorbe l'errore in virgola
    mobile (gli hue validi sono multipli esatti di 1/12); il mod esterno copre
    l'eventuale arrotondamento a 12.

    Args:
        hue: Hue tonale in [0, 1) dal circolo delle quinte.

    Returns:
        La classe di altezza in [0, 11] (convenzione MIR C=0..B=11).
    """
    from huebeat.chroma import FIFTH_STEP_SEMITONES, NUM_PITCH_CLASSES

    step = round(hue * NUM_PITCH_CLASSES) % NUM_PITCH_CLASSES
    return (step * FIFTH_STEP_SEMITONES) % NUM_PITCH_CLASSES


class HarmonicPaletteTracker:
    """Palette armonica dal chroma quando manca un brano Spotify.

    Macchina a stati con isteresi: il pitch class dominante candidato (dedotto da
    `harmony_hue` invertendo il circolo delle quinte) deve restare stabile per
    `HARMONIC_PALETTE_HOLD_SECONDS` prima che venga generata e applicata una nuova
    palette armonica. Un `harmony_hue` None (tonalita' non definita) non azzera la
    candidatura in corso: si tiene l'ultima palette. Un pitch class concreto
    diverso dal candidato riavvia il conteggio (le oscillazioni non cambiano
    nulla). Con `harmony_gain` <= 0 il meccanismo e' spento: nessuna palette,
    comportamento identico a prima.

    Da usare nel thread del loop di rendering: non e' thread-safe.
    """

    def __init__(self) -> None:
        """Inizializza il tracker senza alcuna palette armonica."""
        # Ultima palette armonica applicata (tenuta finche' non ne matura una nuova).
        self._palette: list[tuple[float, float, float]] | None = None
        # Pitch class dell'ultima palette applicata (per non rigenerarla uguale).
        self._applied_pc: int | None = None
        # Pitch class candidato in attesa di stabilizzarsi, e da quando (monotonic).
        self._candidate_pc: int | None = None
        self._candidate_since: float = 0.0

    def update(
        self, harmony_hue: float | None, harmony_gain: float, now: float
    ) -> tuple[list[tuple[float, float, float]] | None, str | None]:
        """Avanza la macchina a stati e restituisce (palette, nota nuova).

        Args:
            harmony_hue: Hue tonale del frame in [0, 1) dal circolo delle quinte,
                o None se la tonalita' non e' definita.
            harmony_gain: Knob armonia; <= 0 spegne il meccanismo.
            now: Istante monotonic del frame (per l'isteresi).

        Returns:
            Coppia `(palette, nota)`: la palette armonica corrente (o None se
            spenta o mai generata) e il nome della nota SOLO nel frame in cui una
            palette nuova viene generata (per il log INFO del chiamante),
            altrimenti None.
        """
        if harmony_gain <= 0.0:
            # Meccanismo spento: nessuna palette, stato azzerato.
            self._palette = None
            self._applied_pc = None
            self._candidate_pc = None
            return None, None

        if harmony_hue is None:
            # Tonalita' non definita: si tiene l'ultima palette, candidatura ferma.
            return self._palette, None

        pitch_class = _hue_to_pitch_class(harmony_hue)
        if pitch_class == self._applied_pc:
            # Gia' la palette mostrata: nessuna candidatura pendente.
            self._candidate_pc = None
            return self._palette, None

        if pitch_class != self._candidate_pc:
            # Candidato nuovo: (ri)avvia il timer di stabilita'.
            self._candidate_pc = pitch_class
            self._candidate_since = now
            return self._palette, None

        if now - self._candidate_since < HARMONIC_PALETTE_HOLD_SECONDS:
            # Candidato stabile ma non ancora abbastanza a lungo.
            return self._palette, None

        # Candidato stabile da almeno HARMONIC_PALETTE_HOLD_SECONDS: genera e applica.
        from huebeat.palette import generate_harmonic_palette

        self._palette = generate_harmonic_palette(harmony_hue, pitch_class)
        self._applied_pc = pitch_class
        self._candidate_pc = None
        return self._palette, _PITCH_CLASS_NAMES[pitch_class]


def _select_palette_target(
    spotify_palette: list | None,
    harmonic_tracker: HarmonicPaletteTracker,
    harmony_hue: float | None,
    harmony_gain: float,
    now: float,
) -> tuple[list | None, str | None]:
    """Sceglie il target della palette: il brano Spotify vince, altrimenti armonica.

    Con un brano Spotify attivo (palette non None) restituisce quella palette e
    non consulta il tracker armonico (il percorso storico resta intatto). Senza
    brano Spotify delega al tracker armonico, che decide in base al chroma e al
    knob `harmony_gain`.

    Args:
        spotify_palette: Palette del brano Spotify corrente, o None se assente.
        harmonic_tracker: Macchina a stati della palette armonica.
        harmony_hue: Hue tonale del frame in [0, 1), o None.
        harmony_gain: Knob armonia (0 = palette armonica disattivata).
        now: Istante monotonic del frame (per l'isteresi del tracker).

    Returns:
        Coppia `(target, nota)`: la palette da passare all'animator (o None per la
        modalita' default) e il nome della nota se e' stata appena generata una
        palette armonica nuova (per il log), altrimenti None.
    """
    if spotify_palette is not None:
        return spotify_palette, None
    return harmonic_tracker.update(harmony_hue, harmony_gain, now)


# --- Frontend di controllo ---
# Frame tra due pubblicazioni dello snapshot verso il frontend (~9 Hz a 47 fps
# effettivi): il dict viene RIASSEGNATO in blocco, mai mutato dopo.
SNAPSHOT_EVERY_FRAMES = 5
# Nome di ripiego per i canali di cui CLIP v2 non risolve il device.
UNNAMED_CHANNEL_LABEL = "Canale"
# Messaggio per la UI quando nessun bridge ha un'area da sincronizzare.
NO_BRIDGE_MESSAGE = "Nessun bridge da sincronizzare: scegli un'area nella sezione BRIDGE."

# --- Coreografia del drop (knob drop_intensity) ---
# Durata del beat usata per il buio quando il BPM non e' agganciato (secondi).
DROP_FALLBACK_BEAT_SECONDS = 0.45
# Limiti del buio pre-esplosione (secondi): mai piu' corto di un lampo
# percepibile, mai cosi' lungo da sembrare un guasto.
DROP_BLACK_MIN_SECONDS = 0.25
DROP_BLACK_MAX_SECONDS = 0.6
# Durata dello "slam" a palette piena dopo il buio (secondi).
DROP_SLAM_SECONDS = 0.5
# Luminosita' residua alla fine dello slam (dissolve verso il normale).
DROP_SLAM_FLOOR = 0.3
# Attenuazione massima durante il build-up (tensione: si scende per esplodere).
BUILD_DIM_MAX = 0.2

# --- Special vocale: fill caldo sui canali con ruolo "voice" ---
# Tinta del fill (hue ambra calda) e saturazione.
VOICE_FILL_HUE = 0.09
VOICE_FILL_SAT = 0.55
# Presenza vocale minima perche' il fill si accenda (evita sfarfallio).
VOICE_FILL_MIN_LEVEL = 0.05

# --- Crossfade al cambio preset a caldo (ondata 3) ---
# Durata della dissolvenza RGB tra il preset vecchio e quello nuovo quando
# l'effetto di un worker viene sostituito a caldo (secondi). Il lerp e' sui
# frame GREZZI dei due effetti; drop/slam e tinta/gamma restano DOPO.
CROSSFADE_SECONDS = 0.4


def _crossfade_entries(
    old_entries: list[tuple[int, tuple[float, float, float]]],
    new_entries: list[tuple[int, tuple[float, float, float]]],
    fraction: float,
) -> list[tuple[int, tuple[float, float, float]]]:
    """Interpola in RGB due liste di entries per channel_id.

    fraction 0 = solo il vecchio, 1 = solo il nuovo. I canali presenti in una
    sola delle due liste passano col loro valore (nessun lerp): copre il caso
    limite in cui i due effetti espongano canali diversi. L'ordine segue le
    entries nuove, poi accoda i canali solo-vecchi.

    Args:
        old_entries: Frame grezzo del preset uscente (channel_id, rgb).
        new_entries: Frame grezzo del preset entrante (channel_id, rgb).
        fraction: Posizione della dissolvenza in [0, 1].

    Returns:
        La lista interpolata (channel_id, rgb), lista nuova.
    """
    old_map = dict(old_entries)
    result: list[tuple[int, tuple[float, float, float]]] = []
    seen: set[int] = set()
    for channel_id, new_rgb in new_entries:
        seen.add(channel_id)
        old_rgb = old_map.get(channel_id)
        if old_rgb is None:
            result.append((channel_id, new_rgb))
        else:
            result.append(
                (
                    channel_id,
                    (
                        old_rgb[0] + (new_rgb[0] - old_rgb[0]) * fraction,
                        old_rgb[1] + (new_rgb[1] - old_rgb[1]) * fraction,
                        old_rgb[2] + (new_rgb[2] - old_rgb[2]) * fraction,
                    ),
                )
            )
    for channel_id, old_rgb in old_entries:
        if channel_id not in seen:
            result.append((channel_id, old_rgb))
    return result


# --- Guardia per-frame del loop (A3) ---
# Errori CONSECUTIVI in un frame dopo i quali il breaker ferma il motore: un
# glitch isolato del device audio, un NaN o un caso limite di un effetto viene
# loggato e il frame saltato, ma un guasto persistente non deve girare a vuoto
# all'infinito. Il contatore si azzera a ogni frame andato a buon fine.
MAX_CONSECUTIVE_FRAME_ERRORS = 50
# Messaggio del breaker (log + UI) quando gli errori consecutivi superano la soglia.
ENGINE_BREAKER_MESSAGE = (
    "Motore fermato: troppi errori consecutivi nel loop di rendering (vedi il log per i dettagli)."
)


def _show_frame(
    entries: list[tuple[int, tuple[float, float, float]]],
    palette: list | None,
    voice_level: float,
    voice_channels: set[int],
    positions: dict[int, float],
    build_dim: float,
    now: float,
    black_until: float,
    slam_until: float,
) -> list[tuple[int, tuple[float, float, float]]]:
    """Post-pass coreografico sul frame di UN worker (vale per ogni preset).

    Ordine di priorita': buio del drop > slam a palette piena > (attenuazione
    di build + fill vocale). Il fill vocale SOSTITUISCE il colore sui canali
    voice (non un MAX per componente, che mescolando con l'effetto darebbe
    tinte sporche e incoerenti tra preset): sopra la presenza vocale minima il
    canale voice mostra il fill ambra pesato da voice_level, sotto segue
    l'effetto normale (attenuato dal build come gli altri).

    Args:
        entries: Frame prodotto dall'effetto, lista (channel_id, rgb).
        palette: Palette corrente (per lo slam), o None.
        voice_level: Presenza vocale stimata del blocco [0, 1].
        voice_channels: Id dei canali con ruolo "voice" su questo bridge.
        positions: Posizione spaziale reale [0, 1] per channel_id (dall'effetto);
            usata dallo slam per la continuita' cromatica con chase/waterfall,
            con fallback all'indice per i canali senza posizione nota.
        build_dim: Attenuazione di build gia' calcolata [0, 1] (0 = nessuna).
        now: time.monotonic() del frame.
        black_until: Fine della finestra di buio del drop (monotonic).
        slam_until: Fine della finestra di slam (monotonic).

    Returns:
        Il frame trasformato (lista nuova, l'input non viene mutato).
    """
    if now < black_until:
        return [(channel_id, (0.0, 0.0, 0.0)) for channel_id, _ in entries]

    if now < slam_until:
        # Slam: ogni canale al colore pieno della palette (dissolve in coda).
        strength = DROP_SLAM_FLOOR + (1.0 - DROP_SLAM_FLOOR) * (
            (slam_until - now) / DROP_SLAM_SECONDS
        )
        strength = min(max(strength, 0.0), 1.0)
        count = len(entries)
        slammed: list[tuple[int, tuple[float, float, float]]] = []
        for index, (channel_id, _) in enumerate(entries):
            # Posizione reale del canale dall'effetto (chase/waterfall seguono un
            # percorso a U non monotono in x): continuita' col frame di flusso.
            fallback = index / (count - 1) if count > 1 else 0.5
            position = positions.get(channel_id, fallback)
            hue, sat, _scale = palette_channel_base(palette, position)
            slammed.append((channel_id, colorsys.hsv_to_rgb(hue, sat, strength)))
        return slammed

    out: list[tuple[int, tuple[float, float, float]]] = []
    dim = 1.0 - build_dim
    voice_rgb: tuple[float, float, float] | None = None
    if voice_channels and voice_level >= VOICE_FILL_MIN_LEVEL:
        voice_rgb = colorsys.hsv_to_rgb(VOICE_FILL_HUE, VOICE_FILL_SAT, voice_level)
    for channel_id, (red, green, blue) in entries:
        if voice_rgb is not None and channel_id in voice_channels:
            # Un solo proprietario del colore sui canali voice: sostituzione,
            # non blend, cosi' il risultato e' identico in tutti i preset.
            out.append((channel_id, voice_rgb))
        else:
            out.append((channel_id, (red * dim, green * dim, blue * dim)))
    return out


def _apply_tint(
    entries: list[tuple[int, tuple[float, float, float]]],
    hue_offset: float,
    gamma: float,
) -> list[tuple[int, tuple[float, float, float]]]:
    """Applica rotazione manuale dell'hue e curva gamma a un frame (post-pass).

    Chiamata solo quando almeno un knob e' fuori dal default (hue_offset != 0 o
    gamma != 1.0): coi default il percorso e' saltato del tutto dal chiamante,
    quindi qui i controlli per-entry servono solo a evitare lavoro inutile
    quando un solo knob e' attivo.

    Args:
        entries: Frame (channel_id, rgb) gia' coreografato.
        hue_offset: Rotazione dell'hue sulla ruota in [0, 1] (0 = nessuna).
        gamma: Esponente percettivo per componente (1.0 = lineare, identico).

    Returns:
        Il frame con hue ruotato e gamma applicata (lista nuova).
    """
    rotate = hue_offset != 0.0
    curve = gamma != 1.0
    out: list[tuple[int, tuple[float, float, float]]] = []
    for channel_id, (red, green, blue) in entries:
        if rotate:
            hue, sat, value = colorsys.rgb_to_hsv(red, green, blue)
            hue = (hue + hue_offset) % 1.0
            red, green, blue = colorsys.hsv_to_rgb(hue, sat, value)
        if curve:
            red, green, blue = red**gamma, green**gamma, blue**gamma
        out.append((channel_id, (red, green, blue)))
    return out


# --- Lock di istanza singola del motore ---
# Attesa massima perche' il lock si liberi prima di dichiarare occupata l'altra
# istanza: un riavvio ravvicinato (power OFF seguito subito da ON) parte mentre
# il motore precedente sta ancora chiudendo sessione DTLS ed entertainment, e il
# suo lock cade solo alla fine di quella chiusura.
LOCK_RETRY_SECONDS = 15.0
# Passo tra due tentativi di acquisizione durante l'attesa.
LOCK_RETRY_STEP_SECONDS = 0.5


def _acquire_engine_lock() -> TextIO:
    """Acquisisce il lock di istanza singola del motore (non bloccante).

    Apre (creando se serve) il lockfile in `paths.data_dir()` e ci prende un
    byte-range lock esclusivo Windows via `msvcrt.locking`: se un'altra istanza
    lo tiene gia', alza un errore chiaro SENZA toccare il bridge. Il lock si
    rilascia alla chiusura del file handle (finally di run_engine) o alla morte
    del processo (Windows libera i lock dei file alla chiusura): non resta mai
    orfano dopo un crash.

    Returns:
        Il file handle aperto che TIENE il lock (da passare a
        `_release_engine_lock`).

    Raises:
        RuntimeError: Se un'altra istanza sta gia' pilotando le luci.
    """
    import msvcrt

    lock_path = paths.data_dir() / paths.ENGINE_LOCK_FILENAME
    handle = open(lock_path, "a+")
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as err:
        handle.close()
        raise RuntimeError("un'altra istanza di huebeat sta gia' pilotando le luci") from err
    return handle


def _acquire_engine_lock_retrying(stop_event: threading.Event) -> TextIO | None:
    """Acquisisce il lock di istanza ritentando per LOCK_RETRY_SECONDS.

    Il lock del motore precedente puo' essere ancora appeso per qualche secondo
    dopo un power OFF (chiusura DTLS + stop entertainment via REST): fallire al
    primo colpo farebbe morire il riavvio accodato dall'app. Ogni attesa passa
    da `stop_event.wait`, quindi uno stop durante l'attesa esce subito e senza
    errore (il motore non doveva partire).

    Args:
        stop_event: Event che chiede l'uscita ordinata dal motore.

    Returns:
        Il file handle che TIENE il lock, o None se lo stop e' arrivato durante
        l'attesa (niente motore da avviare).

    Raises:
        RuntimeError: Se il lock resta occupato oltre LOCK_RETRY_SECONDS.
    """
    deadline = time.monotonic() + LOCK_RETRY_SECONDS
    while True:
        try:
            return _acquire_engine_lock()
        except RuntimeError:
            if stop_event.is_set() or time.monotonic() >= deadline:
                raise
            logger.debug("Lock del motore occupato: ritento tra %.1f s.", LOCK_RETRY_STEP_SECONDS)
        if stop_event.wait(LOCK_RETRY_STEP_SECONDS):
            return None


def _release_engine_lock(handle: TextIO) -> None:
    """Rilascia il lock del motore e chiude il file handle (mai solleva).

    Args:
        handle: Il file handle restituito da `_acquire_engine_lock`.
    """
    import msvcrt

    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    finally:
        handle.close()


@contextlib.contextmanager
def _sigint_ignored() -> Iterator[None]:
    """Ignora temporaneamente SIGINT per proteggere il cleanup da un 2o Ctrl+C.

    Durante la chiusura ordinata (close DTLS + stop_entertainment) un secondo
    Ctrl+C troncherebbe il cleanup lasciando l'area "active" sul bridge. Qui
    SIGINT viene ignorato per la durata del blocco e ripristinato dopo.
    `signal.signal` funziona solo dal main thread: nell'app il motore gira in un
    thread di lavoro (e SIGINT arriva comunque al main thread), quindi fuori dal
    main thread il context manager e' un no-op sicuro.

    Yields:
        None: esegue il blocco protetto.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def cmd_run(_args: argparse.Namespace) -> int:
    """Sincronizza le luci con lo spettro dell'audio di sistema fino a Ctrl+C.

    Se Spotify e' configurato e autenticato, ogni brano riceve una palette
    generata dal suo id e ogni cambio di traccia viene annunciato. Con
    control_port > 0 avvia anche il server del frontend di controllo
    (effetto, device, ruoli, tuning e stato live regolabili a caldo).

    Args:
        _args: Argomenti del sottocomando (non usati).

    Returns:
        Exit code del comando.
    """
    from huebeat import tuning
    from huebeat.control import ControlBus, ControlServer
    from huebeat.onboarding import PairingManager, SpotifyLinkManager

    _require_paired()

    # Travasa i knob persistiti (sezione [tuning]) nell'istanza runtime.
    tuning.live.apply(settings.tuning.model_dump())

    bus = ControlBus(settings.effect)
    server: ControlServer | None = None
    if settings.control_port:
        from huebeat.audio import list_loopback_devices
        from huebeat.process_audio import list_audio_apps

        bus.list_devices = list_loopback_devices
        bus.list_audio_apps = list_audio_apps
        # Manager di onboarding anche in run: il pannello Spotify resta usabile
        # e il wizard di pairing e' disponibile. Niente on_paired: in run il
        # processo va riavviato dopo il pairing (a differenza dell'app desktop).
        server = ControlServer(
            bus,
            settings.control_port,
            persisted=_initial_overrides(),
            pairing=PairingManager(),
            spotify_link=SpotifyLinkManager(),
        )
        try:
            server.start()
            print(f"Frontend di controllo: http://127.0.0.1:{settings.control_port}/")
        except OSError:
            # Porta occupata da un'altra istanza di huebeat: il motore va
            # avanti lo stesso, solo senza pannello proprio.
            logger.warning(
                "Porta %d occupata (altra istanza?): proseguo senza frontend.",
                settings.control_port,
            )
            server = None

    stop_event = threading.Event()
    exit_code = EXIT_OK
    try:
        run_engine(bus, stop_event)
    except KeyboardInterrupt:
        print("\nInterrotto.")
    except RuntimeError as err:
        # Lock di istanza singola occupato: messaggio chiaro, nessun bridge
        # toccato. Il motore non e' partito.
        print(f"Errore: {err}")
        exit_code = EXIT_ERROR
    finally:
        # Protegge la chiusura da un secondo Ctrl+C (vedi _sigint_ignored).
        with _sigint_ignored():
            stop_event.set()
            if server is not None:
                server.stop()

    return exit_code


def _initial_overrides() -> dict:
    """Override di partenza per il server di controllo (stato persistito).

    Returns:
        Dict con effetto, device, hotkey, tuning e ruoli correnti: il primo
        salvataggio di config.local.toml riparte da qui senza perdere nulla.
    """
    from huebeat import tuning

    overrides = {
        "effect": settings.effect,
        "audio_device": settings.audio_device,
        "toggle_hotkey": settings.toggle_hotkey,
        "invert_x": settings.invert_x,
        "tuning": tuning.live.as_dict(),
        "channel_roles": {str(k): v for k, v in settings.channel_roles.items()},
        "bridge_sync": dict(settings.bridge_sync),
    }
    # Ruoli per-bridge: una sezione TOML per bridge (chiave puntata).
    for bridge_id, bridge_roles in settings.channel_roles_by_bridge.items():
        overrides[f"channel_roles_by_bridge.{bridge_id}"] = {
            str(k): v for k, v in bridge_roles.items()
        }
    return overrides


def _build_channel_infos(channels: list[dict], roles: dict[int, str]) -> list:
    """Costruisce i ChannelInfo dell'area applicando invert_x e i ruoli dati.

    Args:
        channels: Dict grezzi dei canali della sessione.
        roles: Mappa channel_id -> ruolo forzato (i mancanti restano spaziali).

    Returns:
        Lista di ChannelInfo pronta per create_effect.
    """
    from huebeat.effects import ChannelInfo

    # invert_x specchia l'asse spaziale quando le posizioni configurate
    # nell'app Hue non corrispondono al montaggio fisico reale.
    x_sign = -1.0 if settings.invert_x else 1.0
    return [
        ChannelInfo(
            channel_id=channel["channel_id"],
            x=x_sign * channel.get("position", {}).get("x", 0.0),
            y=channel.get("position", {}).get("y", 0.0),
            z=channel.get("position", {}).get("z", 0.0),
            role=roles.get(channel["channel_id"]),
        )
        for channel in channels
    ]


def _lights_payload(
    channels: list[dict], names: dict[int, str], roles: dict[int, str]
) -> list[dict]:
    """Compone l'elenco luci per lo snapshot del frontend.

    Args:
        channels: Dict grezzi dei canali della sessione.
        names: Mappa channel_id -> nome del device (via CLIP v2, best-effort).
        roles: Mappa channel_id -> ruolo forzato corrente.

    Returns:
        Lista di dict con channel_id, name e role ("" = spaziale).
    """
    return [
        {
            "channel_id": channel["channel_id"],
            "name": names.get(channel["channel_id"])
            or f"{UNNAMED_CHANNEL_LABEL} {channel['channel_id']}",
            "role": roles.get(channel["channel_id"], ""),
        }
        for channel in channels
    ]


def run_engine(bus: "object", stop_event: threading.Event) -> None:
    """Avvia il motore proteggendolo col lock di istanza singola.

    Prende il lock in `paths.data_dir()` PRIMA di toccare qualsiasi bridge: se
    un'altra istanza sta gia' pilotando le luci alza un errore chiaro e non fa
    partire nulla (niente ping-pong sullo slot DTLS). L'acquisizione ritenta per
    LOCK_RETRY_SECONDS: un riavvio ravvicinato puo' trovare il lock del motore
    precedente non ancora rilasciato. Il lock si rilascia sempre nel finally (e
    comunque alla chiusura del processo): mai orfano. Vale sia per `run` sia per
    l'app desktop, che riusano questa stessa funzione.

    Args:
        bus: Il ControlBus condiviso col server di controllo.
        stop_event: Event che chiede l'uscita ordinata dal loop.

    Raises:
        RuntimeError: Se un'altra istanza di huebeat sta gia' pilotando le luci.
    """
    lock_handle = _acquire_engine_lock_retrying(stop_event)
    if lock_handle is None:
        # Stop arrivato mentre si aspettava il lock: niente da avviare.
        return
    try:
        _run_engine_locked(bus, stop_event)
    finally:
        _release_engine_lock(lock_handle)


def _run_engine_locked(bus: "object", stop_event: threading.Event) -> None:
    """Loop realtime di sync: audio -> analisi -> effetto -> Bridge.

    Estratto da cmd_run per essere riusabile dall'app desktop (che lo esegue
    in un thread di lavoro). Il ritmo del loop e' dato dalla lettura audio
    bloccante. Tutta l'interazione col frontend passa dal bus con letture e
    riassegnazioni atomiche: il loop non prende MAI lock ne' fa I/O di rete
    (lo snapshot e' un dict riassegnato; il server lo serializza per conto suo).
    Chiamata solo da `run_engine`, che tiene il lock di istanza singola.

    Args:
        bus: Il ControlBus condiviso col server di controllo.
        stop_event: Event che chiede l'uscita ordinata dal loop.
    """
    from huebeat import tuning
    from huebeat.analysis import HOP_DIVISOR, FeatureAnalyzer
    from huebeat.beat import BeatTracker
    from huebeat.effects import create_effect
    from huebeat.multibridge import WorkerPool, resolve_plans
    from huebeat.palette import PaletteAnimator
    from huebeat.source import AudioSource
    from huebeat.stereo import StereoAnalyzer
    from huebeat.structure import StructureTracker

    # Multi-bridge: un worker per bridge, ognuno col suo setup con retry in un
    # thread proprio (i rifiuti DTLS transitori di un bridge non bloccano gli
    # altri) e il suo SessionKeeper. Il loop parte SUBITO: i worker diventano
    # pronti quando la loro sessione si stabilisce; con zero piani il motore
    # gira comunque (livelli audio vivi nella UI) e la UI spiega il perche'.
    pool = WorkerPool()
    bridges_version, bridges_map = bus.bridges_request
    plans = resolve_plans(bridges_map if bridges_version else None)
    pool.apply_plans(plans)
    if plans:
        bus.last_error = None
        print(f"Sync in avvio su {len(plans)} bridge (effetto '{settings.effect}').")
    else:
        bus.last_error = NO_BRIDGE_MESSAGE

    # Ruoli canale: mappa per-bridge con fallback sul flat legacy
    # per i bridge senza mappa propria (compat a bridge singolo).
    legacy_roles: dict[int, str] = dict(settings.channel_roles)
    roles_by_bridge: dict[str, dict[int, str]] = {
        bridge_id: dict(bridge_roles)
        for bridge_id, bridge_roles in settings.channel_roles_by_bridge.items()
    }
    roles_version = bus.roles_request[0]

    def _roles_for(bridge_id: str) -> dict[int, str]:
        per_bridge = roles_by_bridge.get(bridge_id)
        return per_bridge if per_bridge is not None else legacy_roles

    current_effect_name: str = bus.effect_request

    # Sospensione luci: Event condiviso tra loop, hotkey e frontend.
    lights_enabled = bus.lights_enabled
    hotkey_holder: dict = {"hotkey": None, "combo": settings.toggle_hotkey}

    def _on_toggle(active: bool) -> None:
        if active:
            lights_enabled.set()
        else:
            lights_enabled.clear()
        print("huebeat: luci riattivate." if active else "huebeat: luci sospese (hotkey).")

    def _start_hotkey(combo: str):
        from huebeat.hotkey import HotkeyToggle

        candidate = HotkeyToggle(combo, _on_toggle)
        return candidate if candidate.start() else None

    if settings.toggle_hotkey:
        hotkey_holder["hotkey"] = _start_hotkey(settings.toggle_hotkey)
        if hotkey_holder["hotkey"] is not None:
            print(f"Hotkey attiva: {settings.toggle_hotkey} sospende/riattiva le luci.")

    def _change_hotkey(combo: str) -> str | None:
        """Sostituisce l'hotkey a caldo (eseguito nel thread del server)."""
        old = hotkey_holder["hotkey"]
        if old is not None:
            old.stop()
        hotkey_holder["hotkey"] = None
        hotkey_holder["combo"] = combo
        if combo:
            candidate = _start_hotkey(combo)
            if candidate is None:
                return f"registrazione hotkey '{combo}' fallita"
            hotkey_holder["hotkey"] = candidate
        return None

    def _sync_hotkey_state(enabled: bool) -> None:
        """Riallinea lo stato interno dell'hotkey a un toggle fatto dalla UI."""
        if hotkey_holder["hotkey"] is not None:
            hotkey_holder["hotkey"].set_active(enabled)

    bus.on_hotkey_change = _change_hotkey
    bus.on_lights_sync = _sync_hotkey_state

    poller = _maybe_start_spotify_poller()
    # Dinamica della palette (crossfade, drift, energia). Alimentata dal brano
    # Spotify quando c'e'; in sua assenza dalla palette armonica del chroma
    # (harmonic_tracker), se harmony_gain > 0. Con harmony_gain 0 e senza Spotify
    # resta la modalita' default (nessuna palette), identica a prima.
    animator = PaletteAnimator()
    # Palette armonica guidata dalla tonalita' (chroma) quando manca Spotify.
    harmonic_tracker = HarmonicPaletteTracker()
    # Ultimo track_id annunciato: serve a stampare "In riproduzione" solo al
    # cambio di traccia, non a ogni frame.
    last_track_id: str | None = None
    # Ultima palette animata (per ri-applicarla a un effetto appena ricreato).
    current_palette: list | None = None

    frame_count = 0
    start_monotonic = time.monotonic()
    bus.engine_running = True

    initial_target = (
        settings.audio_process if settings.audio_source == "process" else settings.audio_device
    )
    # Hop tra finestre sovrapposte (B2): la cattura avanza di questo passo, quindi
    # il block rate del loop e' sample_rate/hop (non /block_size). Stessa formula
    # della cattura (block_size // HOP_DIVISOR): i due valori coincidono sempre.
    hop = max(1, settings.audio_block_size // HOP_DIVISOR)
    try:
        with AudioSource(
            settings.audio_source, initial_target, settings.audio_block_size
        ) as capture:
            analyzer = FeatureAnalyzer(capture.sample_rate, settings.audio_block_size, hop)
            tracker = BeatTracker(capture.sample_rate / hop)
            stereo_analyzer = StereoAnalyzer(capture.sample_rate, settings.audio_block_size)
            structure = StructureTracker(capture.sample_rate / hop)
            # Finestre della coreografia del drop (monotonic; 0 = inattive).
            drop_black_until = 0.0
            drop_slam_until = 0.0
            # Crossfade al cambio preset (ondata 3): bridge_id -> (effetto
            # uscente, fine finestra monotonic). Popolato al cambio effetto,
            # svuotato per bridge quando la finestra si chiude.
            crossfade_by_bridge: dict[str, tuple[object, float]] = {}
            # Canali col ruolo "voice" per bridge (C16): ricalcolati solo quando
            # l'effetto viene ricreato (cambio effetto/ruoli), non a ogni frame.
            voice_channels_by_bridge: dict[str, set[int]] = {}
            # Guardia per-frame (A3): errori consecutivi per il breaker.
            consecutive_errors = 0
            logger.info(
                "Cattura audio a %d Hz, blocco %d campioni.",
                capture.sample_rate,
                settings.audio_block_size,
            )
            # Il ritmo del loop e' dato dalla lettura audio bloccante: niente sleep.
            while not stop_event.is_set():
                # A3: una eccezione in un frame (glitch del device, NaN, caso
                # limite di un effetto) viene loggata e il frame saltato, cosi'
                # non uccide il motore; un guasto persistente ferma il motore col
                # breaker dopo MAX_CONSECUTIVE_FRAME_ERRORS errori consecutivi.
                try:
                    block = capture.read_block()
                    # La cattura puo' seguire un cambio dell'output di default: se
                    # il device nuovo ha un sample rate diverso, analizzatore e
                    # beat tracker vanno ricreati (dipendono dal rate).
                    if capture.sample_rate != analyzer.sample_rate:
                        logger.info(
                            "Cattura passata a %d Hz: ricreo analizzatore e beat tracker.",
                            capture.sample_rate,
                        )
                        analyzer = FeatureAnalyzer(
                            capture.sample_rate, settings.audio_block_size, hop
                        )
                        tracker = BeatTracker(capture.sample_rate / hop)
                        stereo_analyzer = StereoAnalyzer(
                            capture.sample_rate, settings.audio_block_size
                        )
                        structure = StructureTracker(capture.sample_rate / hop)

                    # --- Comandi dal frontend (letture atomiche, mai lock) ---
                    if bus.effect_request != current_effect_name:
                        current_effect_name = bus.effect_request
                        crossfade_end = time.monotonic() + CROSSFADE_SECONDS
                        for worker in pool.workers.values():
                            # Crossfade dal preset uscente: conserva l'istanza
                            # vecchia (il worker la stava gia' rendendo) finche'
                            # la finestra resta aperta. Solo sul cambio PRESET,
                            # non sul cambio ruoli (che ricrea con taglio netto).
                            if worker.effect is not None and CROSSFADE_SECONDS > 0.0:
                                crossfade_by_bridge[worker.plan.creds.id] = (
                                    worker.effect,
                                    crossfade_end,
                                )
                            worker.effect = None  # ricreato sotto coi canali della sua area
                        logger.info("Effetto cambiato a caldo: %s.", current_effect_name)
                    device_request = bus.device_request
                    if device_request is not None:
                        bus.device_request = None
                        capture.request_target(device_request)
                    source_request = bus.source_request
                    if source_request is not None:
                        bus.source_request = None
                        capture.swap(source_request[0], source_request[1])
                    requested_version, requested_roles = bus.roles_request
                    if requested_version != roles_version:
                        roles_version = requested_version
                        roles_by_bridge = {**roles_by_bridge, **requested_roles}
                        for worker in pool.workers.values():
                            worker.effect = None
                        logger.info("Ruoli canale aggiornati a caldo: %s.", requested_roles)
                    requested_bridges_version, requested_bridges_map = bus.bridges_request
                    if requested_bridges_version != bridges_version:
                        bridges_version = requested_bridges_version
                        plans = resolve_plans(requested_bridges_map)
                        pool.apply_plans(plans)
                        bus.last_error = None if plans else NO_BRIDGE_MESSAGE
                        logger.info("Sync bridge aggiornato a caldo: %d attivi.", len(plans))

                    features = analyzer.process(block)
                    beat = tracker.update(features.flux)
                    stereo_image = stereo_analyzer.process(capture.last_stereo)
                    show_state = structure.update(features, beat)
                    # Freeze dell'AGC nel breakdown: al drop le bande rientrano
                    # con l'headroom della sezione piena (vedi analysis).
                    analyzer.set_agc_hold(show_state.phase == "breakdown")
                    show_gain = tuning.live.drop_intensity
                    if show_state.drop_fired and show_gain > 0.0:
                        beat_seconds = (
                            60.0 / beat.bpm
                            if beat.locked and beat.bpm > 0.0
                            else DROP_FALLBACK_BEAT_SECONDS
                        )
                        black_seconds = min(
                            max(beat_seconds, DROP_BLACK_MIN_SECONDS), DROP_BLACK_MAX_SECONDS
                        )
                        now_show = time.monotonic()
                        drop_black_until = now_show + black_seconds
                        drop_slam_until = drop_black_until + DROP_SLAM_SECONDS
                        logger.info("Drop: buio %.2f s + slam a palette piena.", black_seconds)
                    now_frame = time.monotonic()
                    # Palette: il brano Spotify vince sempre (percorso storico
                    # intatto); in sua assenza, con harmony_gain > 0, una palette
                    # armonica guidata dal chroma. harmony_hue/strength dal
                    # contratto AudioFeatures (getattr con default: assenti =
                    # nessun bend ne' palette armonica).
                    harmony_hue = getattr(features, "harmony_hue", None)
                    harmony_strength = getattr(features, "harmony_strength", 0.0)
                    spotify_palette = poller.palette if poller is not None else None  # type: ignore[attr-defined]
                    target_palette, harmonic_note = _select_palette_target(
                        spotify_palette,
                        harmonic_tracker,
                        harmony_hue,
                        tuning.live.harmony_gain,
                        now_frame,
                    )
                    if harmonic_note is not None:
                        logger.info("Palette armonica dal chroma: tonalita' %s.", harmonic_note)
                    animator.set_target(target_palette)
                    current_palette = animator.update(
                        features.onset_strength,
                        features.section_energy,
                        warmth=features.warmth,
                        beat=beat,
                        harmony_hue=harmony_hue,
                        harmony_strength=harmony_strength,
                    )
                    # B5: al cambio brano azzera structure e beat tracker
                    # (rollmax/refrattario e BPM del brano vecchio non valgono
                    # piu'). Solo su un cambio vero: non al PRIMO brano, dove i
                    # tracker sono gia' freschi e potrebbero aver appena lockato.
                    # Annuncio e reset solo col poller Spotify attivo.
                    if poller is not None:
                        new_track_id = _announce_track(poller, last_track_id)
                        if last_track_id is not None and new_track_id != last_track_id:
                            structure.reset()
                            tracker.reset()
                            logger.info("Cambio brano: reset di structure e beat tracker.")
                        last_track_id = new_track_id
                    lights_on = lights_enabled.is_set()
                    build_dim = (
                        BUILD_DIM_MAX * show_state.build_intensity * show_gain
                        if show_state.phase == "build"
                        else 0.0
                    )
                    # B9 + Tinta: rotazione hue e gamma lette UNA volta per frame.
                    # Coi default (0.0 e 1.0) il post-pass e' saltato del tutto.
                    hue_offset = tuning.live.hue_offset
                    gamma = tuning.live.gamma
                    apply_tint = hue_offset != 0.0 or gamma != 1.0
                    for worker in pool.workers.values():
                        if not worker.ready:
                            continue
                        bridge_id = worker.plan.creds.id
                        worker_roles = _roles_for(bridge_id)
                        if worker.effect is None:
                            infos = _build_channel_infos(worker.session.channels, worker_roles)
                            worker.effect = create_effect(current_effect_name, infos)
                            # C16: canali voce cacheati alla (ri)creazione dell'effetto.
                            voice_channels_by_bridge[bridge_id] = {
                                channel_id
                                for channel_id, role in worker_roles.items()
                                if role == "voice"
                            }
                        if current_palette is not None:
                            worker.effect.set_palette(current_palette)
                        set_stereo = getattr(worker.effect, "set_stereo", None)
                        if set_stereo is not None:
                            set_stereo(stereo_image)
                        if lights_on:
                            entries = worker.effect.render(features, beat)
                            # Crossfade dal preset uscente (ondata 3): finche' la
                            # finestra e' aperta renderizza ANCHE il vecchio e
                            # lerpa RGB per channel_id sui frame grezzi. Sta PRIMA
                            # di show_frame e tinta/gamma (che restano dopo), cosi'
                            # drop/slam e post-pass non ne sono toccati.
                            cross = crossfade_by_bridge.get(bridge_id)
                            if cross is not None:
                                old_effect, cross_end = cross
                                if now_frame >= cross_end:
                                    crossfade_by_bridge.pop(bridge_id, None)
                                else:
                                    if current_palette is not None:
                                        old_effect.set_palette(current_palette)  # type: ignore[attr-defined]
                                    old_set_stereo = getattr(old_effect, "set_stereo", None)
                                    if old_set_stereo is not None:
                                        old_set_stereo(stereo_image)
                                    old_entries = old_effect.render(features, beat)  # type: ignore[attr-defined]
                                    fraction = 1.0 - (cross_end - now_frame) / CROSSFADE_SECONDS
                                    entries = _crossfade_entries(old_entries, entries, fraction)
                            # A6: posizioni spaziali reali dei canali (fallback
                            # all'indice) per la continuita' cromatica dello slam.
                            positions = getattr(worker.effect, "positions", {})
                            entries = _show_frame(
                                entries,
                                current_palette,
                                features.voice_level,
                                voice_channels_by_bridge.get(bridge_id, set()),
                                positions,
                                build_dim,
                                now_frame,
                                drop_black_until,
                                drop_slam_until,
                            )
                            if apply_tint:
                                entries = _apply_tint(entries, hue_offset, gamma)
                        else:
                            entries = worker.black_entries
                        worker.send(entries)
                except Exception:  # noqa: BLE001 - una eccezione non deve uccidere il motore
                    consecutive_errors += 1
                    logger.warning(
                        "Frame saltato per un errore (%d consecutivi).",
                        consecutive_errors,
                        exc_info=True,
                    )
                    if consecutive_errors >= MAX_CONSECUTIVE_FRAME_ERRORS:
                        bus.last_error = ENGINE_BREAKER_MESSAGE
                        logger.error(ENGINE_BREAKER_MESSAGE)
                        break
                    continue
                consecutive_errors = 0

                frame_count += 1
                if frame_count % SNAPSHOT_EVERY_FRAMES == 0:
                    now_playing = poller.now_playing if poller is not None else None  # type: ignore[attr-defined]
                    workers = list(pool.workers.values())
                    ready_workers = [w for w in workers if w.ready]
                    lights_payload = []
                    for worker in ready_workers:
                        for item in _lights_payload(
                            worker.session.channels,
                            worker.light_names,
                            _roles_for(worker.plan.creds.id),
                        ):
                            item["bridge"] = worker.display_name
                            item["bridge_id"] = worker.plan.creds.id
                            lights_payload.append(item)
                    first_effect = ready_workers[0].effect if ready_workers else None
                    bus.snapshot = {
                        "area": " + ".join(w.session.area_name for w in ready_workers),
                        "channel_count": sum(len(w.session.channels) for w in ready_workers),
                        "fps": settings.stream_fps,
                        "uptime_seconds": int(time.monotonic() - start_monotonic),
                        "effect": current_effect_name,
                        "adaptive_mode": getattr(first_effect, "mode", None),
                        "levels": {
                            "bass": features.bass,
                            "mid": features.mid,
                            "high": features.high,
                            "voice": features.voice_level,
                        },
                        "onset": features.onset_strength,
                        "bpm": beat.bpm,
                        "bpm_locked": beat.locked,
                        "beat_confidence": beat.confidence,
                        "section_energy": features.section_energy,
                        "warmth": features.warmth,
                        "palette": current_palette,
                        "track": (
                            {
                                "title": now_playing.title,
                                "artists": now_playing.artists,
                                "is_playing": now_playing.is_playing,
                                "volume": now_playing.volume_percent,
                            }
                            if now_playing is not None
                            else None
                        ),
                        "device": capture.device_display_name,
                        "audio_source": capture.source_type,
                        "source_target": capture.target,
                        "show_phase": show_state.phase,
                        "reconnecting": any(w.reconnecting for w in ready_workers),
                        # Tentativo corrente della riconnessione piu' avanti
                        # tra i worker: la UI lo mostra invece di restare muta.
                        "reconnect_attempt": max(
                            (w.reconnect_attempts for w in ready_workers if w.reconnecting),
                            default=0,
                        ),
                        "lights": lights_payload,
                        "bridges": [
                            {
                                "id": w.plan.creds.id,
                                "name": w.display_name,
                                "area": w.session.area_name if w.ready else "",
                                "channel_count": len(w.session.channels) if w.ready else 0,
                                "active": w.ready,
                                "reconnecting": w.reconnecting,
                                "error": w.last_error or "",
                            }
                            for w in workers
                        ],
                        "hotkey": hotkey_holder["combo"],
                    }
    finally:
        # C7: la chiusura ordinata (stop worker -> close DTLS + stop_entertainment)
        # va protetta da un secondo Ctrl+C, altrimenti l'area resta "active" sul
        # bridge. SIGINT ignorato per la durata del cleanup, poi ripristinato.
        with _sigint_ignored():
            bus.engine_running = False
            bus.on_hotkey_change = None
            bus.on_lights_sync = None
            if hotkey_holder["hotkey"] is not None:
                hotkey_holder["hotkey"].stop()
            if poller is not None:
                poller.stop()  # type: ignore[attr-defined]
            pool.stop_all()


def _announce_track(poller: object, last_track_id: str | None) -> str | None:
    """Stampa il brano corrente se e' cambiato rispetto all'ultimo annunciato.

    Args:
        poller: Il `NowPlayingPoller` attivo (tipizzato come object perche'
            importato localmente).
        last_track_id: track_id dell'ultimo brano annunciato, o None.

    Returns:
        Il track_id corrente da usare come nuovo riferimento (invariato se non
        c'e' nulla in riproduzione o la traccia non e' cambiata).
    """
    now_playing = poller.now_playing  # type: ignore[attr-defined]
    if now_playing is None or now_playing.track_id == last_track_id:
        return last_track_id
    # now_playing.artists e' gia' la stringa unita da ", " (il join per
    # carattere di una stringa produceva "A, d, e, l, e").
    line = f"In riproduzione: {now_playing.title} - {now_playing.artists}"
    # Doppia rete oltre al reconfigure di main(): nell'app/exe lo stdout visto
    # a runtime puo' non essere quello riconfigurato all'avvio (visto sul
    # campo: motore morto su titolo CJK NONOSTANTE il fix globale).
    # L'annuncio e' cosmetico: non deve mai poter uccidere il motore.
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"))
    return now_playing.track_id


def _cmd_app_lazy(args: argparse.Namespace) -> int:
    """Dispatch pigro del sottocomando app (pywebview importato solo qui).

    Args:
        args: Argomenti del sottocomando.

    Returns:
        Exit code del comando.
    """
    from huebeat.app import cmd_app

    return cmd_app(args)


def _build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser argparse con i sottocomandi.

    Returns:
        Il parser configurato.
    """
    parser = argparse.ArgumentParser(
        prog="huebeat",
        description="Sync musica-luci Philips Hue in tempo reale via Entertainment API.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pair_parser = subparsers.add_parser(
        "pair", help="Esegue il pairing col Bridge e salva i secrets in .env."
    )
    pair_parser.set_defaults(func=cmd_pair)

    test_parser = subparsers.add_parser(
        "test-lights", help="Sweep di colori sui canali per verificare le luci."
    )
    test_parser.set_defaults(func=cmd_test_lights)

    run_parser = subparsers.add_parser(
        "run", help="Sincronizza le luci con lo spettro dell'audio di sistema."
    )
    run_parser.set_defaults(func=cmd_run)

    app_parser = subparsers.add_parser(
        "app", help="Avvia l'app desktop di controllo (finestra nativa + tray)."
    )
    app_parser.add_argument(
        "--tray",
        action="store_true",
        help="Parte nascosta nella tray (usato dall'avvio automatico).",
    )
    app_parser.set_defaults(func=_cmd_app_lazy)

    spotify_login_parser = subparsers.add_parser(
        "spotify-login", help="Login OAuth PKCE a Spotify per la palette per-brano."
    )
    spotify_login_parser.set_defaults(func=cmd_spotify_login)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint della CLI: configura il logging, fa il dispatch, gestisce gli errori.

    Args:
        argv: Argomenti da riga di comando; None per usare `sys.argv`.

    Returns:
        Exit code del processo.
    """
    # Su Windows stdout/stderr possono essere cp1252: un titolo di brano con
    # caratteri fuori codepage NON deve uccidere il motore (successo:
    # UnicodeEncodeError dalla print di _announce_track su un titolo con
    # caratteri CJK). Con
    # errors="replace" i caratteri non rappresentabili diventano '?' a video;
    # il log su file resta utf-8 completo.
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:  # noqa: BLE001 - stream esotico: meglio senza che morti
                pass

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    # httpx logga una riga INFO per OGNI richiesta: col polling Spotify ogni
    # 5 s spamma il terminale. I suoi WARNING/ERROR restano visibili.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Da exe (--noconsole) lo stderr e' invisibile: i log vanno anche su file
    # nella home, altrimenti un errore non lascia alcuna traccia.
    if paths.is_frozen():
        file_handler = logging.FileHandler(paths.data_dir() / paths.LOG_FILENAME, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Import locali: gli errori di dominio dei moduli paralleli sono catturati
    # qui. `spotify` puo' non esistere ancora (sviluppo parallelo): se manca,
    # usiamo un tipo sentinella che non intercetta nulla.
    from huebeat.bridge import BridgeError
    from huebeat.streaming import StreamingError

    try:
        from huebeat.spotify import SpotifyError
    except ImportError:

        class SpotifyError(Exception):  # type: ignore[no-redef]
            """Sentinella: `huebeat.spotify` non e' disponibile in questo ambiente."""

    try:
        return args.func(args)
    except (BridgeError, StreamingError, SpotifyError) as err:
        logger.debug("Errore di dominio.", exc_info=True)
        print(f"Errore: {err}")
        return EXIT_ERROR
    except ValueError as err:
        logger.debug("Errore di configurazione.", exc_info=True)
        print(f"Errore: {err}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
