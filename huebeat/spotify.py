"""Integrazione Spotify per huebeat: OAuth PKCE e polling del brano corrente.

Fornisce l'identita' del brano in riproduzione sull'account Spotify dell'utente:
al cambio di traccia la palette viene GENERATA in modo deterministico dall'id
della traccia (vedi `huebeat.palette`), non estratta dalla cover. La strada
della cover e' stata abbandonata (troppe cover a dominanza bianca -> palette
slavate); l'implementazione precedente non e' inclusa.

Il modulo copre due responsabilita':

- `SpotifyAuth`: login interattivo OAuth 2.0 Authorization Code + PKCE per client
  pubblico (nessun client_secret), con cache su file di access/refresh token e
  refresh automatico alla scadenza.
- `NowPlayingPoller`: thread daemon che interroga periodicamente l'endpoint
  currently-playing, mantenendo uno snapshot thread-safe del brano corrente e
  della sua palette generata, ricalcolata solo al cambio di traccia.

Convenzione output, coerente col resto del progetto: `print` solo per le istruzioni
utente del login interattivo, `logging` per la diagnostica del poller.
"""

import base64
import hashlib
import json
import logging
import os
import secrets
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from huebeat.palette import generate_palette

logger = logging.getLogger(__name__)

# Lock di modulo che serializza le scritture del file token: piu' thread o
# istanze che rinnovano il token in parallelo non devono sovrapporre la
# scrittura (il file contiene SECRETS: access e refresh token).
_save_lock = threading.Lock()


def _atomic_write_text(path: Path, text: str) -> None:
    """Scrive `text` in `path` in modo atomico (tmp nella stessa dir + replace).

    Crea un temporaneo con nome unico (tempfile.mkstemp) nella stessa directory,
    lo scrive e lo rinomina con os.replace: un crash a meta' scrittura non lascia
    mai un file token troncato. Il temporaneo viene rimosso se qualcosa fallisce
    prima del replace.

    Args:
        path: Percorso di destinazione.
        text: Contenuto da scrivere (UTF-8).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# --- Endpoint OAuth e API Spotify ---
AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
CURRENTLY_PLAYING_URL = "https://api.spotify.com/v1/me/player/currently-playing"
# Stato playback completo (scope user-read-playback-state): come currently-playing
# ma con in piu' il device attivo (volume_percent, supports_volume). Il poller
# lo preferisce e ripiega su currently-playing coi token a scope vecchio.
PLAYER_URL = "https://api.spotify.com/v1/me/player"
# Volume del device attivo (scope user-modify-playback-state e Premium):
# PUT con volume_percent 0-100 in query. Successo = 204 No Content.
PLAYER_VOLUME_URL = "https://api.spotify.com/v1/me/player/volume"
VOLUME_MIN_PERCENT = 0
VOLUME_MAX_PERCENT = 100
# Comandi playback (richiedono scope user-modify-playback-state e Premium):
# mappa azione -> (metodo HTTP, URL). Successo = 204 No Content.
PLAYBACK_ENDPOINTS: dict[str, tuple[str, str]] = {
    "play": ("PUT", "https://api.spotify.com/v1/me/player/play"),
    "pause": ("PUT", "https://api.spotify.com/v1/me/player/pause"),
    "next": ("POST", "https://api.spotify.com/v1/me/player/next"),
    "previous": ("POST", "https://api.spotify.com/v1/me/player/previous"),
}
# Azioni playback valide (per la validazione nel server di controllo).
PLAYBACK_ACTIONS = frozenset(PLAYBACK_ENDPOINTS)

# --- Parametri OAuth PKCE ---
# Scope: lettura del brano corrente, stato playback (device/volume) e controllo
# del playback (play/pause/next/previous/volume dal pannello). NOTA: i token
# gia' salvati con scope piu' vecchi non coprono le funzioni nuove: serve
# rifare il login una volta (il poller nel frattempo degrada da solo).
SCOPE = "user-read-currently-playing user-read-playback-state user-modify-playback-state"
# redirect_uri: Spotify VIETA "localhost", serve l'IP letterale 127.0.0.1.
REDIRECT_HOST = "127.0.0.1"
REDIRECT_PORT = 8888
REDIRECT_PATH = "/callback"
REDIRECT_URI = f"http://{REDIRECT_HOST}:{REDIRECT_PORT}{REDIRECT_PATH}"
# Metodo PKCE: SHA256 del verifier (S256), l'unico consigliato da Spotify.
CODE_CHALLENGE_METHOD = "S256"
GRANT_AUTHORIZATION_CODE = "authorization_code"
GRANT_REFRESH_TOKEN = "refresh_token"

# --- Generazione del code_verifier PKCE (RFC 7636) ---
# Alfabeto ammesso per il verifier: unreserved characters [A-Za-z0-9._~-].
VERIFIER_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
# Lunghezza del verifier generato (RFC 7636 ammette 43-128 caratteri).
VERIFIER_LENGTH = 64
# Byte casuali per lo state anti-CSRF (poi esadecimale, quindi 32 caratteri).
STATE_NBYTES = 16

# --- Login interattivo (server locale di callback) ---
# Timeout complessivo per completare il login nel browser.
LOGIN_TIMEOUT_SECONDS = 180.0
# Pagina mostrata nel browser al termine del login.
LOGIN_SUCCESS_HTML = (
    "<html><body><h2>Login riuscito, puoi chiudere questa scheda</h2></body></html>"
)
LOGIN_ERROR_HTML = "<html><body><h2>Login fallito. Torna al terminale.</h2></body></html>"

# --- Cache dei token ---
# Margine sottratto alla scadenza reale: rinnova il token un po' prima che scada
# davvero, per non farsi trovare con un token spirato a meta' richiesta.
TOKEN_EXPIRY_MARGIN_SECONDS = 60

# --- HTTP ---
HTTP_OK = 200
HTTP_NO_CONTENT = 204
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429
# Timeout per lo scambio/refresh dei token OAuth.
TOKEN_TIMEOUT_SECONDS = 10.0
# Timeout per la GET dell'endpoint currently-playing.
API_TIMEOUT_SECONDS = 10.0
# Attesa di default in caso di 429 senza header Retry-After valido.
DEFAULT_RETRY_AFTER_SECONDS = 5.0

# --- Polling ---
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
# Tipo di contenuto che sappiamo gestire (podcast/episodi vengono ignorati).
TRACK_TYPE = "track"


class SpotifyError(Exception):
    """Errore di autenticazione o API Spotify."""


def _code_challenge(verifier: str) -> str:
    """Calcola il code_challenge PKCE (S256) da un code_verifier.

    Applica SHA256 al verifier codificato ASCII e ne restituisce il base64url
    senza padding, come richiesto da RFC 7636 e dalla doc Spotify. Esposta come
    funzione a se' (con verifier in ingresso) per essere testabile col test
    vector di RFC 7636.

    Args:
        verifier: Il code_verifier PKCE (caratteri ASCII dell'alfabeto ammesso).

    Returns:
        Il code_challenge: base64url(SHA256(verifier)) senza il padding "=".
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _generate_code_verifier() -> str:
    """Genera un code_verifier PKCE casuale.

    Returns:
        Una stringa di `VERIFIER_LENGTH` caratteri dall'alfabeto unreserved
        [A-Za-z0-9._~-], generata con `secrets` (adatto a scopi crittografici).
    """
    return "".join(secrets.choice(VERIFIER_ALPHABET) for _ in range(VERIFIER_LENGTH))


class SpotifyAuth:
    """Gestisce l'autenticazione OAuth PKCE verso Spotify e la cache dei token.

    Implementa Authorization Code + PKCE per client pubblico (nessun
    client_secret): `interactive_login` fa il login una volta via browser e
    salva i token su file; `get_access_token` restituisce un access token valido,
    rinnovandolo automaticamente col refresh token alla scadenza.

    Attributes:
        client_id: Il Client ID dell'app Spotify.
        cache_path: Percorso del file JSON con i token.
    """

    def __init__(self, client_id: str, cache_path: Path) -> None:
        """Inizializza l'autenticatore.

        Args:
            client_id: Client ID dell'applicazione Spotify (client pubblico).
            cache_path: File JSON dove salvare/leggere access token, refresh
                token e scadenza.
        """
        self.client_id = client_id
        self.cache_path = cache_path

    def has_tokens(self) -> bool:
        """Indica se esiste una cache di token gia' salvata.

        Returns:
            True se il file di cache esiste (non ne verifica la validita').
        """
        return self.cache_path.exists()

    def interactive_login(self) -> None:
        """Esegue il login OAuth interattivo via browser e salva i token.

        Genera verifier/challenge PKCE e uno state anti-CSRF, apre il browser
        sull'URL di authorize, avvia un server locale su `REDIRECT_URI` che
        cattura la singola richiesta di callback, verifica lo state, scambia il
        code con i token e li salva nella cache. Blocca fino al completamento o
        al timeout.

        Raises:
            SpotifyError: Se il login non si completa entro
                `LOGIN_TIMEOUT_SECONDS`, se Spotify restituisce un errore, se lo
                state non corrisponde o se lo scambio del code fallisce.
        """
        verifier = _generate_code_verifier()
        challenge = _code_challenge(verifier)
        state = secrets.token_hex(STATE_NBYTES)

        authorize_url = self._build_authorize_url(challenge, state)
        print("Apro il browser per il login a Spotify.")
        print(f"Se non si apre, incolla questo URL nel browser:\n{authorize_url}")
        webbrowser.open(authorize_url)

        code = self._await_callback_code(state)
        self._exchange_code(code, verifier)
        print("Login a Spotify riuscito. Token salvati.")

    def get_access_token(self) -> str:
        """Restituisce un access token valido, rinnovandolo se scaduto.

        Legge la cache; se il token e' scaduto (oltre `expires_at`) esegue un
        refresh e aggiorna la cache. Non fa login interattivo: se la cache manca,
        segnala all'utente il comando da eseguire.

        Returns:
            Un access token Spotify valido.

        Raises:
            SpotifyError: Se la cache non esiste o se il refresh fallisce.
        """
        tokens = self._load_tokens()
        if tokens is None:
            raise SpotifyError("Cache token assente. Esegui: python -m huebeat spotify-login")

        if time.time() >= tokens["expires_at"]:
            logger.info("Access token Spotify scaduto: eseguo il refresh.")
            tokens = self._refresh(tokens["refresh_token"])

        return tokens["access_token"]

    def _build_authorize_url(self, challenge: str, state: str) -> str:
        """Compone l'URL completo dell'endpoint authorize.

        Args:
            challenge: Il code_challenge PKCE (S256).
            state: Lo state anti-CSRF da verificare al ritorno.

        Returns:
            L'URL di authorize con tutti i parametri di query.
        """
        query = urlencode(
            {
                "client_id": self.client_id,
                "response_type": "code",
                "redirect_uri": REDIRECT_URI,
                "code_challenge_method": CODE_CHALLENGE_METHOD,
                "code_challenge": challenge,
                "state": state,
                "scope": SCOPE,
            }
        )
        return f"{AUTHORIZE_URL}?{query}"

    def _await_callback_code(self, expected_state: str) -> str:
        """Avvia il server locale e attende il code dal redirect di Spotify.

        Args:
            expected_state: Lo state generato all'avvio, da confrontare con
                quello restituito nel callback.

        Returns:
            Il codice di autorizzazione (`code`) ricevuto nel callback.

        Raises:
            SpotifyError: Se scade il timeout, se lo state non corrisponde o se
                Spotify restituisce un errore invece del code.
        """
        result: dict[str, str] = {}
        handler = _make_callback_handler(expected_state, result)
        server = HTTPServer((REDIRECT_HOST, REDIRECT_PORT), handler)
        server.timeout = LOGIN_TIMEOUT_SECONDS

        try:
            # handle_request serve UNA sola richiesta e ritorna; con timeout
            # scaduto ritorna senza aver popolato `result`.
            server.handle_request()
        finally:
            server.server_close()

        if "error" in result:
            raise SpotifyError(f"Login Spotify rifiutato: {result['error']}")
        if "code" not in result:
            raise SpotifyError(f"Login non completato entro {LOGIN_TIMEOUT_SECONDS:.0f} s.")
        return result["code"]

    def _exchange_code(self, code: str, verifier: str) -> None:
        """Scambia il code di autorizzazione con access e refresh token.

        Args:
            code: Il codice di autorizzazione ricevuto nel callback.
            verifier: Il code_verifier PKCE usato per generare il challenge.

        Raises:
            SpotifyError: Se la richiesta al token endpoint fallisce.
        """
        data = {
            "grant_type": GRANT_AUTHORIZATION_CODE,
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": self.client_id,
            "code_verifier": verifier,
        }
        payload = self._post_token(data)
        self._save_tokens(
            access_token=payload["access_token"],
            refresh_token=payload["refresh_token"],
            expires_in=payload["expires_in"],
        )

    def _refresh(self, refresh_token: str) -> dict:
        """Rinnova l'access token usando il refresh token e aggiorna la cache.

        La risposta di refresh PUO' non contenere un nuovo refresh_token: in quel
        caso si conserva quello vecchio (semantica verificata sulla doc Spotify).

        Args:
            refresh_token: Il refresh token corrente.

        Returns:
            Il dict dei token aggiornati (access_token, refresh_token,
            expires_at), gia' salvato in cache.

        Raises:
            SpotifyError: Se la richiesta di refresh fallisce.
        """
        data = {
            "grant_type": GRANT_REFRESH_TOKEN,
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        }
        payload = self._post_token(data)
        # Spotify puo' omettere un nuovo refresh_token: conserva il vecchio.
        new_refresh_token = payload.get("refresh_token") or refresh_token
        return self._save_tokens(
            access_token=payload["access_token"],
            refresh_token=new_refresh_token,
            expires_in=payload["expires_in"],
        )

    def _post_token(self, data: dict) -> dict:
        """Esegue una POST al token endpoint di Spotify.

        Client pubblico: nessun client_secret e nessun header Authorization; il
        client_id viaggia nel body. Content-Type
        application/x-www-form-urlencoded (httpx lo imposta passando `data`).

        Args:
            data: Campi del corpo form-urlencoded.

        Returns:
            Il JSON di risposta come dict.

        Raises:
            SpotifyError: Se la richiesta fallisce o la risposta non e' valida.
        """
        try:
            response = httpx.post(TOKEN_URL, data=data, timeout=TOKEN_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as err:
            raise SpotifyError(f"Richiesta token a Spotify fallita: {err}") from err

    def _load_tokens(self) -> dict | None:
        """Legge i token dalla cache su file.

        Returns:
            Il dict dei token, o None se il file non esiste.

        Raises:
            SpotifyError: Se il file esiste ma e' illeggibile o malformato.
        """
        if not self.cache_path.exists():
            return None
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            raise SpotifyError(
                f"Cache token Spotify illeggibile ({self.cache_path}): {err}"
            ) from err

    def _save_tokens(self, access_token: str, refresh_token: str, expires_in: int) -> dict:
        """Salva i token nella cache, calcolando la scadenza assoluta.

        La scadenza salvata e' `ora + expires_in - margine`, cosi' il token viene
        rinnovato con un po' di anticipo rispetto alla scadenza reale.

        Args:
            access_token: Access token da salvare.
            refresh_token: Refresh token da salvare.
            expires_in: Durata di validita' dell'access token in secondi.

        Returns:
            Il dict dei token salvato (access_token, refresh_token, expires_at).
        """
        tokens = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": time.time() + expires_in - TOKEN_EXPIRY_MARGIN_SECONDS,
        }
        with _save_lock:
            _atomic_write_text(self.cache_path, json.dumps(tokens))
        return tokens


# Registro di modulo delle istanze SpotifyAuth per file di token: una sola
# istanza per percorso, condivisa tra poller e server di controllo, cosi' non
# convivono due autenticatori che rinnovano e riscrivono lo stesso file.
_auth_registry: dict[Path, SpotifyAuth] = {}
_auth_registry_lock = threading.Lock()


def get_auth(client_id: str, cache_path: Path) -> SpotifyAuth:
    """Restituisce l'istanza SpotifyAuth condivisa per un file di token.

    Mantiene una sola istanza per `cache_path` (registro di modulo protetto da
    lock): poller e server di controllo condividono lo stesso autenticatore
    invece di crearne uno per ogni comando, evitando due oggetti che rinnovano e
    riscrivono in parallelo lo stesso file. Se il client id e' cambiato (nuovo
    login) aggiorna quello dell'istanza gia' registrata.

    Args:
        client_id: Client ID dell'app Spotify (public client PKCE).
        cache_path: Percorso del file dei token, chiave del registro.

    Returns:
        L'istanza SpotifyAuth condivisa per quel percorso.
    """
    key = Path(cache_path)
    with _auth_registry_lock:
        auth = _auth_registry.get(key)
        if auth is None:
            auth = SpotifyAuth(client_id, key)
            _auth_registry[key] = auth
        elif auth.client_id != client_id:
            auth.client_id = client_id
        return auth


def send_playback_command(auth: SpotifyAuth, action: str) -> None:
    """Invia un comando di controllo playback all'API Spotify.

    Esegue la richiesta all'endpoint corrispondente all'azione (vedi
    `PLAYBACK_ENDPOINTS`) col bearer token fornito da `auth`. Richiede un
    account Premium e un token con lo scope `user-modify-playback-state`
    (i login fatti col vecchio scope di sola lettura vanno rifatti).

    Args:
        auth: Autenticatore Spotify gia' collegato (token in cache).
        action: Una tra "play", "pause", "next", "previous".

    Raises:
        ValueError: Se `action` non e' tra quelle supportate (bug del
            chiamante, non un errore dell'API).
        SpotifyError: Se il token e' rifiutato (401), se manca Premium o lo
            scope e' quello vecchio (403), se nessun dispositivo e' attivo
            (404), se Spotify limita le richieste (429), per gli altri status
            inattesi o per errori di rete.
    """
    if action not in PLAYBACK_ENDPOINTS:
        raise ValueError(f"azione playback sconosciuta: {action}")
    method, url = PLAYBACK_ENDPOINTS[action]
    token = auth.get_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = httpx.request(method, url, headers=headers, timeout=API_TIMEOUT_SECONDS)
    except httpx.HTTPError as err:
        raise SpotifyError(f"Comando playback non inviato (errore di rete): {err}") from err
    _raise_for_player_status(response)


def set_volume(auth: SpotifyAuth, percent: int) -> None:
    """Imposta il volume del device Spotify attivo.

    Esegue la PUT su `PLAYER_VOLUME_URL` con volume_percent in query. Richiede
    un account Premium, un token con lo scope `user-modify-playback-state` e
    un device che supporti il controllo volume.

    Args:
        auth: Autenticatore Spotify gia' collegato (token in cache).
        percent: Volume desiderato, intero 0-100.

    Raises:
        ValueError: Se `percent` non e' un intero nel range 0-100 (bug del
            chiamante, non un errore dell'API).
        SpotifyError: Stessi casi di `send_playback_command` (401/403/404/429,
            status inattesi, errori di rete).
    """
    if not isinstance(percent, int) or isinstance(percent, bool):
        raise ValueError(f"volume non intero: {percent!r}")
    if not (VOLUME_MIN_PERCENT <= percent <= VOLUME_MAX_PERCENT):
        raise ValueError(f"volume fuori range {VOLUME_MIN_PERCENT}-{VOLUME_MAX_PERCENT}: {percent}")
    token = auth.get_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = httpx.put(
            PLAYER_VOLUME_URL,
            params={"volume_percent": percent},
            headers=headers,
            timeout=API_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as err:
        raise SpotifyError(f"Comando volume non inviato (errore di rete): {err}") from err
    _raise_for_player_status(response)


def _raise_for_player_status(response: httpx.Response) -> None:
    """Traduce gli status di errore degli endpoint player in SpotifyError.

    Args:
        response: La risposta HTTP di un comando player (playback o volume).

    Raises:
        SpotifyError: Con messaggio azionabile per 401/403/404/429 e generico
            per gli altri status non 2xx. Nessuna eccezione sui 2xx.
    """
    if response.is_success:  # la doc promette 204, ma ogni 2xx e' un successo
        return
    if response.status_code == HTTP_UNAUTHORIZED:
        raise SpotifyError("Token Spotify rifiutato: riprova, o rifai il login dal pannello.")
    if response.status_code == HTTP_FORBIDDEN:
        raise SpotifyError(
            "Spotify ha rifiutato il comando: serve Premium oppure un nuovo login "
            "(scope aggiornato). Rifai il login Spotify dal pannello."
        )
    if response.status_code == HTTP_NOT_FOUND:
        raise SpotifyError(
            "Nessun dispositivo Spotify attivo: avvia la riproduzione una volta dall'app Spotify."
        )
    if response.status_code == HTTP_TOO_MANY_REQUESTS:
        raise SpotifyError("Troppe richieste a Spotify: riprova tra qualche secondo.")
    raise SpotifyError(f"Comando rifiutato da Spotify (HTTP {response.status_code}).")


def _make_callback_handler(
    expected_state: str, result: dict[str, str]
) -> type[BaseHTTPRequestHandler]:
    """Crea la classe handler per la singola richiesta di callback OAuth.

    L'handler estrae `code`/`state`/`error` dalla query, verifica lo state, scrive
    l'esito in `result` (per mutazione, cosi' il chiamante lo legge dopo
    `handle_request`) e risponde con una paginetta HTML.

    Args:
        expected_state: Lo state atteso, da confrontare con quello ricevuto.
        result: Dict mutato in-place con le chiavi `code` o `error`.

    Returns:
        Una sottoclasse di `BaseHTTPRequestHandler` pronta per `HTTPServer`.
    """

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (nome imposto da BaseHTTPRequestHandler)
            parsed = urlparse(self.path)
            if parsed.path != REDIRECT_PATH:
                self._respond(HTTP_NO_CONTENT, "")
                return

            params = parse_qs(parsed.query)
            error = params.get("error", [None])[0]
            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]

            if error is not None:
                result["error"] = error
                self._respond(HTTP_OK, LOGIN_ERROR_HTML)
                return
            if state != expected_state:
                result["error"] = "state mismatch (possibile CSRF)"
                self._respond(HTTP_OK, LOGIN_ERROR_HTML)
                return
            if code is None:
                result["error"] = "code assente nel callback"
                self._respond(HTTP_OK, LOGIN_ERROR_HTML)
                return

            result["code"] = code
            self._respond(HTTP_OK, LOGIN_SUCCESS_HTML)

        def _respond(self, status: int, body: str) -> None:
            """Scrive una risposta HTML minimale."""
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if body:
                self.wfile.write(body.encode("utf-8"))

        def log_message(self, *_args) -> None:
            """Silenzia il logging su stderr di BaseHTTPRequestHandler."""

    return _CallbackHandler


@dataclass(frozen=True)
class NowPlaying:
    """Snapshot immutabile del brano attualmente in riproduzione.

    Attributes:
        track_id: ID Spotify della traccia.
        title: Titolo del brano.
        artists: Artisti uniti da ", ".
        is_playing: True se la riproduzione e' in corso (non in pausa).
        volume_percent: Volume del device attivo (0-100), o None se ignoto
            (endpoint currently-playing, device senza controllo volume, o
            token senza lo scope user-read-playback-state).
    """

    track_id: str
    title: str
    artists: str
    is_playing: bool
    volume_percent: int | None = None


class NowPlayingPoller:
    """Thread daemon che tiene aggiornati brano corrente e palette da Spotify.

    Interroga l'endpoint currently-playing ogni `poll_interval_s`. Espone due
    snapshot letti da altri thread (`now_playing`, `palette`): la palette viene
    generata (deterministica dall'id traccia) solo quando cambia la traccia.
    Gli errori di rete e i 401 transitori sono gestiti senza far morire il
    thread.

    Thread-safety: le proprieta' `now_playing` e `palette` restituiscono
    riferimenti a oggetti immutabili (dataclass frozen) o a nuove liste, e
    l'aggiornamento avviene per singola assegnazione di attributo. In CPython il
    GIL rende atomica l'assegnazione/lettura di un riferimento, quindi un lettore
    vede sempre uno snapshot coerente (quello vecchio o quello nuovo, mai uno
    stato intermedio) senza bisogno di lock.

    Attributes:
        auth: L'autenticatore che fornisce l'access token.
        poll_interval_s: Intervallo tra due polling, in secondi.
    """

    def __init__(
        self, auth: SpotifyAuth, poll_interval_s: float = DEFAULT_POLL_INTERVAL_SECONDS
    ) -> None:
        """Inizializza il poller (senza avviarlo).

        Args:
            auth: Autenticatore Spotify gia' configurato.
            poll_interval_s: Secondi di attesa tra un polling e il successivo.
        """
        self.auth = auth
        self.poll_interval_s = poll_interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Snapshot condivisi (assegnazione atomica per il GIL, vedi docstring).
        self._now_playing: NowPlaying | None = None
        self._palette: list[tuple[float, float, float]] | None = None
        # Ultima traccia per cui abbiamo calcolato la palette (per rilevare i cambi).
        self._palette_track_id: str | None = None
        # True dopo un 403 dall'endpoint player: token con scope vecchio, si
        # ripiega su currently-playing (niente volume) fino al prossimo login.
        self._player_scope_missing = False

    @property
    def palette(self) -> list[tuple[float, float, float]] | None:
        """La palette generata per il brano corrente, o None se nulla in riproduzione."""
        return self._palette

    @property
    def now_playing(self) -> NowPlaying | None:
        """Lo snapshot del brano corrente, o None se nulla in riproduzione."""
        return self._now_playing

    def start(self) -> None:
        """Avvia il thread di polling in background (daemon).

        Idempotente: se il thread e' gia' attivo non fa nulla.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="spotify-poller", daemon=True)
        self._thread.start()
        logger.info("Poller Spotify avviato (intervallo %.1f s).", self.poll_interval_s)

    def stop(self) -> None:
        """Ferma il thread di polling e ne attende la terminazione."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        logger.info("Poller Spotify fermato.")

    def _run(self) -> None:
        """Loop principale del thread: interroga Spotify fino allo stop.

        Ogni giro attende `poll_interval_s` (interrompibile dallo stop). Le
        eccezioni non gestite dai singoli passi vengono loggate senza far morire
        il thread.
        """
        while not self._stop_event.is_set():
            wait_seconds = self.poll_interval_s
            try:
                wait_seconds = self._poll_once()
            except SpotifyError as err:
                logger.warning("Errore di autenticazione Spotify nel polling: %s", err)
            except httpx.HTTPError as err:
                logger.warning("Errore di rete nel polling Spotify: %s", err)
            except Exception:
                logger.exception("Errore imprevisto nel polling Spotify.")
            # wait() ritorna prima se arriva lo stop: niente ritardo alla chiusura.
            self._stop_event.wait(wait_seconds)

    def _poll_once(self) -> float:
        """Esegue un singolo polling del brano corrente.

        Aggiorna gli snapshot in base alla risposta e, al cambio di traccia,
        ricalcola la palette. Gestisce 200/204/401/429.

        Returns:
            I secondi da attendere prima del prossimo giro (di norma
            `poll_interval_s`; il valore di Retry-After in caso di 429).
        """
        token = self.auth.get_access_token()
        response = self._get_playback_state(token)

        if response.status_code == HTTP_UNAUTHORIZED:
            # Token forse rifiutato nonostante la cache: forza un refresh e ritenta.
            logger.info("401 da Spotify: forzo un refresh del token e ritento.")
            token = self._force_refresh()
            if token is None:
                return self.poll_interval_s
            response = self._get_playback_state(token)
            if response.status_code == HTTP_UNAUTHORIZED:
                logger.warning("Ancora 401 dopo il refresh: continuo il polling.")
                return self.poll_interval_s

        if response.status_code == HTTP_FORBIDDEN and not self._player_scope_missing:
            # L'endpoint player richiede user-read-playback-state: coi token
            # vecchi degrada a currently-playing (tutto funziona tranne il
            # volume) invece di rompere brano e palette.
            self._player_scope_missing = True
            logger.info(
                "403 dall'endpoint player (token con scope vecchio): fallback su "
                "currently-playing, volume non disponibile. Rifai il login "
                "Spotify per abilitarlo."
            )
            response = self._get_playback_state(token)

        if response.status_code == HTTP_TOO_MANY_REQUESTS:
            retry_after = self._parse_retry_after(response)
            logger.info("429 da Spotify: attendo %.1f s (Retry-After).", retry_after)
            return retry_after

        if response.status_code == HTTP_NO_CONTENT:
            self._set_idle()
            return self.poll_interval_s

        if response.status_code == HTTP_OK:
            self._handle_track_payload(response.json())
            return self.poll_interval_s

        logger.warning("Status inatteso da Spotify: %d", response.status_code)
        return self.poll_interval_s

    def _get_playback_state(self, token: str) -> httpx.Response:
        """Chiama l'endpoint di stato playback con il bearer token.

        Usa `PLAYER_URL` (stato completo con device/volume); dopo un 403 da
        scope vecchio (`_player_scope_missing`) ripiega su
        `CURRENTLY_PLAYING_URL`, che ha lo stesso payload salvo `device`.

        Args:
            token: Access token valido.

        Returns:
            La risposta HTTP grezza (lo status viene interpretato dal chiamante).
        """
        url = CURRENTLY_PLAYING_URL if self._player_scope_missing else PLAYER_URL
        headers = {"Authorization": f"Bearer {token}"}
        return httpx.get(url, headers=headers, timeout=API_TIMEOUT_SECONDS)

    def _force_refresh(self) -> str | None:
        """Forza un refresh del token dopo un 401, senza far morire il thread.

        Returns:
            Il nuovo access token, o None se il refresh e' fallito.
        """
        try:
            # Invalida la cache in memoria forzando il ramo di refresh: rileggo la
            # cache e, se ancora valida in scadenza, get_access_token la rinnova.
            # Per forzare comunque il refresh scado esplicitamente il token.
            self._expire_cached_token()
            return self.auth.get_access_token()
        except SpotifyError as err:
            logger.warning("Refresh del token Spotify fallito dopo 401: %s", err)
            return None

    def _expire_cached_token(self) -> None:
        """Marca il token in cache come scaduto per forzare il refresh.

        Riscrive `expires_at` nel passato cosi' che il prossimo
        `get_access_token` esegua un refresh. Silenzioso se la cache manca.
        """
        tokens = self.auth._load_tokens()  # noqa: SLF001 (cooperazione interna al modulo)
        if tokens is None:
            return
        self.auth._save_tokens(  # noqa: SLF001
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            expires_in=0 - int(TOKEN_EXPIRY_MARGIN_SECONDS),
        )

    def _parse_retry_after(self, response: httpx.Response) -> float:
        """Legge l'header Retry-After (secondi) da una risposta 429.

        Args:
            response: La risposta HTTP con status 429.

        Returns:
            I secondi da attendere; `DEFAULT_RETRY_AFTER_SECONDS` se l'header
            manca o non e' un intero valido.
        """
        raw = response.headers.get("Retry-After")
        if raw is None:
            return DEFAULT_RETRY_AFTER_SECONDS
        try:
            return float(int(raw))
        except ValueError:
            return DEFAULT_RETRY_AFTER_SECONDS

    def _handle_track_payload(self, payload: dict) -> None:
        """Interpreta il JSON di currently-playing e aggiorna gli snapshot.

        Se il contenuto non e' una traccia (es. podcast) o `item` e' assente,
        tratta la situazione come "nulla in riproduzione".

        Args:
            payload: Il JSON decodificato della risposta 200.
        """
        item = payload.get("item")
        if payload.get("currently_playing_type") != TRACK_TYPE or item is None:
            self._set_idle()
            return

        # device c'e' solo nella risposta di PLAYER_URL; volume solo se il
        # device lo supporta (supports_volume, es. no su certi Connect).
        device = payload.get("device") or {}
        volume = device.get("volume_percent")
        if not isinstance(volume, int) or device.get("supports_volume") is False:
            volume = None

        now = NowPlaying(
            track_id=item.get("id", ""),
            title=item.get("name", ""),
            artists=", ".join(a.get("name", "") for a in item.get("artists", [])),
            is_playing=bool(payload.get("is_playing", False)),
            volume_percent=volume,
        )
        self._now_playing = now

        # Ricalcola la palette solo al cambio di traccia.
        if now.track_id and now.track_id != self._palette_track_id:
            self._update_palette(now)

    def _update_palette(self, now: NowPlaying) -> None:
        """Genera la palette per una nuova traccia.

        Deterministica dall'id della traccia (nessuna rete, nessun fallimento
        atteso): stesso brano, stessa palette a ogni riproduzione.

        Args:
            now: Lo snapshot della traccia corrente.
        """
        self._palette = generate_palette(now.track_id)
        self._palette_track_id = now.track_id
        logger.info("Palette aggiornata per '%s - %s'.", now.artists, now.title)

    def _set_idle(self) -> None:
        """Porta gli snapshot allo stato "nulla in riproduzione"."""
        self._now_playing = None
        self._palette = None
        self._palette_track_id = None
