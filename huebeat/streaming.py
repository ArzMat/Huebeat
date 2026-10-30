"""Streaming DTLS verso il Bridge Hue (Entertainment API v2).

Espone `EntertainmentStream`, che apre una sessione DTLS-PSK verso la porta
2100 del bridge e invia frame HueStream v2 con i colori RGB dei canali
dell'area Entertainment. L'attivazione dell'area (action "start" via REST) e'
responsabilita' del chiamante e deve avvenire PRIMA di `connect()`.

PSK con hue-application-id come identity e clientkey come chiave (pattern
LedFx); layout del frame verificato byte per byte su hue-entertainment-pykit.

Trasporto DTLS: `huebeat.dtls.DtlsConnection` (puro Python su `cryptography`,
vendorizzato da music-assistant/hue-entertainment). Prima qui
c'era python-mbedtls: il firmware bridge 1.79.1978293000 ha iniziato a
scartarne in silenzio il ClientHello e la libreria (archiviata) non permetteva
rimedi. Il trasporto resta sostituibile dietro la
stessa interfaccia (`EntertainmentStream.connect/send/close`): le call site
che parlano col layer DTLS incapsulano ogni errore in `StreamingError`,
nessuna eccezione del trasporto deve trapelare grezza al resto del codice.
"""

import logging
import struct
import time
from collections.abc import Callable

from huebeat.dtls import DtlsConnection, DtlsProtocolError

logger = logging.getLogger(__name__)

# --- Protocollo HueStream v2 ---
_PROTOCOL_NAME = b"HueStream"
_VERSION_MAJOR = 0x02
_VERSION_MINOR = 0x00
# LedFx usa sequence id fisso 0 (pykit usa 0x07 fisso): non serve incrementare.
_SEQUENCE_ID = 0x00
_RESERVED_16BIT = 0x0000
_COLOR_SPACE_RGB = 0x00
_RESERVED_8BIT = 0x00
# L'entertainment configuration id viaggia come UUID ASCII di 36 caratteri.
_CONFIG_ID_LENGTH = 36
# Per canale: 1 byte di channel id + 3 componenti colore a 16 bit big-endian.
_BYTES_PER_CHANNEL = 7
_MAX_CHANNELS = 20
_MAX_CHANNEL_ID = 255
# Fondo scala di una componente colore a 16 bit.
_COLOR_FULL_SCALE = 65535

# --- Trasporto DTLS ---
_DTLS_PORT = 2100
_HANDSHAKE_MAX_ATTEMPTS = 10
_HANDSHAKE_RETRY_DELAY_S = 0.2


class StreamingError(Exception):
    """Errore dello stream Entertainment."""


class FrameValidationError(StreamingError):
    """Frame non valido (canali oltre limite, channel id o config id malformati).

    Sottoclasse di `StreamingError` per retro-compatibilita' (chi cattura
    `StreamingError` continua a intercettarla). Distingue pero' un errore
    DETERMINISTICO di costruzione del frame da un errore di TRASPORTO: il primo
    non si risolve riconnettendo, il secondo si'.
    """


def _color_component_to_16bit(value: float) -> int:
    """Converte una componente colore float [0, 1] in intero a 16 bit.

    Valori fuori range vengono riportati (clamp) dentro [0, 1].

    Args:
        value: Componente colore in [0, 1].

    Returns:
        Intero in [0, 65535].
    """
    return int(max(0.0, min(1.0, value)) * _COLOR_FULL_SCALE)


def _build_frame(
    config_id: str,
    entries: list[tuple[int, tuple[float, float, float]]],
) -> bytes:
    """Costruisce un frame HueStream v2 completo.

    Layout: 16 byte di header, 36 byte di entertainment configuration id
    (UUID ASCII), poi 7 byte per canale (1 byte channel id + RGB a 16 bit
    big-endian).

    Args:
        config_id: UUID dell'entertainment configuration (36 caratteri).
        entries: Lista di tuple (channel_id, (r, g, b)) con componenti float
            in [0, 1]. Massimo 20 canali.

    Returns:
        Il frame binario pronto per l'invio sul socket DTLS.

    Raises:
        FrameValidationError: Se le entries superano i 20 canali, se un channel
            id e' fuori da [0, 255] o se il config id non e' di 36 caratteri.
    """
    if len(entries) > _MAX_CHANNELS:
        raise FrameValidationError(f"troppi canali: {len(entries)} (massimo {_MAX_CHANNELS})")
    config_id_bytes = config_id.encode("utf-8")
    if len(config_id_bytes) != _CONFIG_ID_LENGTH:
        raise FrameValidationError(
            f"entertainment config id di {len(config_id_bytes)} byte "
            f"(attesi {_CONFIG_ID_LENGTH}): {config_id!r}"
        )

    header = struct.pack(
        ">9sBBBHBB",
        _PROTOCOL_NAME,
        _VERSION_MAJOR,
        _VERSION_MINOR,
        _SEQUENCE_ID,
        _RESERVED_16BIT,
        _COLOR_SPACE_RGB,
        _RESERVED_8BIT,
    )

    parts = [header, config_id_bytes]
    for channel_id, (red, green, blue) in entries:
        if not 0 <= channel_id <= _MAX_CHANNEL_ID:
            raise FrameValidationError(
                f"channel id fuori range [0, {_MAX_CHANNEL_ID}]: {channel_id}"
            )
        parts.append(
            struct.pack(
                ">BHHH",
                channel_id,
                _color_component_to_16bit(red),
                _color_component_to_16bit(green),
                _color_component_to_16bit(blue),
            )
        )
    return b"".join(parts)


class EntertainmentStream:
    """Sessione di streaming DTLS verso il Bridge Hue (porta 2100).

    Ciclo d'uso: `connect()`, poi `send()` a cadenza regolare (il bridge si
    aspetta frame continui), infine `close()`. L'area Entertainment deve
    essere gia' attiva (action "start" via REST) prima di `connect()`.

    Attributes:
        Nessun attributo pubblico: lo stato della sessione e' interno.
    """

    def __init__(
        self,
        host: str,
        application_id: str,
        clientkey_hex: str,
        entertainment_config_id: str,
    ) -> None:
        """Prepara la sessione senza aprire connessioni.

        Args:
            host: IP o hostname del bridge Hue.
            application_id: hue-application-id (UUID), usato come identity PSK
                (NON lo username REST).
            clientkey_hex: Clientkey del pairing, stringa esadecimale.
            entertainment_config_id: UUID dell'entertainment configuration.

        Raises:
            StreamingError: Se la clientkey non e' una stringa esadecimale
                valida.
        """
        self._host = host
        self._application_id = application_id
        try:
            self._psk = bytes.fromhex(clientkey_hex)
        except ValueError as exc:
            raise StreamingError(f"clientkey non valida (hex atteso): {exc}") from exc
        self._config_id = entertainment_config_id
        self._conn: DtlsConnection | None = None

    def connect(
        self,
        abort: Callable[[], bool] | None = None,
        max_attempts: int = _HANDSHAKE_MAX_ATTEMPTS,
    ) -> None:
        """Apre la connessione DTLS-PSK e fa l'handshake.

        Socket bloccante con timeout di 5 secondi (scelta annotata: piu'
        semplice del non bloccante di LedFx). OGNI tentativo usa una
        connessione NUOVA (socket e stato freschi): ritentare l'handshake
        sulla stessa non recupera mai (osservato:
        quando il primo scambio va storto falliscono tutti i tentativi del
        ciclo, e la connessione fresca successiva entra in meno di 1 s: la
        macchina a stati DTLS resta incastrata).

        Args:
            abort: Callable senza argomenti consultato tra un tentativo di
                handshake e l'altro: se ritorna True l'handshake si
                interrompe subito (stop del chiamante), senza aspettare la
                fine del ciclo (il rollback di un worker abbandonato non deve
                stoppare l'area di una sessione piu' nuova, visto sul campo).
            max_attempts: Tentativi di handshake del ciclo. Il default e' il
                valore del setup (~5 s a tentativo, ~50 s in tutto); la
                RICONNESSIONE ne passa molti meno, per tornare al backoff
                (che riprova da solo) invece di restare muta a lungo.

        Raises:
            StreamingError: Se lo stream e' gia' connesso, se l'handshake non
                riesce entro i tentativi previsti o se `abort` lo interrompe.
        """
        if self._conn is not None:
            raise StreamingError("stream gia' connesso: chiamare close() prima di riconnettere")

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            if abort is not None and abort():
                raise StreamingError("handshake DTLS interrotto: stop richiesto dal chiamante")
            time.sleep(_HANDSHAKE_RETRY_DELAY_S)
            # Connessione creata DENTRO il try: se il costruttore (che apre il
            # socket) o l'handshake sollevano, il fd viene chiuso in ogni caso
            # e non viene mai leakato.
            conn: DtlsConnection | None = None
            try:
                conn = DtlsConnection(self._host, _DTLS_PORT, self._application_id, self._psk)
                conn.handshake()
            except (DtlsProtocolError, OSError) as exc:
                last_error = exc
                self._close_quietly(conn)
                logger.debug(
                    "Handshake DTLS, tentativo %d/%d fallito (socket nuovo al prossimo): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                continue
            except Exception as exc:
                # Errore inatteso dal layer DTLS. Non lo ritento in
                # silenzio (non e' un rifiuto transitorio noto): chiudo la
                # connessione e rilancio incapsulato, mai grezzo. Mai BaseException.
                self._close_quietly(conn)
                raise StreamingError(
                    f"handshake DTLS: errore inatteso dal layer DTLS: {exc}"
                ) from exc
            self._conn = conn
            logger.info(
                "Stream Entertainment connesso a %s:%d (tentativo %d)",
                self._host,
                _DTLS_PORT,
                attempt,
            )
            return

        raise StreamingError(
            f"handshake DTLS fallito dopo {max_attempts} tentativi "
            f"(socket nuovo a ogni tentativo): {last_error}"
        )

    def send(
        self,
        entries: list[tuple[int, tuple[float, float, float]]],
    ) -> None:
        """Invia un frame HueStream v2 con i colori dei canali indicati.

        Args:
            entries: Lista di tuple (channel_id, (r, g, b)) con componenti
                float in [0, 1]. Massimo 20 canali.

        Raises:
            FrameValidationError: Se le entries non sono valide (canali oltre
                il limite, channel id o config id malformati): errore
                deterministico, la riconnessione non lo risolve.
            StreamingError: Se lo stream non e' connesso o se l'invio sul socket
                fallisce (errore di trasporto).
        """
        if self._conn is None:
            raise StreamingError("stream non connesso: chiamare connect() prima")
        # _build_frame e' FUORI dal try di trasporto: una FrameValidationError
        # propaga com'e' e non viene scambiata per un errore di rete.
        frame = _build_frame(self._config_id, entries)
        try:
            self._conn.send(frame)
        except OSError as exc:
            raise StreamingError(f"invio frame fallito: {exc}") from exc
        except Exception as exc:
            # Qualsiasi errore inatteso dal layer DTLS incapsulato, mai grezzo.
            raise StreamingError(f"invio frame: errore inatteso dal layer DTLS: {exc}") from exc

    def close(self) -> None:
        """Chiude la connessione DTLS in modo pulito.

        Tollerante alla doppia chiamata: se lo stream e' gia' chiuso non fa
        nulla.
        """
        if self._conn is None:
            return
        self._close_quietly(self._conn)
        self._conn = None
        logger.info("Stream Entertainment chiuso")

    @staticmethod
    def _close_quietly(conn: DtlsConnection | None) -> None:
        """Chiude una connessione ignorando gli errori di chiusura.

        Accetta anche None: nei path di errore di `connect` la connessione
        puo' non esistere ancora, e in cleanup non deve mai sollevare (per
        questo, a differenza delle call site di trasporto, qui anche
        l'`Exception` inattesa viene solo loggata, non rilanciata).

        Args:
            conn: La connessione da chiudere, o None (no-op).
        """
        if conn is None:
            return
        try:
            conn.close()
        except Exception as exc:
            logger.debug("Errore inatteso ignorato in chiusura connessione: %s", exc)
