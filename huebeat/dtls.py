"""Trasporto DTLS 1.2 PSK in puro Python per lo streaming Entertainment.

Implementazione minima dell'handshake DTLS 1.2 con TLS_PSK_WITH_AES_128_GCM_SHA256
(l'unica suite che serve al Bridge Hue) sopra la libreria `cryptography`
(AES-GCM e HMAC-SHA256): niente binding C dedicati al DTLS.

Adattato da music-assistant/hue-entertainment
(https://github.com/music-assistant/hue-entertainment, Apache License 2.0).
Modifiche locali rispetto all'upstream: eccezioni di protocollo normalizzate in
`DtlsProtocolError` (gli assert upstream sono diventati raise espliciti, i
malformati non escono come IndexError/struct.error grezzi), guardia sul tipo di
record della HelloVerifyRequest, docstring del progetto. I byte sul filo sono
IDENTICI all'upstream: il ClientHello volutamente SENZA estensioni e' il motivo
per cui questo trasporto funziona dove mbedtls 2.28 falliva (il firmware bridge
1.79.1978293000 scarta in silenzio il ClientHello post-cookie di mbedtls);
anche i reset non-RFC di message_seq/record_seq
dopo la HelloVerifyRequest sono upstream e verificati accettati dal bridge.

Superficie pubblica: `DtlsConnection` (handshake/send/close) e
`DtlsProtocolError`. Il frame HueStream NON si costruisce qui: resta in
streaming.py (encoder a 16 bit verificato byte per byte).
"""

import hashlib
import hmac
import logging
import os
import socket
import struct
import time
from contextlib import suppress

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

_HANDSHAKE_TIMEOUT_S = 5.0
# Attese del ServerHello, con resend del ClientHello+cookie tra una e l'altra
# (timer stile RFC 6347: 1-2-5 s). Serve DAVVERO: il firmware bridge 1.79
# ignora deterministicamente il PRIMO ClientHello col cookie e accetta il
# resend identico (misurato: col solo timeout da 5 s l'handshake durava 8 s,
# con lo schedule ~2 s, come openssl che ritrasmette a 1 s).
_SERVER_HELLO_WAITS_S = (1.0, 2.0, 5.0)
# Attesa (facoltativa) del Finished del server a fine handshake.
_SERVER_FINISHED_TIMEOUT_S = 1.0

_DTLS_VERSION = b"\xfe\xfd"  # DTLS 1.2
_CT_HANDSHAKE = 0x16
_CT_CHANGE_CIPHER_SPEC = 0x14
_CT_APPLICATION_DATA = 0x17

_HT_CLIENT_HELLO = 0x01
_HT_SERVER_HELLO = 0x02
_HT_HELLO_VERIFY_REQUEST = 0x03
_HT_SERVER_HELLO_DONE = 0x0E
_HT_FINISHED = 0x14

# TLS_PSK_WITH_AES_128_GCM_SHA256
_CIPHER_SUITE = b"\x00\xa8"


class DtlsProtocolError(Exception):
    """Risposta del bridge malformata o inattesa durante l'handshake DTLS."""


class DtlsConnection:
    """Connessione DTLS 1.2 PSK minimale su UDP verso il bridge.

    Ciclo d'uso: costruttore (apre il socket), `handshake()`, poi `send()`
    per i frame applicativi, infine `close()`. Un'istanza copre UNA
    connessione: per ritentare l'handshake crearne una nuova (socket e stato
    freschi).
    """

    def __init__(self, host: str, port: int, identity: str, psk: bytes) -> None:
        """Apre il socket UDP e prepara lo stato dell'handshake.

        Args:
            host: IP o hostname del bridge.
            port: Porta DTLS (2100 per l'Entertainment API).
            identity: Identity PSK (stringa, va sul filo in chiaro).
            psk: Chiave PSK (clientkey del pairing, gia' decodificata da hex).
        """
        self._host = host
        self._port = port
        self._identity = identity.encode("utf-8")
        self._psk = psk
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(_HANDSHAKE_TIMEOUT_S)
        self._sock.connect((host, port))

        self._client_random = b""
        self._server_random = b""
        self._master_secret = b""
        self._epoch = 0
        self._send_seq = 0
        self._client_write_key = b""
        self._client_write_iv = b""
        self._aesgcm: AESGCM | None = None
        self._handshake_messages = bytearray()
        self._msg_seq = 0

    def handshake(self) -> None:
        """Esegue l'handshake DTLS 1.2 PSK completo.

        Raises:
            DtlsProtocolError: Risposta del bridge malformata o inattesa.
            OSError: Timeout (bridge muto) o errore del socket.
        """
        # Flight 1: ClientHello senza cookie.
        self._client_random = _make_random()
        hello1 = self._build_client_hello(cookie=b"")
        self._send_handshake(hello1, msg_type=_HT_CLIENT_HELLO)

        # Flight 2: HelloVerifyRequest col cookie.
        logger.debug("Handshake DTLS %s:%d: attendo HelloVerifyRequest", self._host, self._port)
        cookie = self._parse_hello_verify_request(self._recv())

        # Flight 3-4: ClientHello col cookie -> ServerHello (+ HelloDone).
        data = self._await_server_hello(cookie)
        self._parse_server_flight(data)
        if not self._server_random:
            raise DtlsProtocolError("nessun ServerHello nella flight del server")

        self._derive_keys()

        # Flight 5: ClientKeyExchange + ChangeCipherSpec + Finished.
        cke = struct.pack("!H", len(self._identity)) + self._identity
        self._send_handshake(cke, msg_type=0x10)  # ClientKeyExchange
        self._send_record(_CT_CHANGE_CIPHER_SPEC, b"\x01")
        self._epoch = 1
        self._send_seq = 0
        self._send_encrypted_handshake(self._build_finished())

        # Flight 6: ChangeCipherSpec + Finished del server. Letti e scartati:
        # serve solo la conferma; alcuni bridge non mandano il Finished subito,
        # quindi il timeout qui non e' un errore (comportamento upstream).
        self._sock.settimeout(_SERVER_FINISHED_TIMEOUT_S)
        logger.debug("Handshake DTLS %s:%d: attendo Finished del server", self._host, self._port)
        with suppress(TimeoutError):
            self._recv()  # ChangeCipherSpec
            self._recv()  # Finished (cifrato, non verificato)
        self._sock.settimeout(_HANDSHAKE_TIMEOUT_S)

    def send(self, plaintext: bytes) -> None:
        """Invia dati applicativi cifrati (un frame HueStream).

        Raises:
            OSError: Errore del socket.
        """
        self._send_encrypted(_CT_APPLICATION_DATA, plaintext)

    def close(self) -> None:
        """Chiude il socket UDP, senza mai sollevare."""
        with suppress(Exception):
            self._sock.close()

    # --- Flight ClientHello col cookie ---

    def _send_cookie_client_hello(self, cookie: bytes) -> None:
        """Rimanda il ClientHello con il cookie della HelloVerifyRequest.

        Azzera message_seq, record_seq e transcript come l'upstream: non e'
        RFC 6347 alla lettera, ma e' cio' che il bridge accetta (verificato).
        """
        self._msg_seq = 0
        self._send_seq = 0
        self._handshake_messages = bytearray()
        hello2 = self._build_client_hello(cookie=cookie)
        self._send_handshake(hello2, msg_type=_HT_CLIENT_HELLO)

    def _await_server_hello(self, cookie: bytes) -> bytes:
        """Aspetta il ServerHello, rimandando il ClientHello+cookie sui timeout."""
        self._send_cookie_client_hello(cookie)
        last_err: TimeoutError | None = None
        tries = len(_SERVER_HELLO_WAITS_S)
        try:
            for attempt, wait_s in enumerate(_SERVER_HELLO_WAITS_S, start=1):
                logger.debug(
                    "Handshake DTLS %s:%d: attendo ServerHello (prova %d/%d, %.0f s)",
                    self._host,
                    self._port,
                    attempt,
                    tries,
                    wait_s,
                )
                self._sock.settimeout(wait_s)
                try:
                    return self._recv()
                except TimeoutError as err:
                    last_err = err
                    if attempt < tries:
                        self._send_cookie_client_hello(cookie)
        finally:
            self._sock.settimeout(_HANDSHAKE_TIMEOUT_S)
        if last_err is None:
            raise DtlsProtocolError("attesa ServerHello uscita senza dati ne' timeout")
        raise last_err

    # --- Record layer ---

    def _send_record(self, content_type: int, fragment: bytes) -> None:
        """Invia un record DTLS in chiaro."""
        header = struct.pack(
            "!BHH6sH",
            content_type,
            0xFEFD,
            self._epoch,
            self._send_seq.to_bytes(6, "big"),
            len(fragment),
        )
        self._sock.send(header + fragment)
        self._send_seq += 1

    def _send_encrypted(self, content_type: int, plaintext: bytes) -> None:
        """Invia un record DTLS cifrato AES-128-GCM."""
        if self._aesgcm is None:
            raise DtlsProtocolError("chiavi non derivate: handshake non completato")
        # Nonce esplicito = epoch(2) + seq(6); nonce pieno = IV implicito(4) + esplicito(8).
        explicit_nonce = struct.pack("!H", self._epoch) + self._send_seq.to_bytes(6, "big")
        nonce = self._client_write_iv + explicit_nonce
        aad = explicit_nonce + struct.pack("!BHH", content_type, 0xFEFD, len(plaintext))
        ciphertext_and_tag = self._aesgcm.encrypt(nonce, plaintext, aad)
        fragment = explicit_nonce + ciphertext_and_tag
        header = struct.pack(
            "!BHH6sH",
            content_type,
            0xFEFD,
            self._epoch,
            self._send_seq.to_bytes(6, "big"),
            len(fragment),
        )
        self._sock.send(header + fragment)
        self._send_seq += 1

    def _recv(self) -> bytes:
        """Riceve un datagramma UDP dal bridge."""
        return self._sock.recv(4096)

    # --- Messaggi di handshake ---

    def _build_client_hello(self, cookie: bytes) -> bytes:
        """Costruisce il corpo del ClientHello (deliberatamente SENZA estensioni)."""
        body = bytearray()
        body.extend(_DTLS_VERSION)
        body.extend(self._client_random)
        body.append(0)  # session_id vuoto
        body.append(len(cookie))
        body.extend(cookie)
        body.extend(struct.pack("!H", 2))  # lunghezza lista cipher suite
        body.extend(_CIPHER_SUITE)
        body.append(1)  # lunghezza lista compressioni
        body.append(0)  # null compression
        return bytes(body)

    def _build_finished(self) -> bytes:
        """Costruisce il verify_data del Finished."""
        h = hashlib.sha256(self._handshake_messages).digest()
        return _prf(self._master_secret, b"client finished", h, 12)

    def _send_handshake(self, body: bytes, msg_type: int) -> None:
        """Incapsula un corpo handshake nel suo header e lo invia come record."""
        hs_header = struct.pack(
            "!B3sH3s3s",
            msg_type,
            len(body).to_bytes(3, "big"),
            self._msg_seq,
            (0).to_bytes(3, "big"),
            len(body).to_bytes(3, "big"),
        )
        full_msg = hs_header + body
        self._handshake_messages.extend(full_msg)
        self._msg_seq += 1
        self._send_record(_CT_HANDSHAKE, full_msg)

    def _send_encrypted_handshake(self, body: bytes) -> None:
        """Invia il Finished cifrato."""
        hs_header = struct.pack(
            "!B3sH3s3s",
            _HT_FINISHED,
            len(body).to_bytes(3, "big"),
            self._msg_seq,
            (0).to_bytes(3, "big"),
            len(body).to_bytes(3, "big"),
        )
        self._msg_seq += 1
        self._send_encrypted(_CT_HANDSHAKE, hs_header + body)

    # --- Parsing delle risposte ---

    def _parse_hello_verify_request(self, data: bytes) -> bytes:
        """Estrae il cookie dalla HelloVerifyRequest.

        Raises:
            DtlsProtocolError: Se il record non e' una HelloVerifyRequest.
        """
        # record(13) + handshake(12) + version(2), poi cookie_len(1) + cookie.
        offset = 13 + 12 + 2
        if len(data) <= offset or data[0] != _CT_HANDSHAKE or data[13] != _HT_HELLO_VERIFY_REQUEST:
            raise DtlsProtocolError(
                f"attesa HelloVerifyRequest, ricevuto record tipo "
                f"{data[0] if data else '?'} ({len(data)} B)"
            )
        cookie_len = data[offset]
        cookie = data[offset + 1 : offset + 1 + cookie_len]
        if len(cookie) != cookie_len:
            raise DtlsProtocolError("HelloVerifyRequest troncata (cookie incompleto)")
        return cookie

    def _parse_server_flight(self, data: bytes) -> None:
        """Estrae ServerHello (server_random) e HelloDone dalla flight del server.

        Possono arrivare come piu' record DTLS nello stesso datagramma, o lo
        HelloDone in un datagramma successivo (viene letto qui, best-effort).
        """
        got_server_hello = False
        got_server_hello_done = False
        offset = 0
        while offset + 13 <= len(data):
            record_len = int.from_bytes(data[offset + 11 : offset + 13], "big")
            record_payload = data[offset + 13 : offset + 13 + record_len]
            offset += 13 + record_len

            hs_offset = 0
            while hs_offset + 12 <= len(record_payload):
                hs_type = record_payload[hs_offset]
                hs_len = int.from_bytes(record_payload[hs_offset + 1 : hs_offset + 4], "big")
                hs_msg = record_payload[hs_offset : hs_offset + 12 + hs_len]
                if hs_type == _HT_SERVER_HELLO:
                    # server_random: dopo header handshake(12) + version(2).
                    self._server_random = hs_msg[14 : 14 + 32]
                    self._handshake_messages.extend(hs_msg)
                    got_server_hello = True
                elif hs_type == _HT_SERVER_HELLO_DONE:
                    self._handshake_messages.extend(hs_msg)
                    got_server_hello_done = True
                hs_offset += 12 + hs_len

        if got_server_hello and not got_server_hello_done:
            with suppress(TimeoutError):
                self._parse_server_flight(self._recv())

    # --- Derivazione chiavi ---

    def _derive_keys(self) -> None:
        """Deriva master secret e chiavi di cifratura dal PSK."""
        n = len(self._psk)
        # Pre-master secret PSK: len(N) + zeri(N) + len(N) + psk(N).
        pre_master = struct.pack("!H", n) + b"\x00" * n + struct.pack("!H", n) + self._psk
        self._master_secret = _prf(
            pre_master,
            b"master secret",
            self._client_random + self._server_random,
            48,
        )
        # Key block: client_key(16) + server_key(16) + client_iv(4) + server_iv(4);
        # del server non serve nulla (stream solo in uscita).
        key_block = _prf(
            self._master_secret,
            b"key expansion",
            self._server_random + self._client_random,
            40,
        )
        self._client_write_key = key_block[0:16]
        self._client_write_iv = key_block[32:36]
        self._aesgcm = AESGCM(self._client_write_key)


def _make_random() -> bytes:
    """Genera il random TLS da 32 byte (4 di timestamp + 28 casuali)."""
    return struct.pack("!I", int(time.time())) + os.urandom(28)


def _prf(secret: bytes, label: bytes, seed: bytes, length: int) -> bytes:
    """PRF TLS 1.2 con SHA-256."""
    result = b""
    a = hmac.new(secret, label + seed, hashlib.sha256).digest()
    while len(result) < length:
        result += hmac.new(secret, a + label + seed, hashlib.sha256).digest()
        a = hmac.new(secret, a, hashlib.sha256).digest()
    return result[:length]
