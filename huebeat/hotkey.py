"""Hotkey globale di sistema per sospendere/riattivare le luci di huebeat.

Registra una combinazione di tasti globale (attiva anche quando huebeat non ha
il focus) che a ogni pressione inverte uno stato booleano attivo/disattivo e
notifica il chiamante tramite callback.

Nota di threading: la libreria di hotkey (keyboard) esegue il callback dal
PROPRIO thread listener, non dal thread che ha chiamato start(). Il callback
on_toggle viene quindi invocato da un thread diverso: il chiamante deve usare
un callback thread-safe, tipicamente il semplice set di un flag booleano o di
un threading.Event, senza toccare strutture non protette.

La dipendenza keyboard viene importata dentro start() (import locale) cosi'
questo modulo resta importabile anche su sistemi dove la libreria manca; in tal
caso start() logga un warning e ritorna False, e huebeat continua a funzionare
senza hotkey.
"""

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class HotkeyToggle:
    """Toggle globale attiva/disattiva via hotkey di sistema.

    A ogni pressione della combinazione registrata lo stato interno si inverte e
    viene chiamato on_toggle(nuovo_stato) dal thread del listener della libreria.

    Attributes:
        active: Stato corrente del toggle; parte a True.
    """

    def __init__(self, hotkey: str, on_toggle: Callable[[bool], None]) -> None:
        """Inizializza il toggle senza registrare ancora l'hotkey.

        Args:
            hotkey: Combinazione in sintassi della libreria keyboard, es.
                "ctrl+alt+shift+f9".
            on_toggle: Callback invocato dopo ogni inversione di stato con il
                nuovo valore booleano. Viene chiamato dal thread del listener
                della libreria: deve essere thread-safe.
        """
        self._hotkey = hotkey
        self._on_toggle = on_toggle
        # Stato corrente del toggle: True = luci attive, False = sospese.
        self._active = True
        # Handle restituito da keyboard.add_hotkey, serve per la rimozione.
        self._handle: object | None = None
        # Protegge inversione stato e lettura/scrittura di _handle tra il thread
        # del listener e il thread chiamante.
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        """Stato corrente del toggle (True = attivo).

        Returns:
            True se attivo, False se sospeso.
        """
        with self._lock:
            return self._active

    def set_active(self, active: bool) -> None:
        """Allinea lo stato interno a un cambiamento avvenuto ALTROVE.

        Usato quando la sospensione viene commutata da un altro canale (es. il
        frontend di controllo): la prossima pressione dell'hotkey inverte cosi'
        lo stato REALE delle luci, non uno stantio. Non invoca il callback.

        Args:
            active: Nuovo stato (True = luci attive).
        """
        with self._lock:
            self._active = active

    def _fire(self) -> None:
        """Inverte lo stato e notifica il chiamante (eseguito dal listener)."""
        with self._lock:
            self._active = not self._active
            nuovo_stato = self._active
        # Callback fuori dal lock: se il chiamante fa lavoro pesante o rientra,
        # non blocca ne' rischia deadlock sul nostro lock.
        self._on_toggle(nuovo_stato)

    def start(self) -> bool:
        """Registra l'hotkey globale.

        L'import della libreria keyboard e' locale: se manca, o se la
        registrazione fallisce, viene loggato un warning e la funzione ritorna
        False senza sollevare, cosi' huebeat prosegue senza hotkey.

        Returns:
            True se l'hotkey e' stata registrata, False se la libreria manca o
            la registrazione fallisce.
        """
        try:
            import keyboard  # import locale: modulo importabile anche senza libreria
        except Exception as exc:  # noqa: BLE001 - qualsiasi errore di import degrada a "no hotkey"
            logger.warning("Libreria 'keyboard' non disponibile, hotkey disattivata: %r", exc)
            return False

        try:
            handle = keyboard.add_hotkey(self._hotkey, self._fire)
        except Exception as exc:  # noqa: BLE001 - registrazione best-effort, huebeat deve girare comunque
            logger.warning("Registrazione hotkey '%s' fallita: %r", self._hotkey, exc)
            return False

        with self._lock:
            self._handle = handle
        logger.info("Hotkey '%s' registrata (toggle luci huebeat)", self._hotkey)
        return True

    def stop(self) -> None:
        """Rimuove l'hotkey registrata; idempotente.

        Se non c'e' nulla da rimuovere non fa niente. Eventuali errori nella
        rimozione vengono loggati ma non propagati.
        """
        with self._lock:
            handle = self._handle
            self._handle = None
        if handle is None:
            return
        try:
            import keyboard

            keyboard.remove_hotkey(handle)
        except Exception as exc:  # noqa: BLE001 - la rimozione e' best-effort
            logger.warning("Rimozione hotkey '%s' fallita: %r", self._hotkey, exc)
        else:
            logger.info("Hotkey '%s' rimossa", self._hotkey)
