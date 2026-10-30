"""Tuning runtime di huebeat: i knob regolabili a caldo dal frontend.

Espone `live`, un'istanza module-level di `LiveTuning` letta dagli effetti e
dall'animator a OGNI frame. Il frontend (huebeat.control) scrive gli attributi
da un altro thread: ogni campo e' un float e l'assegnazione di un attributo e'
atomica in CPython, quindi il loop di rendering vede sempre il valore vecchio o
quello nuovo, mai uno stato incoerente. Nessun lock, per il vincolo del loop a
47 fps.

I default coincidono con le costanti storiche dei moduli (spectrum, adaptive,
palette): con config.local.toml assente il comportamento e' identico a prima.
`TUNING_SPECS` descrive ogni knob (etichetta, range, step, unita') ed e' usato
dal server di controllo per presentare i cursori alla UI.
"""

from dataclasses import dataclass, fields

# Divario di isteresi fisso tra soglia di ingresso e di uscita dalla calma
# (energia di sezione): off = on + questo valore.
CALM_HYSTERESIS_GAP = 0.10
# Divario di isteresi fisso tra soglia di ingresso e di uscita dalla raffica
# (eventi/s): off = on - questo valore.
BURST_HYSTERESIS_GAP = 2.0


@dataclass
class LiveTuning:
    """Knob regolabili a caldo, letti dal loop di rendering a ogni frame.

    Attributes:
        punch_gain: Contributo del punch al value HSV (spectrum).
        level_gain: Contributo del livello di banda al value HSV (spectrum).
        bass_contrast_gamma: Gamma di contrasto sul livello bassi (spectrum).
        punch_half_life_seconds: Tempo di dimezzamento del punch (spectrum).
        calm_energy_on: Soglia di energia sotto cui l'adaptive va in pulse.
        burst_density_on: Densita' di colpi (eventi/s) sopra cui va in waterfall.
        min_dwell_seconds: Permanenza minima in un regime dell'adaptive.
        drift_period_seconds: Periodo del drift della palette (fallback senza
            beat lock).
        crossfade_seconds: Durata del crossfade al cambio palette.
        hue_rotation_seconds_per_turn: Secondi per un giro completo della
            rotazione lenta dell'hue.
        drop_intensity: Forza della coreografia del drop (buio + slam);
            0 = coreografia disattivata.
        stereo_gain: Quanto il panning stereo sposta la luce sull'asse
            sinistra-destra; 0 = disattivato.
        beat_predict: Forza del flash predittivo sulla griglia del beat
            quando il tracker e' agganciato; 0 = disattivato.
        dynamics_gain: Quanto le sezioni quiete del brano vengono scurite in
            spectrum (macro-dinamica senza Spotify); 0 = disattivato.
        hue_offset: Rotazione manuale di tutti i colori sulla ruota, soccorso
            per palette sgradite; 0 = nessuna rotazione.
        gamma: Curva percettiva della luminosita' finale; 1.0 = lineare
            (identico a oggi).
        harmony_gain: Quanto gli stop della palette vengono piegati verso l'hue
            armonico del brano (circolo delle quinte); 0 = nessuna piega. Attivo
            solo in modalita' palette (con Spotify).
        mirror: Morph continuo della mappa di banda dei canali verso la simmetria
            a U (i due capi del percorso = bassi, il fondo = alti); 0 = mappa
            spaziale di oggi.
    """

    punch_gain: float = 0.6
    level_gain: float = 0.85
    bass_contrast_gamma: float = 2.0
    punch_half_life_seconds: float = 0.25
    calm_energy_on: float = 0.35
    burst_density_on: float = 7.0
    min_dwell_seconds: float = 2.0
    drift_period_seconds: float = 30.0
    crossfade_seconds: float = 2.0
    hue_rotation_seconds_per_turn: float = 720.0
    drop_intensity: float = 1.0
    stereo_gain: float = 0.35
    beat_predict: float = 0.5
    dynamics_gain: float = 0.0
    hue_offset: float = 0.0
    gamma: float = 1.0
    harmony_gain: float = 0.5
    mirror: float = 0.0

    @property
    def calm_energy_off(self) -> float:
        """Soglia di uscita dalla calma (isteresi fissa sopra quella d'ingresso).

        Returns:
            calm_energy_on + CALM_HYSTERESIS_GAP.
        """
        return self.calm_energy_on + CALM_HYSTERESIS_GAP

    @property
    def burst_density_off(self) -> float:
        """Soglia di uscita dalla raffica (isteresi fissa sotto quella d'ingresso).

        Returns:
            burst_density_on - BURST_HYSTERESIS_GAP, mai sotto zero.
        """
        return max(0.0, self.burst_density_on - BURST_HYSTERESIS_GAP)

    def as_dict(self) -> dict[str, float]:
        """Snapshot dei knob come dict nome -> valore.

        Returns:
            Dict con un'entry per ogni campo del dataclass.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def apply(self, values: dict[str, float]) -> None:
        """Applica un dict di knob, ignorando le chiavi sconosciute.

        Ogni valore viene riportato nel range dichiarato in TUNING_SPECS.
        L'assegnazione avviene campo per campo (atomica): sicura da chiamare
        da un thread diverso dal loop di rendering.

        Args:
            values: Dict nome knob -> nuovo valore.
        """
        for name, value in values.items():
            spec = _SPECS_BY_NAME.get(name)
            if spec is None:
                continue
            clamped = min(max(float(value), spec["min"]), spec["max"])
            setattr(self, name, clamped)


# Descrizione dei knob per la UI: nome campo, etichetta, valore di fabbrica
# (`default`, usato dal frontend per il reset), range, step, unita', gruppo di
# appartenenza (sezione visiva del pannello tuning) e `help` (spiegazione
# mostrata nel tooltip della I cerchiata accanto all'etichetta). Il `default`
# coincide sempre col default del campo omonimo di `LiveTuning`.
TUNING_SPECS: list[dict] = [
    {
        "name": "punch_gain",
        "label": "Punch",
        "default": 0.6,
        "min": 0.0,
        "max": 1.5,
        "step": 0.05,
        "unit": "",
        "group": "spectrum",
        "help": "Forza dei lampi sui colpi: piu' alto = flash piu' intensi su beat e kick.",
    },
    {
        "name": "level_gain",
        "label": "Livello",
        "default": 0.85,
        "min": 0.0,
        "max": 1.5,
        "step": 0.05,
        "unit": "",
        "group": "spectrum",
        "help": "Luminosita' di base che segue il volume della musica, sotto i lampi.",
    },
    {
        "name": "bass_contrast_gamma",
        "label": "Contrasto bassi",
        "default": 2.0,
        "min": 1.0,
        "max": 4.0,
        "step": 0.1,
        "unit": "",
        "group": "spectrum",
        "help": "Contrasto del canale bassi: piu' alto = i bassi deboli si spengono, "
        "restano solo i colpi pieni.",
    },
    {
        "name": "punch_half_life_seconds",
        "label": "Coda punch",
        "default": 0.25,
        "min": 0.08,
        "max": 0.6,
        "step": 0.01,
        "unit": "s",
        "group": "spectrum",
        "help": "Durata della coda del lampo dopo il colpo: corta = flash secchi, "
        "lunga = scie luminose.",
    },
    {
        "name": "calm_energy_on",
        "label": "Soglia calma",
        "default": 0.35,
        "min": 0.0,
        "max": 0.8,
        "step": 0.05,
        "unit": "",
        "group": "adaptive",
        "help": "Sotto questa energia l'effetto adaptive passa al regime calmo: "
        "tutte le luci respirano unite sul beat.",
    },
    {
        "name": "burst_density_on",
        "label": "Soglia raffica",
        "default": 7.0,
        "min": 3.0,
        "max": 14.0,
        "step": 0.5,
        "unit": "/s",
        "group": "adaptive",
        "help": "Colpi al secondo oltre i quali adaptive passa al waterfall: "
        "ogni colpo diventa una banda che scorre (drop veloci).",
    },
    {
        "name": "min_dwell_seconds",
        "label": "Permanenza regime",
        "default": 2.0,
        "min": 0.5,
        "max": 8.0,
        "step": 0.5,
        "unit": "s",
        "group": "adaptive",
        "help": "Tempo minimo in un regime di adaptive prima di poter cambiare: "
        "evita il ping-pong tra effetti.",
    },
    {
        "name": "drift_period_seconds",
        "label": "Drift palette",
        "default": 30.0,
        "min": 8.0,
        "max": 120.0,
        "step": 1.0,
        "unit": "s",
        "group": "palette",
        "help": "Scorrimento dei colori lungo le luci: secondi per un giro completo. "
        "Col BPM agganciato il drift va a tempo e questo valore fa da riserva.",
    },
    {
        "name": "crossfade_seconds",
        "label": "Crossfade brano",
        "default": 2.0,
        "min": 0.0,
        "max": 6.0,
        "step": 0.5,
        "unit": "s",
        "group": "palette",
        "help": "Durata della dissolvenza verso la palette del brano nuovo al cambio traccia.",
    },
    {
        "name": "hue_rotation_seconds_per_turn",
        "label": "Rotazione hue",
        "default": 720.0,
        "min": 120.0,
        "max": 1800.0,
        "step": 60.0,
        "unit": "s/giro",
        "group": "palette",
        "help": "Rotazione lenta di tutta la palette sulla ruota dei colori: secondi per "
        "giro completo. Riparte da zero al cambio brano.",
    },
    {
        "name": "drop_intensity",
        "label": "Drop show",
        "default": 1.0,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "unit": "",
        "group": "show",
        "help": "Coreografia del drop: un beat di buio totale e poi l'esplosione a "
        "palette piena quando il brano scoppia. 0 = disattivata.",
    },
    {
        "name": "stereo_gain",
        "label": "Stereo",
        "default": 0.35,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "unit": "",
        "group": "show",
        "help": "Quanto il panning stereo del brano sposta la luce tra sinistra e "
        "destra dell'impianto. 0 = disattivato.",
    },
    {
        "name": "beat_predict",
        "label": "Beat anticipato",
        "default": 0.5,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "unit": "",
        "group": "show",
        "help": "Flash aggiuntivo sulla griglia del beat PREVISTA quando il BPM e' "
        "agganciato: compensa il ritardo fisico delle lampade. 0 = disattivato.",
    },
    {
        "name": "dynamics_gain",
        "label": "Dinamica",
        "default": 0.0,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "unit": "",
        "group": "show",
        "help": "Scurisce le sezioni quiete del brano (strofe, breakdown) rispetto ai "
        "ritornelli, anche senza Spotify. 0 = disattivato (tutto alla stessa scala).",
    },
    {
        "name": "hue_offset",
        "label": "Tinta",
        "default": 0.0,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "unit": "",
        "group": "show",
        "help": "Ruota tutti i colori lungo la ruota: soccorso immediato quando la "
        "palette del brano non piace. 0 = colori originali.",
    },
    {
        "name": "gamma",
        "label": "Gamma",
        "default": 1.0,
        "min": 0.4,
        "max": 2.5,
        "step": 0.05,
        "unit": "",
        "group": "show",
        "help": "Curva percettiva della luminosita' delle luci: 1.0 = lineare come oggi, "
        "sotto 1 alza le penombre, sopra 1 le abbassa.",
    },
    {
        "name": "harmony_gain",
        "label": "Armonia",
        "default": 0.5,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "unit": "",
        "group": "show",
        "help": "Piega i colori della palette verso l'accordo del brano (circolo delle "
        "quinte): piu' alto = colori piu' intonati alla musica. Solo con Spotify. 0 = "
        "palette originale.",
    },
    {
        "name": "mirror",
        "label": "Simmetria",
        "default": 0.0,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "unit": "",
        "group": "show",
        "help": "Simmetria a U: i due capi del percorso vicino a te diventano i bassi e il "
        "fondo gli alti, invece della mappa sinistra-destra. 0 = mappa di oggi.",
    },
]

_SPECS_BY_NAME = {spec["name"]: spec for spec in TUNING_SPECS}

# Istanza condivisa: gli effetti la leggono a ogni frame, il server di
# controllo la scrive. Inizializzata coi default; l'entrypoint la puo'
# sovrascrivere coi valori persistiti (settings.tuning).
live = LiveTuning()
