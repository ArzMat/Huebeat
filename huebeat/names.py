"""Risoluzione dei nomi reali delle luci per canale via REST CLIP v2.

Il payload dell'entertainment configuration lega ogni canale ai suoi `members`
(service di tipo "entertainment"); ogni service entertainment ha un owner di
tipo "device", e il device porta `metadata.name` (il nome dato nell'app Hue).
Una luce gradient occupa piu' canali: quei canali condividono lo stesso nome.

Best-effort: qualsiasi errore (rete, formato) degrada a dict vuoto e il
chiamante mostra "Canale N". Nessuna interazione con lo streaming DTLS.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

# Stesse convenzioni di huebeat.bridge: TLS del Bridge con certificato self-signed.
_VERIFY_TLS = False
_TIMEOUT_SECONDS = 5.0
_AUTH_HEADER = "hue-application-key"

_ENTERTAINMENT_CONFIG_PATH = "/clip/v2/resource/entertainment_configuration"
_ENTERTAINMENT_PATH = "/clip/v2/resource/entertainment"
_DEVICE_PATH = "/clip/v2/resource/device"


def resolve_channel_names(host: str, username: str, config_id: str) -> dict[int, str]:
    """Mappa channel_id -> nome leggibile della luce che lo occupa.

    Args:
        host: IP/host del Bridge.
        username: Application key del Bridge (da .env).
        config_id: Id della entertainment configuration attiva.

    Returns:
        Dict channel_id -> nome del device; vuoto se qualcosa fallisce.
    """
    try:
        with httpx.Client(
            base_url=f"https://{host}",
            headers={_AUTH_HEADER: username},
            verify=_VERIFY_TLS,
            timeout=_TIMEOUT_SECONDS,
        ) as client:
            configs = _get_data(client, _ENTERTAINMENT_CONFIG_PATH)
            services = _get_data(client, _ENTERTAINMENT_PATH)
            devices = _get_data(client, _DEVICE_PATH)
    except (httpx.HTTPError, ValueError) as err:
        logger.warning("Risoluzione nomi luci fallita: %s", err)
        return {}

    # service entertainment id -> device id proprietario
    service_owner = {
        service.get("id"): service.get("owner", {}).get("rid")
        for service in services
        if service.get("owner", {}).get("rtype") == "device"
    }
    # device id -> nome
    device_name = {
        device.get("id"): device.get("metadata", {}).get("name", "") for device in devices
    }

    config = next((c for c in configs if c.get("id") == config_id), None)
    if config is None:
        logger.warning("Entertainment configuration %s non trovata sul Bridge.", config_id)
        return {}

    names: dict[int, str] = {}
    for channel in config.get("channels", []):
        channel_id = channel.get("channel_id")
        if channel_id is None:
            continue
        # Un canale puo' avere PIU' members (il bridge fonde segmenti di luci
        # gradient diverse): mostrarli tutti evita etichette fuorvianti che
        # cambiano con l'ordine dei members (visto sul campo con Signe+strip).
        member_names: list[str] = []
        for member in channel.get("members", []):
            rid = member.get("service", {}).get("rid")
            name = device_name.get(service_owner.get(rid), "")
            if name and name not in member_names:
                member_names.append(name)
        if member_names:
            names[int(channel_id)] = " + ".join(member_names)
    return names


def _get_data(client: httpx.Client, path: str) -> list[dict]:
    """GET CLIP v2 e ritorna il campo `data`.

    Args:
        client: Client httpx gia' configurato sul Bridge.
        path: Percorso della risorsa CLIP v2.

    Returns:
        La lista `data` della risposta.

    Raises:
        ValueError: Se lo status non e' 200 o il payload non ha `data`.
    """
    response = client.get(path)
    if response.status_code != 200:
        raise ValueError(f"GET {path} -> HTTP {response.status_code}")
    data = response.json().get("data")
    if not isinstance(data, list):
        raise ValueError(f"GET {path}: campo 'data' mancante")
    return data
