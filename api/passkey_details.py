"""Display-only Passkey metadata; never use provider labels to authorize access."""

import json
from functools import lru_cache
from ipaddress import ip_address, ip_network
from pathlib import Path
from uuid import UUID


@lru_cache(maxsize=1)
def providers():
    # Pinned community registry recommended by https://web.dev/articles/webauthn-aaguid.
    # No third-party network request is made when binding or displaying a Passkey.
    return json.loads(Path(__file__).with_name("passkey_providers.json").read_text())["providers"]


def provider_name(aaguid):
    if not aaguid:
        return None
    try:
        identifier = UUID(aaguid)
    except (ValueError, TypeError, AttributeError):
        return None
    return providers().get(str(identifier)) if identifier.int else None


@lru_cache(maxsize=16)
def proxy_networks(cidrs):
    return tuple(ip_network(cidr) for cidr in cidrs)


def client_ip(request, trusted_proxies):
    """Walk X-Forwarded-For from the socket peer, stopping at the first untrusted hop."""
    try:
        address = ip_address(request.client.host) if request.client else None
    except ValueError:
        return None
    if address is None:
        return None
    networks = proxy_networks(tuple(trusted_proxies))

    def trusted(value):
        return any(value in network for network in networks)

    forwarded = request.headers.get("x-forwarded-for", "")
    if trusted(address) and forwarded:
        hops = forwarded.split(",")
        if len(forwarded) > 2048 or len(hops) > 32:
            return None
        for hop in reversed(hops):
            if not trusted(address):
                break
            try:
                address = ip_address(hop.strip())
            except ValueError:
                return None
    return str(address)
