import pytest
from fastapi import Request
from pydantic import ValidationError

from api.auth_store import AuthSettings
from api.passkey_details import client_ip, provider_name


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "invalid",
        "00000000-0000-0000-0000-000000000000",
        "11111111-1111-1111-1111-111111111111",
    ],
)
def test_unknown_provider_is_not_guessed(value):
    assert provider_name(value) is None


def test_provider_lookup_normalizes_uuid_case():
    assert provider_name("EA9B8D66-4D01-1D21-3CE4-B6B48CB575D4") == "Google Password Manager"


@pytest.mark.parametrize(
    "peer,forwarded,expected",
    [
        ("203.0.113.9", "198.51.100.8", "203.0.113.9"),
        ("172.25.0.4", "198.51.100.8, 203.0.113.9", "203.0.113.9"),
        ("172.25.0.4", "198.51.100.8, 203.0.113.9, 10.252.25.198", "203.0.113.9"),
        ("172.25.0.4", "2001:db8::123, 10.252.25.198", "2001:db8::123"),
        ("172.25.0.4", "", "172.25.0.4"),
        ("172.25.0.4", "garbage, 10.252.25.198", None),
        ("172.25.0.4", "203.0.113.9," * 33, None),
        ("not-an-ip", "203.0.113.9", None),
        (None, "203.0.113.9", None),
    ],
)
def test_client_ip_uses_only_an_explicitly_trusted_proxy_chain(peer, forwarded, expected):
    request = Request(
        {
            "type": "http",
            "client": (peer, 1234) if peer else None,
            "headers": [(b"x-forwarded-for", forwarded.encode())],
        }
    )
    assert client_ip(request, ["172.25.0.0/16", "10.252.25.198/32"]) == expected


def test_proxy_config_is_validated_and_defaults_to_trusting_nobody():
    assert AuthSettings(_env_file=None, trusted_proxies=[]).trusted_proxies == []
    assert AuthSettings(_env_file=None, trusted_proxies=["10.252.25.198"]).trusted_proxies == [
        "10.252.25.198/32"
    ]
    with pytest.raises(ValidationError):
        AuthSettings(_env_file=None, trusted_proxies=["not-a-network"])
