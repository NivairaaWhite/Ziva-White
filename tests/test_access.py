"""Who may make the machine DO things over HTTP."""
import pytest

from zivawhite.access import control_allowed, has_valid_key, is_local_request, origin_ok

KEY = "s3cret"


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "127.0.5.9", "::ffff:127.0.0.1"])
def test_loopback_is_local(host):
    assert is_local_request(host, {})


@pytest.mark.parametrize("host", ["192.168.1.20", "10.0.0.5", "8.8.8.8", "testclient", "", None])
def test_everything_else_is_not(host):
    assert not is_local_request(host, {})


@pytest.mark.parametrize("hdr", ["X-Forwarded-For", "x-real-ip", "Forwarded", "X-Forwarded-Host"])
def test_a_reverse_proxy_makes_a_loopback_peer_untrusted(hdr):
    assert not is_local_request("127.0.0.1", {hdr: "203.0.113.9"})


def test_api_key_checks():
    assert has_valid_key({"Authorization": f"Bearer {KEY}"}, KEY)
    assert has_valid_key({"authorization": f"Bearer {KEY}"}, KEY)
    assert not has_valid_key({"Authorization": "Bearer wrong"}, KEY)
    assert not has_valid_key({"Authorization": KEY}, KEY)
    assert not has_valid_key({}, KEY)
    assert not has_valid_key({"Authorization": "Bearer "}, "")            # no key configured: a key can never "match"
    assert not has_valid_key({"Authorization": "Bearer "}, KEY)


def test_origin_rules():
    assert origin_ok({})                                                   # curl / CLI / native: no Origin
    assert origin_ok({"Origin": "http://127.0.0.1:8000", "Host": "127.0.0.1:8000"})    # the bundled web UI
    assert origin_ok({"Origin": "http://localhost:3000", "Host": "127.0.0.1:8000"})
    assert not origin_ok({"Origin": "https://evil.example", "Host": "127.0.0.1:8000"})
    assert not origin_ok({"Origin": "null", "Host": "127.0.0.1:8000"})
    assert not origin_ok({"Origin": "http://localhost.evil.example", "Host": "127.0.0.1:8000"})
    assert not origin_ok({"Origin": "http://127.0.0.1.evil.example:8000", "Host": "127.0.0.1:8000"})
    assert origin_ok({"Origin": "https://me.example", "Host": "x"}, extra_allowed="https://me.example, https://b.example")
    assert not origin_ok({"Origin": "https://evil.example", "Host": "x"}, extra_allowed="https://me.example")


def test_wildcard_cors_never_grants_the_ability_to_control_the_device():
    assert not origin_ok({"Origin": "https://evil.example", "Host": "127.0.0.1:8000"}, extra_allowed="*")


def test_control_needs_trust_AND_a_non_hostile_origin():
    ok = {"Origin": "http://127.0.0.1:8000", "Host": "127.0.0.1:8000"}
    assert control_allowed("127.0.0.1", ok, "")                                       # local, no key configured
    assert not control_allowed("192.168.1.5", ok, "")                                 # LAN, no key: refused
    assert control_allowed("192.168.1.5", {"Authorization": f"Bearer {KEY}"}, KEY)    # LAN with the key
    assert not control_allowed("192.168.1.5", {"Authorization": "Bearer nope"}, KEY)
    evil = {"Origin": "https://evil.example", "Host": "127.0.0.1:8000"}
    assert not control_allowed("127.0.0.1", evil, "")                                 # local peer, hostile web page
    assert not control_allowed("127.0.0.1", {**evil, "Authorization": f"Bearer {KEY}"}, KEY)
