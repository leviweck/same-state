"""Regression tests for the suite-wide network guard (tests/conftest.py).

Background (2026-10-19): a guard that only replaced urllib's urlopen missed a transport that
builds its own opener, and one test request reached a real service. These tests prove the socket-level guard stops
every path loudly, without ever reaching the network (the guard raises before any packet is sent)."""
import socket
import threading

import pytest
from conftest import NetworkBlockedError

from plm_kit.answer_backends import xai


@pytest.mark.parametrize("addr", [("203.0.113.10", 443),          # TEST-NET-3 (documentation range), never routed
                                  ("2001:db8::1", 443),            # IPv6 documentation range
                                  ("example.com", 443)])            # a name: blocked at DNS, before any lookup
def test_create_connection_to_public_address_fails_loudly(addr):
    with pytest.raises(NetworkBlockedError, match="network blocked in tests"):
        socket.create_connection(addr, timeout=1)


def test_raw_connect_and_connect_ex_are_blocked():
    for fn in ("connect", "connect_ex"):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(NetworkBlockedError):
                getattr(s, fn)(("203.0.113.10", 80))
        finally:
            s.close()


def test_dns_lookup_of_public_name_is_blocked():
    with pytest.raises(NetworkBlockedError, match="DNS lookup"):
        socket.getaddrinfo("api.x.ai", 443)


def test_loopback_still_works():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    t = threading.Thread(target=lambda: srv.accept()[0].close(), daemon=True)
    t.start()
    try:
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
    finally:
        t.join(timeout=2)
        srv.close()


def test_guard_survives_a_tests_own_monkeypatch_undo(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "x")
    monkeypatch.undo()
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("203.0.113.10", 443), timeout=1)


@pytest.mark.parametrize("proxy", [None, "http://203.0.113.20:3128"])
def test_xai_no_proxy_transport_cannot_reach_the_network(monkeypatch, proxy):
    """The proxy-free opener (build_opener(ProxyHandler({}))) pointed at the real endpoint gets the guard error,
    never an HTTP response. The key is fake and the payload empty; nothing leaves the machine."""
    for v in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY"):
        if proxy:
            monkeypatch.setenv(v, proxy)
        else:
            monkeypatch.delenv(v, raising=False)
    with pytest.raises(NetworkBlockedError, match="network blocked in tests"):
        xai._post_json("https://api.x.ai/v1/responses", {}, {"Authorization": "Bearer not-a-real-key"}, 5)


# --- DNS helpers, UDP, child processes ----------------------------------------------------------------------------
@pytest.mark.parametrize("fn", ["gethostbyname", "gethostbyname_ex"])
def test_legacy_dns_lookups_are_blocked(fn):
    with pytest.raises(NetworkBlockedError, match="DNS lookup"):
        getattr(socket, fn)("api.x.ai")
    assert getattr(socket, fn)("localhost")                     # local names still resolve


def test_udp_sendto_non_loopback_is_blocked_both_signatures():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(NetworkBlockedError, match="sendto"):
            s.sendto(b"x", ("203.0.113.10", 53))
        with pytest.raises(NetworkBlockedError, match="sendto"):
            s.sendto(b"x", 0, ("203.0.113.10", 53))
    finally:
        s.close()


def test_udp_sendmsg_non_loopback_is_blocked():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(NetworkBlockedError, match="sendmsg"):
            s.sendmsg([b"x"], [], 0, ("203.0.113.10", 53))
    finally:
        s.close()


def test_udp_to_loopback_still_works():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tx.sendto(b"ping", rx.getsockname())
        tx.sendmsg([b"pong"], [], 0, rx.getsockname())
        assert rx.recv(16) == b"ping" and rx.recv(16) == b"pong"
    finally:
        tx.close()
        rx.close()


def test_child_python_processes_are_guarded_too():
    """conftest exports PLM_KIT_TEST_NETGUARD=1 and tests/netguard on PYTHONPATH; a child's sitecustomize installs
    the same guard, so a subprocess can't reach the network either (and still starts normally)."""
    import subprocess
    import sys

    code = ("import socket\n"
            "try:\n"
            "    socket.create_connection(('203.0.113.10', 443), timeout=1)\n"
            "except AssertionError as e:\n"
            "    print('BLOCKED', e)\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit('child was not guarded')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False)
    assert r.returncode == 0, r.stderr
    assert "BLOCKED network blocked in tests" in r.stdout
    r = subprocess.run([sys.executable, "-c", "import socket; print(socket.gethostbyname('localhost'))"],
                       capture_output=True, text=True, timeout=30, check=False)
    assert r.returncode == 0 and r.stdout.strip().startswith("127.")
