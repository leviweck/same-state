"""xAI answer backend (plm_kit/answer_backends/xai.py). Mocked transport only: no test ever reaches the network."""
import io
import urllib.request

import pytest

from plm_kit import answer, config, generate
from plm_kit.answer_backends import xai

MSG = [{"role": "user", "content": "What is the guest network called?\n\n[1] (home-wifi.md)\nLanternGuest."}]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Block every urllib path out of the process: urlopen, any OpenerDirector (the xai transport builds its own
    proxy-free opener) and raw sockets to anything but loopback."""
    import socket

    def boom(*a, **k):
        raise AssertionError("a test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", boom)
    real_connect = socket.socket.connect

    def loopback_only(self, addr):
        host = addr[0] if isinstance(addr, tuple) else addr
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"a test tried to reach the network ({host})")
        return real_connect(self, addr)
    monkeypatch.setattr(socket.socket, "connect", loopback_only)


@pytest.fixture(autouse=True)
def fresh_run():
    """The per-run cost total is process-wide; give every test its own run."""
    xai.reset_run_spend()
    yield
    xai.reset_run_spend()


class Transport:
    def __init__(self, resp=None):
        self.calls = []
        self.resp = resp or {"status": "completed", "model": "grok-4.3",
                             "output": [{"type": "reasoning"},
                                        {"type": "message", "content": [{"type": "output_text", "text": "LanternGuest."}]}],
                             "usage": {"input_tokens": 40, "output_tokens": 30,
                                       "output_tokens_details": {"reasoning_tokens": 20}}}

    def __call__(self, url, payload, headers, timeout):
        self.calls.append({"url": url, "payload": payload, "headers": headers, "timeout": timeout})
        return self.resp


def gen_cfg(**xai_over):
    g = config.load_config()["generation"]
    return dict(g, backend="xai", allow_off_machine=True, xai=dict(g["xai"], **xai_over))


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key-not-real")
    xai.give_consent(tmp_path)
    return tmp_path


def make(cache, transport=None, **over):
    return xai.XaiBackend(gen_cfg(**over), cache=cache, post=transport or Transport())


def test_registered_off_by_default_and_needs_allow_off_machine():
    assert "xai" in answer.available()
    assert answer.discover()["xai"].sends_off_machine is True
    assert config.load_config()["generation"]["backend"] != "xai"
    with pytest.raises(ValueError, match="allow_off_machine"):
        answer.get_backend(dict(gen_cfg(), allow_off_machine=False))
    assert isinstance(answer.get_backend(gen_cfg()), xai.XaiBackend)


def test_request_shape_responses_api_store_false(ready):
    t = Transport()
    be = make(ready, t)
    out = be.generate(MSG, max_tokens=256, stop=["\n\nSources"])
    c = t.calls[0]
    assert c["url"] == "https://api.x.ai/v1/responses"
    assert c["headers"] == {"Authorization": "Bearer test-key-not-real"}
    p = c["payload"]
    assert p["store"] is False and p["model"] == "grok-4.3" and p["max_output_tokens"] == 256
    assert p["input"] == MSG and p["reasoning"] == {"effort": "low"} and "temperature" not in p
    assert out.text == "LanternGuest." and out.meta["usage"]["reasoning_tokens"] == 20
    # 40 in x $1.25 + (30 out + 20 reasoning, always added) x $2.50, per 1M
    assert out.meta["cost_usd"] == pytest.approx((40 * 1.25 + 50 * 2.50) / 1e6)
    assert out.meta["cost_source"] == "tokens"


def test_us_region_and_options(ready):
    t = Transport()
    be = make(ready, t, region="us", reasoning_effort=None, temperature=0.1, model="grok-4.7")
    be.generate(MSG, max_tokens=64)
    p = t.calls[0]["payload"]
    assert t.calls[0]["url"] == "https://us.api.x.ai/v1/responses"
    assert "reasoning" not in p and p["temperature"] == 0.1 and p["model"] == "grok-4.7"
    assert be.price == {"input": pytest.approx(2.2), "output": pytest.approx(6.6)}


def test_reasoning_reported_outside_output_is_still_billed(ready):
    t = Transport({"output_text": "ok", "usage": {"input_tokens": 10, "output_tokens": 5,
                                                   "output_tokens_details": {"reasoning_tokens": 100}}})
    out = make(ready, t).generate(MSG, max_tokens=200)
    assert out.meta["cost_usd"] == pytest.approx((10 * 1.25 + 105 * 2.50) / 1e6)


def test_cost_cap_refuses_before_sending(ready):
    t = Transport()
    be = make(ready, t, max_cost_usd=0.0005)
    with pytest.raises(RuntimeError, match="cost cap"):
        be.generate(MSG, max_tokens=512)       # worst case 512 x $2.50/1M = $0.00128 > cap
    assert t.calls == []                       # nothing was sent


def test_cost_cap_accumulates_over_the_run(ready):
    t = Transport({"output_text": "x", "usage": {"input_tokens": 100, "output_tokens": 300}})
    be = make(ready, t, max_cost_usd=0.002, reasoning_allowance_tokens=0)   # actual 100*1.25+300*2.5 = $0.000875
    be.generate(MSG, max_tokens=300)
    be.generate(MSG, max_tokens=300)
    with pytest.raises(RuntimeError, match="cost cap"):
        be.generate(MSG, max_tokens=300)
    assert len(t.calls) == 2 and be.spent_usd == pytest.approx(2 * 0.000875)


def test_missing_usage_counts_as_worst_case(ready):
    be = make(ready, Transport({"output_text": "x"}))
    out = be.generate(MSG, max_tokens=100)
    assert out.meta["cost_usd"] >= 100 * 2.50 / 1e6


def test_server_reported_cost_is_used_when_higher(ready):
    t = Transport({"output_text": "x", "usage": {"input_tokens": 1, "output_tokens": 1, "cost_in_usd_ticks": 5e7}})
    assert make(ready, t).generate(MSG, max_tokens=50).meta["cost_usd"] == pytest.approx(0.005)


def test_reasoning_is_always_added_even_when_smaller_than_output(ready):
    """Docs are ambiguous on whether output_tokens includes reasoning_tokens: err high and add it."""
    t = Transport({"output_text": "x", "usage": {"input_tokens": 100, "output_tokens": 400,
                                                  "output_tokens_details": {"reasoning_tokens": 150}}})
    out = make(ready, t).generate(MSG, max_tokens=500)
    assert out.meta["cost_usd"] == pytest.approx((100 * 1.25 + 550 * 2.50) / 1e6)


def test_ticks_are_preferred_over_token_math_even_when_lower(ready, monkeypatch):
    # cost-tracking docs: cost_in_usd_ticks is "the actual amount billed" (1 USD = 1e10 ticks)
    usage = {"input_tokens": 1000, "output_tokens": 400, "output_tokens_details": {"reasoning_tokens": 300},
             "cost_in_usd_ticks": 37_756_000}
    be = xai.XaiBackend(gen_cfg(), cache=ready, post=Transport({"output_text": "x", "usage": usage}),
                        today=Clock("2026-10-19"))
    out = be.generate(MSG, max_tokens=500)
    assert out.meta["cost_source"] == "cost_in_usd_ticks"
    assert out.meta["cost_usd"] == pytest.approx(0.0037756, abs=1e-6)             # meta is rounded to 6 dp
    assert xai.run_spent_usd() == pytest.approx(0.0037756) and be.ledger.totals()[0] == pytest.approx(0.0037756)


@pytest.mark.parametrize("ticks", [None, 0, -5, "123", True])
def test_missing_or_invalid_ticks_fall_back_to_tokens(ready, ticks):
    usage = {"input_tokens": 10, "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 5}}
    if ticks is not None:
        usage["cost_in_usd_ticks"] = ticks
    out = make(ready, Transport({"output_text": "x", "usage": usage})).generate(MSG, max_tokens=50)
    assert out.meta["cost_source"] == "tokens"
    assert out.meta["cost_usd"] == pytest.approx((10 * 1.25 + 25 * 2.50) / 1e6)


def test_needs_consent_and_never_sends_without_it(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key-not-real")
    t = Transport()
    be = xai.XaiBackend(gen_cfg(), cache=tmp_path, post=t, stdin=io.StringIO("yes\n"))   # not a TTY
    with pytest.raises(RuntimeError, match="consent"):
        be.generate(MSG, max_tokens=64)
    assert t.calls == [] and not xai.has_consent(tmp_path)


def test_interactive_consent_yes_and_no(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", "test-key-not-real")

    class Tty(io.StringIO):
        def isatty(self):
            return True

    t = Transport()
    with pytest.raises(RuntimeError, match="not allowed"):
        xai.XaiBackend(gen_cfg(), cache=tmp_path, post=t, stdin=Tty("no\n")).generate(MSG, max_tokens=64)
    assert t.calls == []
    xai.XaiBackend(gen_cfg(), cache=tmp_path, post=t, stdin=Tty("yes\n")).generate(MSG, max_tokens=64)
    assert len(t.calls) == 1 and xai.has_consent(tmp_path)
    assert "one-time consent" in capsys.readouterr().err


def test_key_only_from_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    xai.give_consent(tmp_path)
    t = Transport()
    with pytest.raises(ValueError, match="XAI_API_KEY"):
        xai.XaiBackend(gen_cfg(api_key="nope"), cache=tmp_path, post=t).generate(MSG, max_tokens=64)
    assert t.calls == []


@pytest.mark.parametrize("over, msg", [({"region": "eu"}, "region"), ({"reasoning_effort": "max"}, "reasoning"),
                                       ({"max_cost_usd": 0}, "max_cost_usd"), ({"model": "grok-9"}, "prices"),
                                       ({"chars_per_token": 20}, "chars_per_token")])
def test_bad_settings(over, msg):
    with pytest.raises(ValueError, match=msg):
        xai.settings(dict(config.load_config()["generation"]["xai"], **over))


class FakeOpener:
    def __init__(self, fn):
        self.fn = fn

    def open(self, req, timeout=None):
        return self.fn(req, timeout)


def test_http_errors_are_clean_and_hide_the_key(monkeypatch):
    import urllib.error

    def fake(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"error": "bad key"}'))
    monkeypatch.setattr(xai, "_opener", lambda: FakeOpener(fake))
    with pytest.raises(RuntimeError) as e:
        xai._post_json("https://api.x.ai/v1/responses", {}, {"Authorization": "Bearer secret-123"}, 5)
    assert "HTTP 401" in str(e.value) and "XAI_API_KEY" in str(e.value) and "secret-123" not in str(e.value)


def test_works_through_answer_local_and_voice_lite_guard(ready, monkeypatch):
    t = Transport()
    monkeypatch.setattr(xai, "_post_json", t)
    text, rep = generate.answer_local("guest network?", [("home-wifi.md", "LanternGuest")], gen_cfg(), cache=ready)
    assert text == "LanternGuest." and rep["backend"] == "xai" and t.calls[0]["payload"]["store"] is False


def test_direct_construction_also_needs_allow_off_machine(tmp_path):
    t = Transport()
    with pytest.raises(ValueError, match="allow_off_machine"):
        xai.XaiBackend(dict(gen_cfg(), allow_off_machine=False), cache=tmp_path, post=t)
    assert t.calls == []


def test_default_config_never_reaches_xai(monkeypatch):
    """With the shipped config nothing selects xai, so its transport is never touched."""
    def boom(*a, **k):
        raise AssertionError("network")
    monkeypatch.setattr(xai, "_post_json", boom)
    g = config.load_config()["generation"]
    assert g["backend"] != "xai" and g["allow_off_machine"] is False
    with pytest.raises(ValueError):
        answer.get_backend(dict(g, backend="xai"))


# --- region / model combos -------------------------------------------------------------------------------------
def test_us_region_rejects_models_it_does_not_serve():
    base = config.load_config()["generation"]["xai"]
    with pytest.raises(ValueError, match="isn't served on the 'us' regional endpoint"):
        xai.settings(dict(base, region="us"))                      # default grok-4.3 is global-only
    for m in ("grok-4.7", "grok-4.6"):
        assert xai.settings(dict(base, region="us", model=m))["model"] == m   # no silent model switch


def test_config_validation_catches_bad_region_combo(tmp_path):
    import yaml

    cfg = yaml.safe_load((config.DEFAULT_CONFIG).read_text())
    cfg["generation"]["xai"]["region"] = "us"
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="regional endpoint"):
        config.load_config(f)


# --- persistent daily / monthly ledger ---------------------------------------------------------------------------
class Clock:
    def __init__(self, day):
        self.day = day

    def __call__(self):
        return self.day


def test_ledger_reserves_then_records_real_cost_and_stores_no_text(ready):
    t = Transport({"output_text": "x", "usage": {"input_tokens": 100, "output_tokens": 300}})
    be = xai.XaiBackend(gen_cfg(), cache=ready, post=t, today=Clock("2026-10-19"))
    be.generate(MSG, max_tokens=300)
    raw = be.ledger.path.read_text()
    assert be.ledger.path == ready / "spend" / "xai-ledger.json"
    data = __import__("json").loads(raw)
    assert data == {"days": {"2026-10-19": pytest.approx(0.000875)}}
    assert "LanternGuest" not in raw and "test-key" not in raw


def test_daily_cap_persists_across_runs_and_resets_next_day(ready):
    clock = Clock("2026-10-19")
    t = Transport({"output_text": "x", "usage": {"input_tokens": 100, "output_tokens": 300}})
    for _ in range(2):                                          # two separate runs (processes) on the same day
        xai.XaiBackend(gen_cfg(max_cost_usd_per_day=0.002, reasoning_allowance_tokens=0), cache=ready, post=t, today=clock).generate(
            MSG, max_tokens=300)
    with pytest.raises(RuntimeError, match="daily cost cap"):
        xai.XaiBackend(gen_cfg(max_cost_usd_per_day=0.002, reasoning_allowance_tokens=0), cache=ready, post=t, today=clock).generate(
            MSG, max_tokens=300)
    assert len(t.calls) == 2                                    # refused before sending
    clock.day = "2026-10-20"
    xai.XaiBackend(gen_cfg(max_cost_usd_per_day=0.002, reasoning_allowance_tokens=0), cache=ready, post=t, today=clock).generate(MSG, max_tokens=300)
    assert len(t.calls) == 3


def test_monthly_cap(ready):
    clock = Clock("2026-10-01")
    t = Transport({"output_text": "x", "usage": {"input_tokens": 100, "output_tokens": 300}})
    cfg = gen_cfg(max_cost_usd_per_month=0.002, reasoning_allowance_tokens=0)
    for day in ("2026-10-01", "2026-10-02"):
        clock.day = day
        xai.XaiBackend(cfg, cache=ready, post=t, today=clock).generate(MSG, max_tokens=300)
    clock.day = "2026-10-03"
    with pytest.raises(RuntimeError, match="monthly cost cap"):
        xai.XaiBackend(cfg, cache=ready, post=t, today=clock).generate(MSG, max_tokens=300)
    clock.day = "2026-11-01"                                    # new month
    xai.XaiBackend(cfg, cache=ready, post=t, today=clock).generate(MSG, max_tokens=300)
    assert len(t.calls) == 3


def test_failed_request_keeps_its_worst_case_reservation(ready):
    def broken(*a, **k):
        raise RuntimeError("xAI API returned HTTP 500")
    be = xai.XaiBackend(gen_cfg(), cache=ready, post=broken, today=Clock("2026-10-19"))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        be.generate(MSG, max_tokens=400)
    day, _ = be.ledger.totals()
    assert day >= 400 * 2.50 / 1e6                              # never undercounts


def test_unreadable_ledger_blocks_sending(ready):
    led = xai.SpendLedger(ready)
    led.path.parent.mkdir(parents=True, exist_ok=True)
    led.path.write_text("{not json")
    t = Transport()
    with pytest.raises(RuntimeError, match="spend ledger"):
        make(ready, t).generate(MSG, max_tokens=64)
    assert t.calls == []


@pytest.mark.parametrize("over", [{"max_cost_usd_per_day": 0}, {"max_cost_usd_per_month": -1}])
def test_bad_ledger_settings(over):
    with pytest.raises(ValueError, match="max_cost_usd_per"):
        xai.settings(dict(config.load_config()["generation"]["xai"], **over))


# --- per-run cap is per process, not per backend instance ------------------------------------------------------------
def test_run_cap_spans_backend_instances_through_answer_local(ready, monkeypatch):
    """answer_local builds a new backend for every answer (as selftest --answer does ~8 times); the run cap
    must still hold across all of them, refusing before the send that could cross it, and never sending again."""
    t = Transport({"output_text": "ok", "usage": {"input_tokens": 500, "output_tokens": 15_000}})
    monkeypatch.setattr(xai, "_post_json", t)
    cfg = dict(gen_cfg(max_cost_usd=0.20, max_cost_usd_per_day=100.0, reasoning_allowance_tokens=0),
               max_new_tokens=20_000)
    per_answer = (500 * 1.25 + 15_000 * 2.50) / 1e6                  # $0.038125 actual each

    def ask():
        return generate.answer_local("guest network?", [("home-wifi.md", "LanternGuest")], cfg, cache=ready)

    sent = 0
    with pytest.raises(RuntimeError, match="cost cap"):
        for _ in range(20):
            ask()
            sent += 1
    assert 1 < sent < 20 and len(t.calls) == sent
    assert xai.run_spent_usd() == pytest.approx(sent * per_answer)
    assert xai.run_spent_usd() <= 0.20                                # never crossed the cap
    assert xai.run_spent_usd() + 20_000 * 2.50 / 1e6 > 0.20          # refused because the worst case would cross it
    for _ in range(3):                                                # and nothing more goes out in this run
        with pytest.raises(RuntimeError, match="cost cap"):
            ask()
    assert len(t.calls) == sent
    xai.reset_run_spend()                                             # a new run starts from zero
    ask()
    assert len(t.calls) == sent + 1


def test_run_total_is_shared_and_reported(ready):
    t = Transport({"output_text": "x", "usage": {"input_tokens": 100, "output_tokens": 300}})
    a, b = make(ready, t), make(ready, t)
    a.generate(MSG, max_tokens=300)
    out = b.generate(MSG, max_tokens=300)
    assert b.spent_usd == pytest.approx(0.000875)                    # this instance
    assert out.meta["run_spent_usd"] == pytest.approx(2 * 0.000875)  # the whole run


# --- transport: no proxy, clean errors ---------------------------------------------------------------------------
def test_transport_ignores_proxy_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    op = xai._opener()
    # an empty ProxyHandler replaces the default env-reading one and registers no proxy_open handlers at all
    assert not [h for h in op.handlers if isinstance(h, urllib.request.ProxyHandler)]
    assert not any(getattr(h, "proxies", None) for h in op.handlers)
    default = [h for h in urllib.request.build_opener().handlers if isinstance(h, urllib.request.ProxyHandler)]
    assert default and default[0].proxies.get("https") == "http://127.0.0.1:9"   # what the default would have used


def test_transport_connects_directly_not_via_proxy(monkeypatch):
    """Behavioral: with HTTP(S)_PROXY pointing at a local listener, the request still goes straight to the target
    (both are loopback; nothing leaves the machine)."""
    import http.server
    import socket
    import threading

    monkeypatch.undo()                                            # drop no_network for this loopback-only test
    real_connect = socket.socket.connect

    def loopback_only(self, addr):
        assert addr[0] in ("127.0.0.1", "::1"), addr
        return real_connect(self, addr)
    monkeypatch.setattr(socket.socket, "connect", loopback_only)

    hits = {"target": 0, "proxy": 0}

    def handler(kind, body):
        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                hits[kind] += 1
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        return H

    target = http.server.HTTPServer(("127.0.0.1", 0), handler("target", b'{"output_text": "ok"}'))
    proxy = http.server.HTTPServer(("127.0.0.1", 0), handler("proxy", b'{"output_text": "via proxy"}'))
    for srv in (target, proxy):
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        purl = f"http://127.0.0.1:{proxy.server_port}"
        for v in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy"):
            monkeypatch.setenv(v, purl)
        for v in ("NO_PROXY", "no_proxy"):
            monkeypatch.setenv(v, "")
        out = xai._post_json(f"http://127.0.0.1:{target.server_port}/v1/responses", {"x": 1},
                             {"Authorization": "Bearer secret-123"}, 5)
        assert out == {"output_text": "ok"} and hits == {"target": 1, "proxy": 0}
    finally:
        target.shutdown()
        proxy.shutdown()


@pytest.mark.parametrize("exc, expect", [
    (ConnectionResetError(104, "Connection reset by peer"), "ConnectionResetError"),
    (BrokenPipeError(32, "Broken pipe"), "BrokenPipeError"),
    (OSError("weird"), "OSError"),
    (TimeoutError("timed out"), "could not reach"),
])
def test_transport_os_errors_become_runtime_errors(monkeypatch, exc, expect):
    def fake(req, timeout):
        raise exc
    monkeypatch.setattr(xai, "_opener", lambda: FakeOpener(fake))
    with pytest.raises(RuntimeError) as e:
        xai._post_json("https://api.x.ai/v1/responses", {}, {"Authorization": "Bearer secret-123"}, 5)
    assert expect in str(e.value) and "secret-123" not in str(e.value)


class FakeResp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.mark.parametrize("body, expect", [(b"<html>gateway</html>", "isn't JSON"), (b"\xff\xfe", "isn't JSON"),
                                          (b"[1, 2]", "unexpected JSON")])
def test_transport_non_json_200_is_a_runtime_error(monkeypatch, body, expect):
    monkeypatch.setattr(xai, "_opener", lambda: FakeOpener(lambda req, timeout: FakeResp(body)))
    with pytest.raises(RuntimeError, match=expect) as e:
        xai._post_json("https://api.x.ai/v1/responses", {}, {"Authorization": "Bearer secret-123"}, 5)
    assert "secret-123" not in str(e.value) and "gateway" not in str(e.value)


def test_transport_reset_inside_backend_keeps_reservation_and_hides_key(ready, monkeypatch):
    def fake(req, timeout):
        raise ConnectionResetError(104, "Connection reset by peer")
    monkeypatch.setattr(xai, "_opener", lambda: FakeOpener(fake))
    monkeypatch.setenv("XAI_API_KEY", "secret-456")
    be = xai.XaiBackend(gen_cfg(), cache=ready, today=Clock("2026-10-19"))
    with pytest.raises(RuntimeError) as e:
        be.generate(MSG, max_tokens=100)
    assert "secret-456" not in str(e.value) and be.ledger.totals()[0] >= 100 * 2.50 / 1e6


# --- ledger: locked read-modify-write, atomic file -----------------------------------------------------------------
def test_ledger_concurrent_processes_lose_no_updates(tmp_path):
    import subprocess
    import sys

    pytest.importorskip("fcntl")
    code = ("import sys; from plm_kit.answer_backends.xai import SpendLedger\n"
            "led = SpendLedger(sys.argv[1], today=lambda: '2026-10-19')\n"
            "for _ in range(40): led.add(0.01)\n")
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    env = dict(__import__("os").environ, PYTHONPATH=str(root))
    procs = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], env=env) for _ in range(4)]
    assert all(p.wait(timeout=60) == 0 for p in procs)
    led = xai.SpendLedger(tmp_path, today=lambda: "2026-10-19")
    assert led.totals()[0] == pytest.approx(4 * 40 * 0.01)
    assert sorted(f.name for f in led.path.parent.iterdir()) == ["xai-ledger.json", "xai-ledger.json.lock"]


def test_ledger_reserve_is_check_and_add_in_one_step(tmp_path):
    led = xai.SpendLedger(tmp_path, today=lambda: "2026-10-19")
    led.reserve(0.6, 1.0, None)
    with pytest.raises(RuntimeError, match="daily cost cap"):
        led.reserve(0.6, 1.0, None)
    assert led.totals()[0] == pytest.approx(0.6)                 # the refused reservation added nothing


def test_ledger_write_is_atomic(tmp_path, monkeypatch):
    led = xai.SpendLedger(tmp_path, today=lambda: "2026-10-19")
    led.add(0.5)
    before = led.path.read_text()

    def fail(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(xai.os, "replace", fail)
    with pytest.raises(OSError):
        led.add(0.25)
    assert led.path.read_text() == before                        # old ledger intact
    assert [f.name for f in led.path.parent.iterdir() if f.suffix == ".tmp"] == []   # temp file cleaned up


def test_run_reservation_released_when_ledger_refuses(ready):
    t = Transport()
    be = xai.XaiBackend(gen_cfg(max_cost_usd_per_day=0.0001), cache=ready, post=t)
    with pytest.raises(RuntimeError, match="daily cost cap"):
        be.generate(MSG, max_tokens=64)
    assert t.calls == [] and xai.run_spent_usd() == 0.0


# --- reasoning allowance (xAI bills reasoning on top of max_output_tokens) -----------------------------------------
def test_estimate_includes_reasoning_allowance_at_output_rate(ready):
    t = Transport()
    be = make(ready, t, reasoning_allowance_tokens=1000)
    est_in = be.estimate_input_tokens(MSG)
    out = be.generate(MSG, max_tokens=200)
    assert out.meta["estimate_usd"] == pytest.approx((est_in * 1.25 + (200 + 1000) * 2.50) / 1e6, abs=1e-6)
    assert t.calls[0]["payload"]["max_output_tokens"] == 200     # the allowance is never sent as a limit


def test_allowance_makes_the_cap_refuse_before_sending(ready):
    t = Transport()
    # visible answer alone: 200 x $2.50/1M = $0.0005 < cap; with a 4096 allowance: ~ $0.0107 > cap
    with pytest.raises(RuntimeError, match="cost cap"):
        make(ready, t, max_cost_usd=0.005, reasoning_allowance_tokens=4096).generate(MSG, max_tokens=200)
    assert t.calls == []
    make(ready, t, max_cost_usd=0.005, reasoning_allowance_tokens=0).generate(MSG, max_tokens=200)
    assert len(t.calls) == 1


def test_default_allowance_is_documented_and_validated():
    base = config.load_config()["generation"]["xai"]
    assert xai.settings(base)["reasoning_allowance_tokens"] == 4096
    with pytest.raises(ValueError, match="reasoning_allowance_tokens"):
        xai.settings(dict(base, reasoning_allowance_tokens=-1))


def test_overshoot_records_the_real_cost_everywhere(ready, capsys):
    """Reasoning beyond the allowance: the real (higher) cost goes to the run total and the ledger, and is flagged."""
    big = {"output_text": "x", "usage": {"input_tokens": 100, "output_tokens": 50,
                                         "output_tokens_details": {"reasoning_tokens": 5000}}}
    t = Transport(big)
    be = xai.XaiBackend(gen_cfg(reasoning_allowance_tokens=100), cache=ready, post=t, today=Clock("2026-10-19"))
    out = be.generate(MSG, max_tokens=64)
    real = (100 * 1.25 + 5050 * 2.50) / 1e6                       # reasoning reported outside output_tokens
    assert out.meta["cost_usd"] == pytest.approx(real, abs=1e-6)
    assert out.meta["overshoot"] is True and out.meta["cost_usd"] > out.meta["estimate_usd"]
    assert out.meta["overshoot_usd"] == pytest.approx(real - out.meta["estimate_usd"], abs=1e-6)
    assert xai.run_spent_usd() == pytest.approx(real)              # run total = real cost, not the estimate
    assert be.ledger.totals()[0] == pytest.approx(real)            # ledger = real cost
    assert be.ledger.overshoots() == 1
    assert "more than the pre-send estimate" in capsys.readouterr().err
    data = __import__("json").loads(be.ledger.path.read_text())
    assert data["overshoots"] == {"2026-10-19": 1} and set(data) == {"days", "overshoots"}


def test_overshoot_counts_toward_the_next_cap_check(ready):
    big = {"output_text": "x", "usage": {"input_tokens": 100, "output_tokens": 50,
                                         "output_tokens_details": {"reasoning_tokens": 20_000}}}
    t = Transport(big)                                              # each answer really costs ~ $0.050
    make(ready, t, max_cost_usd=0.06, reasoning_allowance_tokens=100).generate(MSG, max_tokens=64)
    with pytest.raises(RuntimeError, match="cost cap"):             # 0.050 real + small estimate > 0.06
        make(ready, t, max_cost_usd=0.06, reasoning_allowance_tokens=4096).generate(MSG, max_tokens=64)
    assert len(t.calls) == 1


def test_no_overshoot_flag_when_within_estimate(ready):
    out = make(ready, Transport()).generate(MSG, max_tokens=256)
    assert out.meta["overshoot"] is False and out.meta["overshoot_usd"] == 0


# --- ledger lock: Windows fallback and fail-closed ------------------------------------------------------------------
class FakeMsvcrt:
    LK_LOCK, LK_UNLCK = 1, 0

    def __init__(self, fail_times=0):
        self.calls, self.fail_times = [], fail_times

    def locking(self, fd, mode, nbytes):
        self.calls.append((mode, nbytes))
        if mode == self.LK_LOCK and self.fail_times:
            self.fail_times -= 1
            raise OSError(36, "Resource deadlock avoided")


def test_ledger_uses_msvcrt_when_fcntl_is_missing(ready, monkeypatch):
    fake = FakeMsvcrt(fail_times=1)                               # first attempt busy, second succeeds
    monkeypatch.setattr(xai, "_lock_impl", lambda: ("msvcrt", fake))
    t = Transport()
    be = xai.XaiBackend(gen_cfg(), cache=ready, post=t, today=Clock("2026-10-19"))
    be.generate(MSG, max_tokens=64)
    assert len(t.calls) == 1 and be.ledger.totals()[0] > 0
    locks = [c for c in fake.calls if c[0] == FakeMsvcrt.LK_LOCK]
    unlocks = [c for c in fake.calls if c[0] == FakeMsvcrt.LK_UNLCK]
    assert len(locks) == 1 + 2 * 1 and len(unlocks) == 2       # reserve (busy once, then ok) + correction
    assert be.ledger.lock_path.read_bytes() == b""               # the lock file holds nothing


def test_ledger_fails_closed_without_any_lock(ready, monkeypatch):
    monkeypatch.setattr(xai, "_lock_impl", lambda: (None, None))
    t = Transport()
    be = xai.XaiBackend(gen_cfg(), cache=ready, post=t, today=Clock("2026-10-19"))
    with pytest.raises(RuntimeError, match="can't lock the xAI spend ledger"):
        be.generate(MSG, max_tokens=64)
    assert t.calls == [] and xai.run_spent_usd() == 0.0 and not be.ledger.path.exists()


def test_msvcrt_lock_timeout_fails_closed(ready, monkeypatch):
    monkeypatch.setattr(xai, "_lock_impl", lambda: ("msvcrt", FakeMsvcrt(fail_times=10**6)))
    monkeypatch.setattr(xai.SpendLedger, "LOCK_TIMEOUT_S", 0)
    t = Transport()
    with pytest.raises(RuntimeError, match="timed out locking"):
        xai.XaiBackend(gen_cfg(), cache=ready, post=t).generate(MSG, max_tokens=64)
    assert t.calls == []


def test_posix_lock_times_out_and_fails_closed_when_held_elsewhere(ready, monkeypatch):
    """Another process holds the ledger lock: the deadline loop gives up and nothing is sent."""
    import subprocess
    import sys

    pytest.importorskip("fcntl")
    led = xai.SpendLedger(ready)
    led.path.parent.mkdir(parents=True, exist_ok=True)
    code = ("import fcntl, sys, time\n"
            "f = open(sys.argv[1], 'a+b'); fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
            "print('held', flush=True); time.sleep(30)\n")
    holder = subprocess.Popen([sys.executable, "-c", code, str(led.lock_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        monkeypatch.setattr(xai.SpendLedger, "LOCK_TIMEOUT_S", 0.3)
        t = Transport()
        with pytest.raises(RuntimeError, match="timed out locking the xAI spend ledger.*nothing was sent"):
            xai.XaiBackend(gen_cfg(), cache=ready, post=t).generate(MSG, max_tokens=64)
        assert t.calls == [] and xai.run_spent_usd() == 0.0
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_post_response_lock_failure_keeps_the_reserved_estimate(ready, monkeypatch):
    """If the correction after the response can't lock, the user is told it was already billed and the ledger keeps
    the (higher) reserved estimate; the run total holds the real cost."""
    real_impl, calls = xai._lock_impl, {"n": 0}

    def flaky():
        calls["n"] += 1
        return real_impl() if calls["n"] == 1 else (None, None)   # reserve OK, correction fails
    monkeypatch.setattr(xai, "_lock_impl", flaky)
    t = Transport({"output_text": "x", "usage": {"input_tokens": 100, "output_tokens": 300}})
    be = xai.XaiBackend(gen_cfg(), cache=ready, post=t, today=Clock("2026-10-19"))
    estimate = be.cost(be.estimate_input_tokens(MSG), 300 + 4096)
    with pytest.raises(RuntimeError, match="already sent and billed by xAI; the ledger keeps its reserved estimate"):
        be.generate(MSG, max_tokens=300)
    assert len(t.calls) == 1
    monkeypatch.setattr(xai, "_lock_impl", real_impl)
    real = (100 * 1.25 + 300 * 2.50) / 1e6
    assert be.ledger.totals()[0] == pytest.approx(estimate) and estimate > real    # errs high
    assert xai.run_spent_usd() == pytest.approx(real)
