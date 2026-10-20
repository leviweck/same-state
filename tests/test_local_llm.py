"""No-GPU backends (Ollama, llama.cpp server, llama-cpp-python), mocked: no network, no model files."""
import json
import re

import pytest

from plm_kit import answer, cli, generate, local_llm, models
from plm_kit.answer_backends.llamacpp import LlamaCppPythonBackend, LlamaCppServerBackend
from plm_kit.answer_backends.ollama import OllamaBackend


def g(local=None, **backends):
    return {"local": local or {}, **backends}


def test_registry_is_pinned_permissive_and_has_a_default():
    assert models.DEFAULT_LOCAL_GENERATOR in models.LOCAL_GENERATORS
    for name, spec in models.LOCAL_GENERATORS.items():
        models.local_generator(name)                        # 40-char revision + permissive license
        assert spec["license"] in ("apache-2.0", "mit")
        assert spec["file"].endswith(".gguf") and len(spec["sha256"]) == 64 and spec["ollama"]
    assert models.LOCAL_GENERATORS[models.DEFAULT_LOCAL_GENERATOR]["ram_gb"] <= 8   # fits an 8-16 GB machine
    with pytest.raises(ValueError):
        models.local_generator("nope")


def test_loopback_only_by_default():
    assert local_llm.check_url("http://127.0.0.1:11434/") == ("http://127.0.0.1:11434", False)
    assert local_llm.check_url("http://localhost:8080") == ("http://localhost:8080", False)
    assert local_llm.check_url("http://[::1]:8080")[1] is False
    for bad in ("http://192.168.1.20:11434", "https://example.com", "http://10.0.0.5:8080"):
        with pytest.raises(ValueError, match="refusing"):
            local_llm.check_url(bad)
        with pytest.raises(ValueError, match="BOTH"):                     # one flag alone is not enough
            local_llm.check_url(bad, allow_remote=True)
        with pytest.raises(ValueError, match="BOTH"):
            local_llm.check_url(bad, allow_off_machine=True)
    assert local_llm.check_url("http://192.168.1.20:11434", allow_remote=True, allow_off_machine=True) == \
        ("http://192.168.1.20:11434", True)
    with pytest.raises(ValueError):
        local_llm.check_url("file:///etc/passwd")


def test_localhost_must_resolve_to_loopback(monkeypatch):
    """A remapped /etc/hosts entry (localhost -> a LAN address) is treated as remote and refused."""
    real = local_llm.socket.getaddrinfo
    monkeypatch.setattr(local_llm.socket, "getaddrinfo",
                        lambda host, *a, **k: [(2, 1, 6, "", ("192.168.1.50", 0))] if host == "localhost"
                        else real(host, *a, **k))
    with pytest.raises(ValueError, match="refusing"):
        local_llm.check_url("http://localhost:11434")
    monkeypatch.setattr(local_llm.socket, "getaddrinfo",
                        lambda host, *a, **k: [(2, 1, 6, "", ("127.0.0.1", 0)), (2, 1, 6, "", ("10.1.1.1", 0))])
    with pytest.raises(ValueError, match="refusing"):                     # every address must be loopback
        local_llm.check_url("http://localhost:11434")


@pytest.mark.parametrize("cls,key,field", [(OllamaBackend, "ollama", "host"),
                                           (LlamaCppServerBackend, "llamacpp_server", "url")])
def test_remote_server_needs_both_flags_and_counts_as_off_machine(cls, key, field):
    remote = {key: {field: "http://192.168.1.20:8080"}}
    for cfg in (g(**remote), g({"allow_remote": True}, **remote), dict(g(**remote), allow_off_machine=True)):
        with pytest.raises(ValueError, match="refusing"):
            cls(cfg, post=lambda *a: {})
    be = cls(dict(g({"allow_remote": True}, **remote), allow_off_machine=True), post=lambda *a: {})
    assert be.sends_off_machine is True
    assert cls(g(), post=lambda *a: {}).sends_off_machine is False
    with pytest.raises(ValueError, match="refusing"):                     # same rule through the registry
        answer.get_backend(dict(g({"allow_remote": True}, **remote), backend=key))


@pytest.fixture
def proxy_listener(monkeypatch):
    """A local 'proxy' that records any connection; every proxy env var points at it."""
    import socket
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    srv.settimeout(0.2)
    seen, stop = [], threading.Event()

    def run():
        while not stop.is_set():
            try:
                c, _ = srv.accept()
            except OSError:
                continue
            c.settimeout(2)
            data = b""
            try:                                  # read the whole request: headers, then Content-Length body bytes
                while True:
                    chunk = c.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                    head, sep, body = data.partition(b"\r\n\r\n")
                    if sep:
                        m = re.search(rb"(?im)^content-length:\s*(\d+)", head)
                        if len(body) >= (int(m.group(1)) if m else 0):
                            break
            except OSError:
                pass
            seen.append(data)
            c.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{srv.getsockname()[1]}"
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(var, url)
    for var in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(var, raising=False)
    yield seen
    stop.set()
    t.join(2)
    srv.close()


@pytest.fixture
def model_server():
    """A tiny local 'model server' answering any POST with an OpenAI-style reply."""
    import http.server
    import threading

    got = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            got.append(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({"choices": [{"message": {"content": "ok"}}],
                               "message": {"content": "ok"}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", got
    srv.shutdown()
    srv.server_close()


def test_proxy_env_vars_never_see_notes(proxy_listener, model_server):
    url, got = model_server
    secret = "NOTE-TEXT-7f3a voice example line"
    msgs = [{"role": "user", "content": secret}]
    assert LlamaCppServerBackend(g(llamacpp_server={"url": url})).generate(msgs, max_tokens=5).text == "ok"
    assert OllamaBackend(g(ollama={"host": url, "model": "m"})).generate(msgs, max_tokens=5).text == "ok"
    assert len(got) == 2 and all(secret.encode() in b for b in got)        # went straight to the server
    import time
    time.sleep(0.3)
    assert proxy_listener == []                                           # the proxy received nothing
    # and the plain stdlib call WOULD have used the proxy (proves the test can catch a leak)
    import urllib.request
    with pytest.raises(OSError):                                          # the fake proxy just hangs up
        urllib.request.urlopen(urllib.request.Request(url, data=b"leak"), timeout=2)
    time.sleep(0.3)
    assert proxy_listener
    req = b"".join(proxy_listener)
    port = url.rsplit(":", 1)[1].encode()
    assert req.startswith(b"POST " + url.encode())                        # proxy-style absolute request line
    assert re.search(rb"(?im)^host:\s*127\.0\.0\.1:" + port + rb"\s*$", req)
    assert req.endswith(b"leak")


def test_strip_think():
    assert local_llm.strip_think("<think>hmm</think>\n\nThe answer.") == "The answer."
    assert local_llm.strip_think("Partial <think>never closed") == "Partial"


def test_registry_has_builtins_and_rejects_unknown():
    names = answer.available()
    assert {"none", "transformers_peft", "ollama", "llamacpp_server", "llamacpp_python"} <= set(names)
    with pytest.raises(ValueError, match="unknown generation.backend"):
        answer.get_backend({"backend": "nope"})
    with pytest.raises(ValueError, match="answers are off"):
        answer.get_backend({"backend": "none"}).chat([], 5)


@pytest.fixture
def unregister_fakecloud():
    yield
    answer._REGISTRY.pop("fakecloud", None)
    answer._REGISTRY.pop("bad", None)


def test_a_drop_in_backend_registers_without_core_changes(tmp_path, monkeypatch, unregister_fakecloud):
    mod = tmp_path / "my_cloud_backend.py"
    mod.write_text(
        "from plm_kit.answer import Answer, AnswerBackend, register\n"
        "@register('fakecloud')\n"
        "class FakeCloud(AnswerBackend):\n"
        "    sends_off_machine = True\n"
        "    def generate(self, messages, *, max_tokens, temperature=None, stop=None):\n"
        "        return Answer('cloud says ' + self.backend_cfg.get('model', '?'), {'n': len(messages)})\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = {"backend": "fakecloud", "backend_modules": ["my_cloud_backend"], "fakecloud": {"model": "m1"}}
    with pytest.raises(ValueError, match="allow_off_machine"):
        answer.get_backend(cfg)                     # notes never leave the machine by default
    be = answer.get_backend(dict(cfg, allow_off_machine=True))
    assert be.generate([{"role": "user", "content": "q"}], max_tokens=5).text == "cloud says m1"
    assert list(be.stream([{"role": "user", "content": "q"}], max_tokens=5)) == ["cloud says m1"]
    text, rep = generate.answer_local("q", [("a.md", "x")], dict(cfg, allow_off_machine=True))
    assert text == "cloud says m1" and rep["backend"] == "fakecloud"
    with pytest.raises(ValueError, match="already registered"):
        answer.register("fakecloud")(type("Other", (answer.AnswerBackend,), {"generate": lambda *a, **k: None}))
    with pytest.raises(TypeError):
        answer.register("bad")(object)


def test_ollama_request_shape():
    seen = {}

    def post(url, payload, timeout):
        seen.update(url=url, payload=payload, timeout=timeout)
        return {"message": {"content": "<think>x</think>Hello"}}

    be = OllamaBackend(g({"threads": 4}), post=post)
    assert be.chat([{"role": "user", "content": "q"}], 50) == "Hello"
    a = be.generate([{"role": "user", "content": "q"}], max_tokens=5, temperature=0.0, stop=["\n\n"])
    assert a.text == "Hello" and a.meta["backend"] == "ollama" and seen["payload"]["options"]["stop"] == ["\n\n"]
    assert seen["payload"]["options"]["temperature"] == 0.0
    be.chat([{"role": "user", "content": "q"}], 50)
    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    p = seen["payload"]
    assert p["model"] == "qwen3:4b-instruct" and p["stream"] is False and p["think"] is False
    assert p["options"]["num_predict"] == 50 and p["options"]["num_thread"] == 4


def test_llamacpp_server_request_shape_and_bad_reply():
    seen = {}

    def post(url, payload, timeout):
        seen.update(url=url, payload=payload)
        return {"choices": [{"message": {"content": "Hi"}}]}

    be = LlamaCppServerBackend(g(llamacpp_server={"url": "http://127.0.0.1:18080"}), post=post)
    assert be.chat([{"role": "user", "content": "q"}], 10) == "Hi"
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    bad = LlamaCppServerBackend(g(), post=lambda *a: {"error": "x"})
    with pytest.raises(RuntimeError, match="unexpected reply"):
        bad.chat([], 5)


def test_unreachable_server_is_a_clear_error():
    be = LlamaCppServerBackend(g({"timeout_s": 2}, llamacpp_server={"url": "http://127.0.0.1:9"}))
    with pytest.raises(RuntimeError, match="can't reach"):
        be.chat([{"role": "user", "content": "q"}], 5)


def test_llamacpp_python_loader(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF")
    calls = {}

    class FakeLlama:
        def __init__(self, **kw):
            calls["init"] = kw

        def create_chat_completion(self, messages, max_tokens, temperature, stop=None):
            calls["chat"] = (messages, max_tokens, temperature)
            return {"choices": [{"message": {"content": " ok "}}]}

    be = LlamaCppPythonBackend(g({"threads": 2}, llamacpp_python={"gguf_path": str(f)}), loader=FakeLlama)
    assert be.chat([{"role": "user", "content": "q"}], 7) == "ok"
    assert calls["init"]["model_path"] == str(f) and calls["init"]["n_threads"] == 2
    with pytest.raises(RuntimeError, match="GGUF model not found"):
        LlamaCppPythonBackend(g(llamacpp_python={"gguf_path": str(tmp_path / "missing.gguf")}), loader=FakeLlama)


def test_answer_local_plain_uses_the_grounded_prompt():
    got = []
    text, rep = generate.answer_local("rate?", [("a.md", "rate is 5")], {"backend": "ollama"},
                                      chat=lambda m, n: got.append(m) or "5")
    assert text == "5" and rep["voice_lite"] is False
    assert "ONLY the excerpts" in got[0][0]["content"] and "rate is 5" in got[0][0]["content"]


def test_config_validates_backend_and_local_settings():
    cfg = cli.load_config()
    assert cfg["generation"]["backend"] == "auto"                # no-GPU by default; the CUDA path is opt-in
    from plm_kit.config import validate

    for bad in ({"backend": "gpt"}, {"local": {"model": "unknown"}}, {"local": {"temperature": -1}},
                {"voice_lite": {"copy_guard_ngram": 20}}, {"voice_lite": {"enabled": True}}):
        c = json.loads(json.dumps(cfg))
        c["generation"].update(bad)
        with pytest.raises(ValueError):
            validate(c)


def test_ask_answer_with_local_backend(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "c"))
    monkeypatch.setattr(local_llm, "post_json", lambda url, payload, timeout: {"message": {"content": "Reset it."}})
    rc = cli.main(["ask", "factory reset AX-7731", "--answer", "--backend", "ollama", "--k", "2"])
    out = capsys.readouterr()
    assert rc == 0 and "Reset it." in out.out and "sources: home-wifi.md" in out.err


def test_download_gguf_plans_pinned_file(tmp_path, monkeypatch):
    from plm_kit import download

    cfg = cli.load_config()
    items = download.plan(cfg, ["gguf"])
    assert items[0][0] == "gguf" and items[0][2]["file"] == models.LOCAL_GENERATORS["qwen3-4b-instruct"]["file"]
    got = []
    monkeypatch.setattr(download, "fetch_file", lambda spec, d: got.append((spec["hf_id"], spec["revision"])) or str(d / spec["file"]))
    import io

    buf = io.StringIO()
    done = download.run(cfg, ["gguf"], cache=str(tmp_path / "c"), assume_yes=True, out=buf)
    assert "LICENSE NOTICE" in buf.getvalue() and "Qwen3-4B-Q4_K_M.gguf" in buf.getvalue()
    assert got == [("Qwen/Qwen3-4B-GGUF", models.LOCAL_GENERATORS["qwen3-4b-instruct"]["revision"])]
    assert done["gguf"].startswith(str(tmp_path / "c"))


@pytest.mark.llm
@pytest.mark.skipif(not __import__("os").environ.get("PLM_KIT_TEST_LLM_URL"),
                    reason="set PLM_KIT_TEST_LLM_URL=http://127.0.0.1:8080 (a running llama-server) to run")
def test_real_llamacpp_server_end_to_end():
    import os

    from plm_kit import voice_lite as vl
    from plm_kit.selftest import STYLE_EXAMPLES

    gen = {"backend": "llamacpp_server", "max_new_tokens": 80,
           "local": {"temperature": 0}, "llamacpp_server": {"url": os.environ["PLM_KIT_TEST_LLM_URL"]},
           "voice_lite": {"examples_file": str(STYLE_EXAMPLES)}}
    ex = [("business/invoicing.md", "Invoices go out on the 1st and the 15th of each month. Payment terms are net 15.")]
    text, _ = generate.answer_local("When do invoices go out?", ex, gen, voice_lite=False)
    assert "15" in text
    text, _rep = generate.answer_local("When do invoices go out?", ex, gen, voice_lite=True)
    assert text and vl.find_copies(text, vl.load_examples(STYLE_EXAMPLES), 12) == []


# --- generation.backend: auto (the default) ------------------------------------------------------------------------

@pytest.fixture
def ollama_version_server():
    """A tiny local stand-in for Ollama's GET /api/version."""
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"version": "0.0.0"}' if self.path == "/api/version" else b"{}"
            self.send_response(200 if self.path == "/api/version" else 404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _closed_port_url():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def test_auto_is_the_default_and_validates():
    from plm_kit.config import validate

    assert answer.DEFAULT_BACKEND == "auto" and "auto" in answer.available()
    cfg = cli.load_config()
    validate(cfg)


def test_auto_picks_ollama_when_it_answers(ollama_version_server, monkeypatch, proxy_listener):
    monkeypatch.setattr(local_llm, "llama_cpp_installed", lambda: True)    # Ollama still wins
    gen = g(ollama={"host": ollama_version_server})
    assert local_llm.resolve_auto(gen) == "ollama"
    be = answer.get_backend(dict(gen, backend="auto"))
    assert be.name == "ollama" and be.model == "qwen3:4b-instruct"         # the Qwen3-4B registry tag
    assert answer.get_backend(gen).name == "ollama"                        # no backend key -> auto
    assert proxy_listener == []                                            # the probe ignores proxies too


def test_auto_falls_back_to_llamacpp_python_when_ollama_is_down(monkeypatch, tmp_path):
    monkeypatch.setattr(local_llm, "llama_cpp_installed", lambda: True)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    gen = g(ollama={"host": _closed_port_url()}, llamacpp_python={"gguf_path": str(gguf)})
    assert local_llm.resolve_auto(gen) == "llamacpp_python"
    monkeypatch.setattr(LlamaCppPythonBackend, "__init__", lambda self, gen_cfg, *, cache=None: None)
    assert answer.get_backend(dict(gen, backend="auto")).name == "llamacpp_python"


def test_auto_without_either_is_a_clear_install_error(monkeypatch):
    monkeypatch.setattr(local_llm, "llama_cpp_installed", lambda: False)
    gen = g(ollama={"host": _closed_port_url()})
    with pytest.raises(RuntimeError) as e:
        answer.get_backend(dict(gen, backend="auto"))
    msg = str(e.value)
    assert "ollama pull qwen3:4b-instruct" in msg and "requirements-local.txt" in msg
    assert "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu" in msg
    assert "download --components gguf" in msg and "transformers_peft" in msg


def test_auto_treats_a_refused_remote_ollama_host_as_unreachable(monkeypatch):
    monkeypatch.setattr(local_llm, "llama_cpp_installed", lambda: False)
    with pytest.raises(RuntimeError, match="No answer backend found"):
        local_llm.resolve_auto(g(ollama={"host": "http://192.168.1.20:11434"}))   # never contacted


def test_ask_answer_with_auto_and_nothing_installed_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "c"))
    monkeypatch.setattr(local_llm, "llama_cpp_installed", lambda: False)
    monkeypatch.setattr(local_llm, "ollama_reachable", lambda gen_cfg, timeout=2.0: False)
    rc = cli.main(["ask", "factory reset AX-7731", "--answer", "--k", "2"])
    assert rc == 2 and "No answer backend found" in capsys.readouterr().err


def test_selftest_without_answer_needs_no_backend(tmp_path, monkeypatch):
    from plm_kit import selftest

    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "c"))
    def boom(*a, **k):
        raise AssertionError("selftest without --answer must not touch an answer backend")

    monkeypatch.setattr(local_llm, "resolve_auto", boom)
    monkeypatch.setattr(answer, "get_backend", boom)
    import io

    assert selftest.run(cli.load_config(), out=io.StringIO()) == 0
