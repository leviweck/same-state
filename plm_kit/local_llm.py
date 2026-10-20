"""Helpers shared by the no-GPU answer backends (plm_kit/answer_backends/ollama.py, llamacpp.py).

HTTP goes through the standard library only (no new dependency) and, by default, ONLY to a loopback address
(127.0.0.1 / ::1 / localhost that resolves to loopback): your notes are in the prompt, so they must not leave the
machine by accident. A non-loopback host needs BOTH `generation.local.allow_remote: true` and
`generation.allow_off_machine: true` (see PRIVACY.md). Requests never go through a proxy: http_proxy/HTTPS_PROXY and
friends are ignored, so a proxy can't see your notes or voice-lite examples.
No telemetry, no model download here (see `plm-kit download --components gguf` or `ollama pull`)."""
import importlib.util
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

LOCAL_BACKENDS = ("ollama", "llamacpp_server", "llamacpp_python")
DEFAULTS = {                                   # generation.local: shared by the local backends
    "model": "qwen3-4b-instruct",              # registry key in plm_kit/models.py LOCAL_GENERATORS
    "n_ctx": 8192,
    "threads": None,                           # None = backend default
    "temperature": 0.2,
    "timeout_s": 600,
    "allow_remote": False,
}
BACKEND_DEFAULTS = {                           # generation.<backend>: per-backend keys
    "ollama": {"host": "http://127.0.0.1:11434", "model": None},        # model None = registry Ollama tag
    "llamacpp_server": {"url": "http://127.0.0.1:8080"},
    "llamacpp_python": {"gguf_path": None},                             # None = <cache>/gguf/<registry file>
}
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def settings(gen_cfg):
    s = dict(DEFAULTS)
    s.update({k: v for k, v in ((gen_cfg or {}).get("local") or {}).items() if v is not None})
    if float(s["temperature"]) < 0:
        raise ValueError("generation.local.temperature must be >= 0")
    if int(s["n_ctx"]) < 512:
        raise ValueError("generation.local.n_ctx must be >= 512")
    return s


def backend_settings(gen_cfg, name):
    s = dict(BACKEND_DEFAULTS[name])
    s.update({k: v for k, v in ((gen_cfg or {}).get(name) or {}).items() if v is not None})
    return s


def strip_think(text):
    """Qwen3-style hybrid models may emit a <think>...</think> block; answers never include it."""
    text = _THINK.sub("", text or "")
    if "<think>" in text:   # unterminated (hit max tokens while thinking): drop from the tag on
        text = text.split("<think>", 1)[0]
    return text.strip()


def _is_loopback_host(host):
    """True only for a literal loopback IP, or `localhost` when EVERY address it resolves to is loopback
    (a remapped /etc/hosts entry fails)."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    if host.lower().rstrip(".") != "localhost":
        return False
    try:
        addrs = {ai[4][0] for ai in socket.getaddrinfo(host, None)}
    except OSError:
        return False
    return bool(addrs) and all(ipaddress.ip_address(a.split("%", 1)[0]).is_loopback for a in addrs)


def check_url(url, allow_remote=False, allow_off_machine=False):
    """Returns (url without trailing slash, is_remote). A non-loopback host is refused unless BOTH
    generation.local.allow_remote and generation.allow_off_machine are true; then it counts as off-machine."""
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError(f"bad local model URL {url!r}: need http(s)://host:port")
    host = u.hostname
    remote = not _is_loopback_host(host)
    if remote and not (allow_remote and allow_off_machine):
        raise ValueError(f"refusing to send your notes to {host}: only a loopback address (127.0.0.1, ::1, localhost "
                         "resolving to loopback) is allowed. A server elsewhere needs BOTH "
                         "generation.local.allow_remote: true and generation.allow_off_machine: true (see PRIVACY.md)")
    return url.rstrip("/"), remote


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # no proxy, ever (env vars ignored)


def open_url(req, timeout):
    """Every local-backend HTTP call goes through here: proxy settings from the environment are ignored."""
    return _OPENER.open(req, timeout=timeout)


def post_json(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with open_url(req, timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"local model server returned HTTP {e.code} at {url}: {body}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise RuntimeError(f"can't reach the local model server at {url} ({e}). Is it running? "
                           "See docs/NO_GPU.md.") from e


AUTO_ORDER = ("ollama", "llamacpp_python")
AUTO_HELP = ("No answer backend found (generation.backend: auto). No GPU needed; install either one:\n"
             "  1. Ollama (https://ollama.com): install it, start it, then `ollama pull {tag}`\n"
             "  2. llama-cpp-python: `pip install -r requirements-local.txt "
             "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`, then "
             "`plm-kit download --components gguf`\n"
             "Then run the command again. Have a CUDA GPU? Set generation.backend: transformers_peft. "
             "Search without answers (excerpts only) works with nothing installed. See docs/NO_GPU.md.")


def ollama_reachable(gen_cfg, timeout=2.0):
    """True if an Ollama server answers GET /api/version at generation.ollama.host (same loopback rules)."""
    gen_cfg = gen_cfg or {}
    try:
        base, _ = check_url(backend_settings(gen_cfg, "ollama")["host"], settings(gen_cfg)["allow_remote"],
                            bool(gen_cfg.get("allow_off_machine")))
    except ValueError:
        return False
    try:
        with open_url(urllib.request.Request(base + "/api/version"), timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError):
        return False


def llama_cpp_installed():
    return importlib.util.find_spec("llama_cpp") is not None


def resolve_auto(gen_cfg):
    """generation.backend: auto -> `ollama` if reachable, else `llamacpp_python` if installed, else a clear error."""
    if ollama_reachable(gen_cfg):
        return "ollama"
    if llama_cpp_installed():
        return "llamacpp_python"
    from plm_kit.models import local_generator

    tag = backend_settings(gen_cfg, "ollama").get("model") or local_generator(settings(gen_cfg)["model"])["ollama"]
    raise RuntimeError(AUTO_HELP.format(tag=tag))
