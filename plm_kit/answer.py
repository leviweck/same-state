"""Pluggable answer backends: one small interface, a registry keyed by the config name, and drop-in discovery.

Select with `generation.backend` (CLI: `plm-kit ask --answer --backend NAME`). Built in:
  none               no generation (excerpts only)
  auto               default: ollama if it is running, else llamacpp_python if installed, else a clear error
  transformers_peft  transformers + peft on a CUDA GPU (opt-in)
  ollama             Ollama on this machine            (plm_kit/answer_backends/ollama.py)
  llamacpp_server    llama.cpp `llama-server`           (plm_kit/answer_backends/llamacpp.py)
  llamacpp_python    llama-cpp-python, in process      (plm_kit/answer_backends/llamacpp.py)

Config: `generation.<backend name>` holds that backend's own keys (e.g. generation.ollama.host / .model);
`generation.local` holds settings shared by the local backends (model registry key, n_ctx, temperature, ...).

Adding a backend (e.g. a hosted API) WITHOUT touching core:
  1. Drop a module into plm_kit/answer_backends/ (auto-imported), or ship a package that declares an entry point
     in group "plm_kit.answer_backends", or list the module in `generation.backend_modules`.
  2. In it:
         from plm_kit.answer import AnswerBackend, Answer, register

         @register("myapi")
         class MyApi(AnswerBackend):
             sends_off_machine = True          # notes leave this computer -> needs generation.allow_off_machine: true
             def __init__(self, gen_cfg, *, cache=None):
                 super().__init__(gen_cfg, cache=cache)
                 self.opts = self.backend_cfg   # == gen_cfg["myapi"] or {}
                 # read secrets from the environment (e.g. os.environ["MYAPI_KEY"]); never from config files
             def generate(self, messages, *, max_tokens, temperature=None, stop=None):
                 ...
                 return Answer(text, {"model": ..., "usage": ...})
  3. Set generation.backend: myapi (and generation.allow_off_machine: true if it sends data off the machine).
"""
import abc
import dataclasses
import importlib
import pkgutil
from collections.abc import Iterator
from typing import ClassVar

ENTRY_POINT_GROUP = "plm_kit.answer_backends"
_REGISTRY: dict = {}
_discovered = False


@dataclasses.dataclass
class Answer:
    text: str
    meta: dict = dataclasses.field(default_factory=dict)   # backend, model, timings, usage, ... (backend-defined)


class AnswerBackend(abc.ABC):
    """Base class. Subclasses set `name` (via @register) and implement generate()."""
    name: ClassVar[str] = ""
    sends_off_machine: ClassVar[bool] = False   # True for hosted APIs: core then requires generation.allow_off_machine

    def __init__(self, gen_cfg, *, cache=None):
        self.gen_cfg = gen_cfg or {}
        self.cache = cache
        self.backend_cfg = dict(self.gen_cfg.get(self.name) or {})

    @abc.abstractmethod
    def generate(self, messages, *, max_tokens, temperature=None, stop=None) -> Answer:
        """messages: [{"role": "system"|"user"|"assistant", "content": str}]. temperature None = backend default.
        stop: optional list of stop strings. Returns Answer(text, meta). Raise RuntimeError for runtime failures
        (server down, model missing) and ValueError for bad settings; the CLI prints both cleanly."""

    def stream(self, messages, *, max_tokens, temperature=None, stop=None) -> Iterator[str]:
        """Optional token streaming. Default: one chunk with the whole answer."""
        yield self.generate(messages, max_tokens=max_tokens, temperature=temperature, stop=stop).text

    def chat(self, messages, max_tokens):
        """(messages, max_tokens) -> text; the callable shape voice lite's copy guard uses."""
        return self.generate(messages, max_tokens=max_tokens).text


def register(name):
    """Class decorator: make a backend selectable as generation.backend: <name>."""
    def deco(cls):
        if not (isinstance(cls, type) and issubclass(cls, AnswerBackend)):
            raise TypeError(f"{cls!r} must subclass AnswerBackend")
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"answer backend {name!r} is already registered by {_REGISTRY[name].__module__}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def discover(extra_modules=()):
    """Import built-in/drop-in modules in plm_kit.answer_backends, entry points, and generation.backend_modules."""
    global _discovered
    if not _discovered:
        from plm_kit import answer_backends

        for m in pkgutil.iter_modules(answer_backends.__path__):
            importlib.import_module(f"{answer_backends.__name__}.{m.name}")
        try:
            from importlib.metadata import entry_points

            for ep in entry_points(group=ENTRY_POINT_GROUP):
                ep.load()
        except Exception as e:  # noqa: BLE001 - a broken third-party plugin must not break the kit
            import sys

            print(f"note: skipped an answer-backend plugin ({e})", file=sys.stderr)
        _discovered = True
    for mod in extra_modules or ():
        importlib.import_module(mod)
    return dict(_REGISTRY)


DEFAULT_BACKEND = "auto"   # no GPU needed: ollama if it's running, else llamacpp_python (plm_kit/local_llm.py)


def available(gen_cfg=None):
    return sorted(set(discover((gen_cfg or {}).get("backend_modules") or ())) | {"auto"})


def get_backend(gen_cfg, *, name=None, cache=None) -> AnswerBackend:
    gen_cfg = gen_cfg or {}
    reg = discover(gen_cfg.get("backend_modules") or ())
    name = name or gen_cfg.get("backend", DEFAULT_BACKEND)
    if name == "auto":
        from plm_kit import local_llm

        name = local_llm.resolve_auto(gen_cfg)
    if name not in reg:
        raise ValueError(f"unknown generation.backend {name!r}; available: {sorted(reg)}")
    cls = reg[name]
    if cls.sends_off_machine and not gen_cfg.get("allow_off_machine"):
        raise ValueError(f"backend {name!r} sends your notes off this machine; set generation.allow_off_machine: "
                         "true to allow it (see PRIVACY.md)")
    return cls(gen_cfg, cache=cache)
