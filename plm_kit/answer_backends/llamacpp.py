"""llama.cpp backends.
- `llamacpp_server`: llama.cpp's `llama-server` (OpenAI-compatible POST /v1/chat/completions).
  Config: generation.llamacpp_server.url (default http://127.0.0.1:8080).
- `llamacpp_python`: llama-cpp-python in this process (`pip install -r requirements-local.txt`).
  Config: generation.llamacpp_python.gguf_path (default <cache>/gguf/<registry file>, from
  `plm-kit download --components gguf`).
Shared settings in generation.local."""
import pathlib

from plm_kit import local_llm, paths
from plm_kit.answer import Answer, AnswerBackend, register


@register("llamacpp_server")
class LlamaCppServerBackend(AnswerBackend):
    def __init__(self, gen_cfg, *, cache=None, post=None):
        super().__init__(gen_cfg, cache=cache)
        self.s = local_llm.settings(gen_cfg)
        b = local_llm.backend_settings(gen_cfg, "llamacpp_server")
        base, remote = local_llm.check_url(b["url"], self.s["allow_remote"],
                                           bool((gen_cfg or {}).get("allow_off_machine")))
        self.sends_off_machine = remote   # a remote server means your notes leave this machine
        self.url = base + "/v1/chat/completions"
        self.post = post or local_llm.post_json

    def generate(self, messages, *, max_tokens, temperature=None, stop=None):
        payload = {"messages": messages, "max_tokens": int(max_tokens), "stream": False,
                   "temperature": float(self.s["temperature"] if temperature is None else temperature),
                   "chat_template_kwargs": {"enable_thinking": False}}
        if stop:
            payload["stop"] = list(stop)
        r = self.post(self.url, payload, float(self.s["timeout_s"]))
        try:
            text = local_llm.strip_think(r["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected reply from llama-server: {str(r)[:200]}") from e
        return Answer(text, {"backend": self.name, "model": r.get("model"), "usage": r.get("usage")})


@register("llamacpp_python")
class LlamaCppPythonBackend(AnswerBackend):
    def __init__(self, gen_cfg, *, cache=None, loader=None):
        super().__init__(gen_cfg, cache=cache)
        from plm_kit.models import local_generator

        self.s = local_llm.settings(gen_cfg)
        b = local_llm.backend_settings(gen_cfg, "llamacpp_python")
        p = b.get("gguf_path") or paths.cache_dir(cache) / "gguf" / local_generator(self.s["model"])["file"]
        self.path = pathlib.Path(p).expanduser()
        if not self.path.is_file():
            raise RuntimeError(f"GGUF model not found at {self.path}. Run `plm-kit download --components gguf` "
                               "or set generation.llamacpp_python.gguf_path")
        if loader is None:
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise RuntimeError("llamacpp_python needs `pip install -r requirements-local.txt --extra-index-url "
                                   "https://abetlen.github.io/llama-cpp-python/whl/cpu` "
                                   "(llama-cpp-python, MIT); or use the ollama / llamacpp_server backend") from e
            loader = Llama
        kw = {"model_path": str(self.path), "n_ctx": int(self.s["n_ctx"]), "verbose": False}
        if self.s.get("threads"):
            kw["n_threads"] = int(self.s["threads"])
        self.llm = loader(**kw)

    def generate(self, messages, *, max_tokens, temperature=None, stop=None):
        r = self.llm.create_chat_completion(
            messages=messages, max_tokens=int(max_tokens), stop=list(stop) if stop else None,
            temperature=float(self.s["temperature"] if temperature is None else temperature))
        return Answer(local_llm.strip_think(r["choices"][0]["message"]["content"]),
                      {"backend": self.name, "model": self.path.name, "usage": r.get("usage")})
