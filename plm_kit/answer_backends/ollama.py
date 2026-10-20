"""Backend `ollama`: Ollama's local HTTP API (POST /api/chat). You install Ollama and `ollama pull` the model.
Config: generation.ollama.host (default http://127.0.0.1:11434), generation.ollama.model (default: the registry
tag for generation.local.model, e.g. qwen3:4b-instruct); shared settings in generation.local."""
from plm_kit import local_llm
from plm_kit.answer import Answer, AnswerBackend, register


@register("ollama")
class OllamaBackend(AnswerBackend):
    def __init__(self, gen_cfg, *, cache=None, post=None):
        super().__init__(gen_cfg, cache=cache)
        from plm_kit.models import local_generator

        self.s = local_llm.settings(gen_cfg)
        b = local_llm.backend_settings(gen_cfg, "ollama")
        base, remote = local_llm.check_url(b["host"], self.s["allow_remote"],
                                           bool((gen_cfg or {}).get("allow_off_machine")))
        self.sends_off_machine = remote   # a remote server means your notes leave this machine
        self.url = base + "/api/chat"
        self.model = b.get("model") or local_generator(self.s["model"])["ollama"]
        self.post = post or local_llm.post_json

    def generate(self, messages, *, max_tokens, temperature=None, stop=None):
        opts = {"temperature": float(self.s["temperature"] if temperature is None else temperature),
                "num_predict": int(max_tokens), "num_ctx": int(self.s["n_ctx"])}
        if self.s.get("threads"):
            opts["num_thread"] = int(self.s["threads"])
        if stop:
            opts["stop"] = list(stop)
        r = self.post(self.url, {"model": self.model, "messages": messages, "stream": False, "think": False,
                                 "options": opts}, float(self.s["timeout_s"]))
        text = local_llm.strip_think((r.get("message") or {}).get("content", ""))
        return Answer(text, {"backend": self.name, "model": self.model, "eval_count": r.get("eval_count"),
                             "total_duration_ns": r.get("total_duration")})
