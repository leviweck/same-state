"""Built-in non-local backends: `none` (excerpts only) and `transformers_peft` (CUDA GPU, the original path)."""
from plm_kit import paths
from plm_kit.answer import Answer, AnswerBackend, register


@register("none")
class NoAnswer(AnswerBackend):
    def generate(self, messages, *, max_tokens, temperature=None, stop=None):
        raise ValueError("generation.backend is 'none': answers are off (excerpts only). Pick a backend, "
                         "e.g. --backend ollama")


@register("transformers_peft")
class TransformersPeft(AnswerBackend):
    """Qwen3-8B (pinned) in 4-bit with your optional LoRA adapter; needs `plm-kit download --include-generator`."""

    def __init__(self, gen_cfg, *, cache=None):
        super().__init__(gen_cfg, cache=cache)
        self.model_dir = paths.cache_dir(cache) / "qwen3-8b-src"
        if not (self.model_dir / "config.json").is_file():
            raise RuntimeError("Generator not downloaded. Run `plm-kit download --include-generator` first.")

    def generate(self, messages, *, max_tokens, temperature=None, stop=None):
        from plm_kit import generate as g

        prompt = "\n\n".join(m["content"] for m in messages)
        text = g.generate(str(self.model_dir), prompt, dict(self.gen_cfg, max_new_tokens=max_tokens))
        return Answer(text, {"backend": self.name, "model": self.gen_cfg.get("base_model")})
