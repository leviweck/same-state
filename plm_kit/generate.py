"""Answer generation. Only used by `plm-kit ask --answer` (and `plm-kit selftest --answer`). It never downloads.

generation.backend:
- auto (default): no GPU needed. Ollama if it's running, else llama-cpp-python if installed, else a clear error.
- ollama | llamacpp_server | llamacpp_python: no GPU needed (CPU or Apple Silicon), picked explicitly.
- transformers_peft (opt-in): transformers + peft, 4-bit on a CUDA GPU; `plm-kit download --include-generator` first.
- none, or any backend registered via plm_kit/answer.py (pluggable; see that module for how to add one).
Voice lite (generation.voice_lite, local backends): few-shot style from your 5-8 example lines, with the
verbatim-copy guard in plm_kit/voice_lite.py."""

PROMPT = ("Answer the question using ONLY the excerpts from the user's notes below. "
          "If the excerpts don't contain the answer, say it isn't in the notes.\n\n{ctx}\n\nQuestion: {q}\nAnswer:")


def build_prompt(question, excerpts):
    ctx = "\n\n".join(f"[{i + 1}] ({doc})\n{text}" for i, (doc, text) in enumerate(excerpts))
    return PROMPT.format(ctx=ctx, q=question)


def generate(model_dir, prompt, gen_cfg):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("answer mode needs a CUDA GPU (transformers + peft, 4-bit). Use excerpts mode on CPU.")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=q, device_map="auto",
                                                 local_files_only=True)
    if gen_cfg.get("adapter_dir"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, gen_cfg["adapter_dir"], local_files_only=True)
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", enable_thinking=False).to(model.device)
    out = model.generate(ids, max_new_tokens=gen_cfg.get("max_new_tokens", 512), do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def answer_local(question, excerpts, gen_cfg, cache=None, backend=None, voice_lite=None, chat=None):
    """Answer with any registered backend (plm_kit/answer.py). Returns (text, report).
    `backend`: overrides generation.backend. `voice_lite`: None -> generation.voice_lite.enabled decides.
    `chat`: injectable (messages, max_tokens) -> str for tests; default = the selected backend's chat()."""
    from plm_kit import answer
    from plm_kit import voice_lite as vl

    vs = vl.settings(gen_cfg)
    use_voice = vs["enabled"] if voice_lite is None else voice_lite
    if use_voice and not vs.get("examples_file"):
        raise ValueError("voice lite needs generation.voice_lite.examples_file")
    if chat is None:
        be = answer.get_backend(gen_cfg, name=backend, cache=cache)
        chat, name = be.chat, be.name
    else:
        name = backend or (gen_cfg or {}).get("backend")
    prompt = build_prompt(question, excerpts)
    max_tokens = int((gen_cfg or {}).get("max_new_tokens") or 512)
    report = {"backend": name, "voice_lite": bool(use_voice)}
    if not use_voice:
        return chat([{"role": "user", "content": prompt}], max_tokens), report
    examples = vl.load_examples(vs["examples_file"])
    text, guard = vl.guarded_generate(chat, prompt, examples, vs, max_tokens)
    report["copy_guard"] = guard
    return text, report
