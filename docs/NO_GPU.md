# No-GPU answers and voice lite

Search and excerpts (`plm-kit ask`) have always run on CPU. This page covers **generated answers without a GPU**
(CPU, or Apple Silicon via Metal) and **voice lite** (answers in your style with no training).
No GPU is needed out of the box. A GPU is only used by the opt-in `transformers_peft` backend.

## 1. Pick a backend
You don't have to: the default `generation.backend: auto` picks one for each answer.
1. **Ollama**, if it answers at `generation.ollama.host` (default `http://127.0.0.1:11434`); model `qwen3:4b-instruct`
   (the registry tag for `generation.local.model`) unless `generation.ollama.model` says otherwise.
2. Else **llama-cpp-python**, if it's installed (`requirements-local.txt`); it loads the GGUF from
   `plm-kit download --components gguf`.
3. Else a clear error with install steps for either. Search, excerpts and `plm-kit selftest` without `--answer` never
   need a backend.

To pin a backend, set `generation.backend` in your config (or pass `plm-kit ask --answer --backend ...` per question).

| Backend | You install | Runs on | Notes |
|---|---|---|---|
| `ollama` | [Ollama](https://ollama.com) (MIT), then `ollama pull qwen3:4b-instruct` | Windows, macOS (Metal), Linux; CPU or GPU | Easiest. The kit talks to `http://127.0.0.1:11434` |
| `llamacpp_server` | [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) release binary, then `llama-server -m <file>.gguf` | Same; prebuilt CPU, Metal, Vulkan builds | The kit talks to `http://127.0.0.1:8080` (OpenAI-compatible API) |
| `llamacpp_python` | `pip install -r requirements-local.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu` (llama-cpp-python, MIT) | In this Python process | Loads the `.gguf` from `plm-kit download --components gguf` (or `generation.llamacpp_python.gguf_path`) |
| `transformers_peft` (opt-in) | `requirements.txt` + a CUDA GPU | NVIDIA GPU | Advanced; never picked by `auto` |
| `none` | nothing | - | Answers off; `ask` shows excerpts only |

Config layout: `generation.backend` picks the backend; each backend's own keys live under `generation.<name>`
(`generation.ollama.host` / `.model`, `generation.llamacpp_server.url`, `generation.llamacpp_python.gguf_path`);
settings shared by the local backends (model, `n_ctx`, `threads`, `temperature`, `timeout_s`, `allow_remote`) live
under `generation.local`.

The kit sends requests **only to a loopback address** (127.0.0.1, ::1, or `localhost` when it resolves to loopback;
a remapped hosts entry is refused) because your notes are in the prompt. A server anywhere else needs **both**
`generation.local.allow_remote: true` and `generation.allow_off_machine: true`, and then counts as sending your
notes off the machine. Requests never use a proxy: `http_proxy` / `HTTPS_PROXY` and similar are ignored. It never downloads models for Ollama or llama-server; you pull or download them.
One case the kit can't detect: if you set `generation.ollama.model` to an Ollama "cloud" model, Ollama itself sends your question and note excerpts to Ollama's servers under Ollama's own terms. The default `qwen3:4b-instruct` runs locally; use a local model if you want everything to stay on your computer.
Thinking-style output (`<think>...</think>`) is switched off and stripped.

## 2. Pick a model
All registry models are Apache-2.0, pinned to a Hugging Face commit, and checked by sha256 when downloaded
(`plm_kit/models.py`, `LOCAL_GENERATORS`). Set `generation.local.model`.

| `generation.local.model` | Ollama tag | GGUF file (first-party repo) | Download | RAM in use (approx.) | For |
|---|---|---|---|---|---|
| `qwen3-4b-instruct` (**default**) | `qwen3:4b-instruct` | `Qwen/Qwen3-4B-GGUF` / `Qwen3-4B-Q4_K_M.gguf` | 2.5 GB | 3.9-6.1 GB (measured, see below) | 8-16 GB machines |
| `granite-4.0-micro` | `granite4:micro` | `ibm-granite/granite-4.0-micro-GGUF` / `granite-4.0-micro-Q4_K_M.gguf` | 2.1 GB | ~4 GB | Alternative publisher |
| `qwen2.5-1.5b-instruct` | `qwen2.5:1.5b` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` / `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1.1 GB | ~3 GB | 8 GB / older laptops |
| `qwen2.5-0.5b-instruct` | `qwen2.5:0.5b` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` / `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 0.5 GB | ~1.5 GB | Smoke tests only |

Notes:
- The Ollama tag for the default is Ollama's build of Qwen3-4B-Instruct-2507; the GGUF file is the original
  hybrid Qwen3-4B with thinking switched off by the kit. Same family and license, slightly different checkpoints.
- Left out on purpose: the 3B and largest Qwen2.5 sizes (not Apache-2.0) and models under custom community
  licenses with use restrictions. If you use another model (any Ollama tag via `generation.ollama.model`,
  or any GGUF via `generation.llamacpp_python.gguf_path`), check its license yourself.
- RAM figures for the other models are guidance (Q4 weights plus an 8k context). On 8 GB machines lower
  `generation.local.n_ctx` to 4096 and use `qwen2.5-1.5b-instruct`. CPU answers take seconds to a minute or more
  depending on the machine and how much of your notes goes into the prompt.

### Default model, measured on an 8-core CPU box
`Qwen3-4B-Q4_K_M.gguf`, no GPU, 8 threads, `n_ctx` 8192, temperature 0, one backend at a time, thinking off.
Measured on an 8-core x86-64 Linux box with 16 GB RAM and the GGUF file already in the OS file cache, using llama.cpp
build b11741 (`llama-server`), Ollama (the same GGUF imported with a Modelfile), and llama-cpp-python 0.3.36 (CPU wheel).
"Short answers" are 6 golden questions from `starter_kit/` (prompts of 73-190 tokens, answers of 3-23 tokens).

| | llama-server | Ollama | llama-cpp-python |
|---|---|---|---|
| Load time (start to ready) | 3.0-3.7 s | 5.1-5.3 s | 2.8 s |
| Time to first token, short answers | 0.6-1.8 s | 0.7-1.1 s | 1.0-2.6 s (one outlier 5.4 s) |
| Total per answer, short answers | 1.1-2.5 s | 1.0-2.4 s | 1.9-4.9 s (one outlier 6.2 s) |
| Generation speed (a ~250-token, 200-word summary) | 10.5 tok/s (24.7 s total) | 11.8 tok/s (23.0 s total) | 10.1 tok/s (25.7 s total) |
| Prompt reading speed (a ~2,100-2,300-token prompt) | 142 tok/s (first token at 14.9 s) | 142 tok/s (first token at 16.1 s) | ~55 tok/s (first token at 37-41 s) |
| Peak memory (RSS) | 5.9-6.1 GB | 3.9-4.2 GB | 5.5 GB (whole Python process) |

Takeaways: on a machine like this, short answers take 1-3 seconds and a long one about 25 seconds. Prompt length
matters most: every ~1,000 tokens of excerpts adds about 7 seconds before the first word (about 18 seconds with
llama-cpp-python, whose prebuilt CPU wheel read prompts more slowly here). Expect slower on laptops with fewer cores.

## 3. Quick start
```bash
# Ollama
ollama pull qwen3:4b-instruct
plm-kit ask "when do invoices go out?" --answer          # auto finds Ollama; --backend ollama pins it

# llama.cpp server
plm-kit download --components gguf          # pinned GGUF into <cache>/gguf/, license notice first
llama-server -m ~/.cache/plm-kit/gguf/Qwen3-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8080
plm-kit ask "when do invoices go out?" --answer --backend llamacpp_server

# in-process
pip install -r requirements-local.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
plm-kit download --components gguf
plm-kit ask "when do invoices go out?" --answer          # auto falls back to llama-cpp-python when Ollama isn't up
```
Answers print to stdout; the source notes used print to stderr. `plm-kit selftest --answer` checks your backend
on the bundled sample notes.

## 4. Voice lite (your style, no training)
Voice lite shows the model 5-8 lines **you** wrote and asks it to match the tone, length and punctuation. Nothing is
trained and nothing leaves your machine.

1. Make a text file **outside the repo** with 5-8 lines you wrote (one per line; `#` lines are comments), e.g.
   short replies or messages that sound like you. JSONL with `{"text": "..."}` per line also works.
   Use your own writing only, not text an assistant drafted for you, or it will learn its own style back.
2. Set `generation.voice_lite.examples_file: /path/to/style-examples.txt`.
3. Per question: `plm-kit ask "..." --answer --backend ollama --voice-lite`, or `generation.voice_lite.enabled: true`.

The examples are labelled **style only, not facts**, so answers still come from your notes.

### Verbatim-copy guard
Voice lite must never hand your example lines back to you (or to whoever reads the answer). After every answer the
kit looks for any run of **12 or more words in a row** copied from an example (`copy_guard_ngram`, 4-12; lower is
stricter), and for any whole example of 4+ words. Case and punctuation are ignored. If it finds one:
1. it regenerates (`retries`, default 1) with a "use new words" instruction;
2. if it still copies, it removes the copied words (`on_copy: redact`, shown as `[...]`) or withholds the answer
   (`on_copy: block`).
stderr says how often it fired. `plm-kit copy-check FILE --examples style-examples.txt` runs the same check on any
text (exit 1 if flagged) and reports only positions and lengths, never the example text.

Voice lite is the style feature in v1. Voice training is not a v1 feature; a trained adapter (advanced, optional,
`transformers_peft` only) is only worth it if it beats voice lite in a blind test.

## 5. Adding a backend (developers)
Backends are plugins behind one small interface in `plm_kit/answer.py`:

```python
from plm_kit.answer import Answer, AnswerBackend, register

@register("mybackend")                     # selectable as generation.backend: mybackend
class MyBackend(AnswerBackend):
    sends_off_machine = False              # True for hosted APIs: then generation.allow_off_machine must be true
    def __init__(self, gen_cfg, *, cache=None):
        super().__init__(gen_cfg, cache=cache)
        opts = self.backend_cfg            # = generation.mybackend (dict); read secrets from env vars, never config
    def generate(self, messages, *, max_tokens, temperature=None, stop=None) -> Answer:
        return Answer(text="...", meta={"model": "..."})
    # optional: def stream(self, messages, *, max_tokens, temperature=None, stop=None) -> Iterator[str]
```
Make it discoverable in any one of three ways, with no change to core code: put the module in
`plm_kit/answer_backends/` (auto-imported), list it in `generation.backend_modules`, or expose it from your own
package as an entry point in the group `plm_kit.answer_backends`. Voice lite, the copy guard, `ask --answer` and
`selftest --answer` work with every backend. A backend that sends notes off the computer is refused unless the
user sets `generation.allow_off_machine: true`.

**The gate trusts the plugin.** `sends_off_machine` is declared by the plugin itself; the kit can't verify it. A
plugin that sets it to `False` but calls a remote API will not be stopped. Only install backends whose code you
have read or whose publisher you trust. The built-in Ollama and llama-server backends set it themselves: `True`
when pointed at a non-loopback URL.
