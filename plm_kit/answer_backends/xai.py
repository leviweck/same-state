"""xAI (Grok) answer backend: generation.backend: xai. OFF by default; it sends your notes off this computer.

To use it, ALL of these are required:
  1. generation.backend: xai and generation.allow_off_machine: true (core refuses otherwise, plm_kit/answer.py).
  2. A one-time consent on this computer: answer `yes` at the prompt on first use, or run
     `python -m plm_kit.answer_backends.xai --consent`. Stored in <user cache>/consent/xai-v1.json.
  3. Your own API key in the environment variable XAI_API_KEY (never in a config file; never logged).

What is sent: the question, the retrieved excerpts of your notes and (if voice lite is on) your style examples, to
https://api.x.ai/v1/responses (or the US regional endpoint). Requests use the Responses API with store: false, so no
conversation state is kept for later continuation. xAI's own retention policy still applies; zero data retention is
a team-wide setting in the xAI Console, not an API parameter (see PRIVACY.md).

Cost cap: before each request the backend computes an estimate from the prompt size (generation.xai.chars_per_token,
deliberately pessimistic), max_tokens and generation.xai.reasoning_allowance_tokens, priced with generation.xai.prices
(visible output and reasoning both at the output rate). After the response the real cost is xAI's reported
usage.cost_in_usd_ticks when present (the amount billed); otherwise input_tokens + output_tokens + reasoning_tokens
at the configured prices (reasoning always added, because the docs don't say whether output_tokens includes it). max_tokens is sent as max_output_tokens, which per xAI's API
reference limits VISIBLE output only; reasoning tokens are billed on top and xAI documents no upper bound for them, so
the allowance is an estimate, not a guarantee. If a response costs more than the estimate, the real cost is still
added to the run total and the ledger, and the overshoot is recorded (Answer.meta, a stderr warning, ledger count).
If what this run already spent plus that estimate would exceed generation.xai.max_cost_usd (per run: one process,
e.g. one `ask` or one whole `selftest --answer`, shared by every backend instance it creates), or today's /
this month's spend on this computer plus that estimate would exceed generation.xai.max_cost_usd_per_day /
max_cost_usd_per_month (persistent ledger, SpendLedger), it refuses BEFORE sending. After each response the real
cost from the reported token usage replaces the reservation.
Transport: a direct HTTPS connection; proxy environment variables are ignored. Ledger updates are locked across
processes (fcntl on macOS/Linux, msvcrt on Windows; no lock available = refuse to send) and written atomically.
Region: the US regional endpoint serves only grok-4.7 and grok-4.6; other models with region: us are rejected.
Prices change: check https://docs.x.ai/developers/pricing and update generation.xai.prices."""
import contextlib
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

from plm_kit.answer import Answer, AnswerBackend, register

ENDPOINTS = {"global": "https://api.x.ai/v1", "us": "https://us.api.x.ai/v1"}
REGION_MULTIPLIER = {"global": 1.0, "us": 1.1}          # US regional endpoint bills 1.1x (xAI pricing page)
# The US regional endpoint serves only these models; anything else gets HTTP 404 there.
# https://docs.x.ai/developers/advanced-api-usage/regions (checked 2026-10-19). Update when xAI adds models.
REGION_MODELS = {"us": ("grok-4.7", "grok-4.6")}
LONG_CONTEXT_TOKENS = 200_000                            # long-context rates start here; the kit refuses instead
EFFORTS = (None, "none", "low", "medium", "high", "xhigh")
CONSENT_VERSION = "xai-v1"
KEY_ENV = "XAI_API_KEY"
NULL_MEANS_MODEL_DEFAULT = ("reasoning_effort", "temperature", "max_cost_usd_per_month")  # null/"" = no value
DEFAULTS = {
    "model": "grok-4.3",
    "region": "global",
    "reasoning_effort": "low",
    "temperature": None,
    "timeout_s": 120,
    "max_cost_usd": 1.0,
    "max_cost_usd_per_day": 2.0,
    "max_cost_usd_per_month": None,
    "chars_per_token": 3.0,
    # INFERENCE: xAI's docs give no bound on reasoning tokens (max_output_tokens covers visible output only). 4096 is a
    # conservative allowance for reasoning_effort "low" on a short notes question; raise it for medium/high/xhigh.
    "reasoning_allowance_tokens": 4096,
    "prices": {"grok-4.3": {"input": 1.25, "output": 2.50}, "grok-4.6": {"input": 2.00, "output": 6.00},
               "grok-4.7": {"input": 2.00, "output": 6.00}},
}
NOTICE = """\
plm-kit: xAI answer backend, one-time consent
With generation.backend: xai, each `ask --answer` sends your question, excerpts of your notes and (if voice lite is
on) your style examples to xAI's API over the internet. xAI keeps API requests for a period under its own policy
(see PRIVACY.md); zero data retention is a team setting in the xAI Console. You pay xAI per token with your own key.
Excerpts-only mode (`plm-kit ask` without --answer) and the local backends never send anything.
Type 'yes' to allow this on this computer: """


# Per-run spend: one total per process, shared by every XaiBackend instance (answer_local builds a new backend per
# answer, so a per-instance total would really be per answer). Each request reserves its worst case here before
# sending and is corrected to the real cost afterwards; a failed request keeps its reservation.
_RUN_LOCK = threading.Lock()
_RUN = {"spent_usd": 0.0}


def run_spent_usd():
    with _RUN_LOCK:
        return _RUN["spent_usd"]


def reset_run_spend():
    """Reset this process's per-run total (tests; a long-lived host starting a new logical run)."""
    with _RUN_LOCK:
        _RUN["spent_usd"] = 0.0


def _reserve_run(worst, cap, reserve=True):
    with _RUN_LOCK:
        spent = _RUN["spent_usd"]
        if spent + worst > cap:
            raise RuntimeError(f"cost cap: this request could cost up to ${worst:.4f}; this run has spent "
                               f"${spent:.4f} of generation.xai.max_cost_usd ${cap:.2f}. Nothing was sent.")
        if reserve:
            _RUN["spent_usd"] = spent + worst


def _adjust_run(delta):
    with _RUN_LOCK:
        _RUN["spent_usd"] = max(0.0, _RUN["spent_usd"] + delta)


def settings(backend_cfg):
    s = dict(DEFAULTS)
    s.update({k: v for k, v in (backend_cfg or {}).items() if v is not None or k in NULL_MEANS_MODEL_DEFAULT})
    if s["region"] not in ENDPOINTS:
        raise ValueError(f"generation.xai.region must be one of {sorted(ENDPOINTS)}")
    if s["reasoning_effort"] not in EFFORTS:
        raise ValueError(f"generation.xai.reasoning_effort must be one of {[e for e in EFFORTS if e]} or null")
    allowed = REGION_MODELS.get(s["region"])
    if allowed and s["model"] not in allowed:
        raise ValueError(f"generation.xai.model {s['model']!r} isn't served on the {s['region']!r} regional endpoint "
                         f"(it serves only {', '.join(allowed)}). Pick one of those models, or set region: global. "
                         "See https://docs.x.ai/developers/advanced-api-usage/regions")
    if float(s["max_cost_usd"]) <= 0:
        raise ValueError("generation.xai.max_cost_usd must be > 0")
    if float(s["max_cost_usd_per_day"]) <= 0:
        raise ValueError("generation.xai.max_cost_usd_per_day must be > 0")
    if s["max_cost_usd_per_month"] is not None and float(s["max_cost_usd_per_month"]) <= 0:
        raise ValueError("generation.xai.max_cost_usd_per_month must be > 0 or null")
    if int(s["reasoning_allowance_tokens"]) < 0:
        raise ValueError("generation.xai.reasoning_allowance_tokens must be >= 0 (reasoning isn't limited by "
                         "max_output_tokens, so this budgets for it in the pre-send estimate)")
    if not 1.0 <= float(s["chars_per_token"]) <= 8.0:
        raise ValueError("generation.xai.chars_per_token must be between 1 and 8 (lower = more cautious)")
    price = (s["prices"] or {}).get(s["model"])
    if not price or float(price.get("input", 0)) <= 0 or float(price.get("output", 0)) <= 0:
        raise ValueError(f"generation.xai.prices has no input/output price (USD per 1M tokens) for model "
                         f"{s['model']!r}; add it from https://docs.x.ai/developers/pricing")
    return s


class SpendLedger:
    """Daily/monthly xAI spend on this computer: <user cache>/spend/xai-ledger.json = {"days": {"YYYY-MM-DD": usd}}.
    Only local dates and dollar amounts: no prompts, answers, questions or keys. Each request reserves its worst case
    BEFORE sending and is corrected to the real cost after the response, so a crash or a failed request never
    undercounts (a failed request keeps its worst-case reservation). An unreadable ledger blocks sending."""
    KEEP_DAYS = 400

    def __init__(self, cache=None, today=None):
        from plm_kit import paths

        self.path = paths.cache_dir(cache) / "spend" / "xai-ledger.json"
        self._today = today or (lambda: time.strftime("%Y-%m-%d"))

    def _load(self):
        return self._load_all()[0]

    def _load_all(self):
        """(days, overshoots): overshoots = {"YYYY-MM-DD": count of responses that cost more than their estimate}."""
        if not self.path.exists():
            return {}, {}
        try:
            data = json.loads(self.path.read_text())
            days, over = data.get("days"), data.get("overshoots") or {}
            if not isinstance(days, dict) or not isinstance(over, dict):
                raise TypeError
            return {str(k): float(v) for k, v in days.items()}, {str(k): int(v) for k, v in over.items()}
        except (OSError, ValueError, TypeError, AttributeError):
            raise RuntimeError(f"can't read the xAI spend ledger {self.path}; nothing was sent. Check or delete it "
                               "(deleting it resets the daily/monthly totals)") from None

    def _save(self, days, overshoots=None):
        """Atomic write: a unique temp file in the same folder, flushed, then os.replace over the ledger."""
        keep = sorted(days)[-self.KEEP_DAYS:]
        data = {"days": {d: round(days[d], 8) for d in keep}}
        over = {d: n for d, n in (overshoots or {}).items() if d in keep or d == self._today()}
        if over:
            data["overshoots"] = over
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".xai-ledger.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(data, indent=1, sort_keys=True))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    LOCK_TIMEOUT_S = 60
    LOCK_POLL_S = 0.05

    @property
    def lock_path(self):
        """Empty file next to the ledger, used only as the lock handle (no contents are ever written)."""
        return self.path.with_name(self.path.name + ".lock")

    @contextlib.contextmanager
    def _locked(self, after_send=False):
        """Exclusive lock across processes for read-modify-write: fcntl.flock (non-blocking, polled) on macOS/Linux,
        msvcrt.locking on Windows, both with the same LOCK_TIMEOUT_S deadline. If neither is available, or the lock
        can't be taken in time, it fails CLOSED with a RuntimeError instead of updating the ledger unlocked.

        Before sending (reserve) that means nothing is sent. After the response (the correction in add()), the
        request has ALREADY been sent and billed by xAI: the user gets an error, and the ledger keeps the reserved
        estimate for that request instead of the real cost. That errs high unless the answer overshot its estimate
        (then the overshoot isn't recorded in the ledger); the run total already holds the real cost."""
        when = ("this answer was already sent and billed by xAI; the ledger keeps its reserved estimate"
                if after_send else "nothing was sent")
        kind, mod = _lock_impl()
        if kind is None:
            raise RuntimeError(f"can't lock the xAI spend ledger on this system (no fcntl or msvcrt); {when}. "
                               "The xai backend needs a working file lock to keep the daily/monthly caps safe.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a+b") as lf:
            fd = lf.fileno()
            deadline = time.monotonic() + self.LOCK_TIMEOUT_S
            while True:
                try:
                    if kind == "fcntl":
                        mod.flock(fd, mod.LOCK_EX | mod.LOCK_NB)
                    else:
                        lf.seek(0)
                        mod.locking(fd, mod.LK_LOCK, 1)        # LK_LOCK itself retries for ~10 s
                    break
                except OSError:                                 # includes BlockingIOError (lock held elsewhere)
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"timed out locking the xAI spend ledger ({self.lock_path}); {when}. "
                                           "Another plm-kit process may be stuck.") from None
                    if kind == "fcntl":
                        time.sleep(self.LOCK_POLL_S)
            try:
                yield
            finally:
                if kind == "fcntl":
                    mod.flock(fd, mod.LOCK_UN)
                else:
                    lf.seek(0)
                    mod.locking(fd, mod.LK_UNLCK, 1)

    def totals(self):
        days, today = self._load(), self._today()
        return days.get(today, 0.0), sum(v for d, v in days.items() if d[:7] == today[:7])

    def check(self, worst, per_day, per_month, totals=None):
        day, month = totals or self.totals()
        if day + worst > per_day:
            raise RuntimeError(f"daily cost cap: this request could cost up to ${worst:.4f}; today's xAI spend on this "
                               f"computer is ${day:.4f} of generation.xai.max_cost_usd_per_day ${per_day:.2f}. "
                               "Nothing was sent.")
        if per_month is not None and month + worst > per_month:
            raise RuntimeError(f"monthly cost cap: this request could cost up to ${worst:.4f}; this month's xAI spend "
                               f"is ${month:.4f} of generation.xai.max_cost_usd_per_month ${per_month:.2f}. "
                               "Nothing was sent.")

    def reserve(self, worst, per_day, per_month):
        """Check the caps and add the estimate in one locked read-modify-write (no race between processes)."""
        with self._locked():
            (days, over), today = self._load_all(), self._today()
            day = days.get(today, 0.0)
            self.check(worst, per_day, per_month, (day, sum(v for d, v in days.items() if d[:7] == today[:7])))
            days[today] = day + float(worst)
            self._save(days, over)

    def add(self, usd, overshoot=False):
        """Correct a reservation after the response (see _locked for what happens if the lock can't be taken)."""
        with self._locked(after_send=True):
            (days, over), today = self._load_all(), self._today()
            days[today] = max(0.0, days.get(today, 0.0) + float(usd))
            if overshoot:
                over[today] = over.get(today, 0) + 1
            self._save(days, over)

    def overshoots(self):
        return self._load_all()[1].get(self._today(), 0)


def _lock_impl():
    """("fcntl", module) on POSIX, ("msvcrt", module) on Windows, (None, None) if neither exists."""
    try:
        import fcntl

        return "fcntl", fcntl
    except ImportError:
        pass
    try:
        import msvcrt

        return "msvcrt", msvcrt
    except ImportError:
        return None, None


def consent_file(cache=None):
    from plm_kit import paths

    return paths.cache_dir(cache) / "consent" / f"{CONSENT_VERSION}.json"


def give_consent(cache=None):
    f = consent_file(cache)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"consent": CONSENT_VERSION, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
    return f


def has_consent(cache=None):
    f = consent_file(cache)
    try:
        return json.loads(f.read_text()).get("consent") == CONSENT_VERSION
    except (OSError, ValueError):
        return False


def _ask_consent(cache, stdin=None, out=None):
    stdin, out = stdin or sys.stdin, out or sys.stderr
    if not stdin.isatty():
        raise RuntimeError("the xai backend needs a one-time consent: run `python -m plm_kit.answer_backends.xai "
                           "--consent` (or answer the prompt in an interactive terminal)")
    out.write(NOTICE)
    out.flush()
    if stdin.readline().strip().lower() != "yes":
        raise RuntimeError("xai backend not allowed (consent not given); nothing was sent")
    give_consent(cache)


def _opener():
    """Direct connections only: an empty ProxyHandler means HTTPS_PROXY / HTTP_PROXY / ALL_PROXY are ignored, so the
    request (with your key and notes) goes straight to xAI and never through a proxy picked up from the environment."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post_json(url, payload, headers, timeout):
    """Default transport: one HTTPS POST to a fixed xAI endpoint, no proxy. Every failure becomes a RuntimeError
    whose message never includes the key or the prompt."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with _opener().open(req, timeout=timeout) as r:
            status, raw = getattr(r, "status", 200), r.read()
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode()).get("error")
        except Exception:  # noqa: BLE001
            msg = None
        hint = " (check XAI_API_KEY)" if e.code in (401, 403) else ""
        hint = hint or (" (prepaid credit used up, or rate limited)" if e.code in (402, 429) else "")
        raise RuntimeError(f"xAI API returned HTTP {e.code}{hint}" + (f": {str(msg)[:200]}" if msg else "")) from None
    except (urllib.error.URLError, TimeoutError) as e:
        reason = getattr(e, "reason", e)
        detail = getattr(reason, "strerror", None) or str(reason)
        raise RuntimeError(f"could not reach the xAI API: {type(reason).__name__}: {detail}"[:300]) from None
    except OSError as e:                                   # ConnectionResetError, BrokenPipeError, ssl errors, ...
        raise RuntimeError(f"connection to the xAI API failed: {type(e).__name__}"
                           + (f": {e.strerror}" if getattr(e, "strerror", None) else "")) from None
    try:
        resp = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise RuntimeError(f"xAI API returned HTTP {status} with a body that isn't JSON ({len(raw)} bytes); "
                           "nothing was parsed") from None
    if not isinstance(resp, dict):
        raise RuntimeError(  # noqa: TRY004 - backend contract: RuntimeError for runtime failures (plm_kit/answer.py)
            f"xAI API returned HTTP {status} with unexpected JSON ({type(resp).__name__})")
    return resp


def _text_of(resp):
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    parts = []
    for item in resp.get("output") or []:
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                    parts.append(c["text"])
    return "".join(parts)


@register("xai")
class XaiBackend(AnswerBackend):
    sends_off_machine = True

    def __init__(self, gen_cfg, *, cache=None, post=None, stdin=None, today=None):
        super().__init__(gen_cfg, cache=cache)
        if not self.gen_cfg.get("allow_off_machine"):        # also enforced by answer.get_backend; never bypassed
            raise ValueError("backend 'xai' sends your notes off this computer; set generation.allow_off_machine: "
                             "true to allow it (see PRIVACY.md)")
        self.s = settings(self.backend_cfg)
        self.price = {k: float(v) * REGION_MULTIPLIER[self.s["region"]]
                      for k, v in self.s["prices"][self.s["model"]].items() if k in ("input", "output")}
        self.url = ENDPOINTS[self.s["region"]] + "/responses"
        self.post = post or _post_json
        self.stdin = stdin
        self.spent_usd = 0.0                       # this instance only (reporting); the cap uses the run total
        self.requests = 0
        self.ledger = SpendLedger(cache, today=today)

    def cost(self, input_tokens, output_tokens):
        return (input_tokens * self.price["input"] + output_tokens * self.price["output"]) / 1e6

    def estimate_input_tokens(self, messages):
        chars = sum(len(str(m.get("content", ""))) + 16 for m in messages)
        return int(chars / float(self.s["chars_per_token"])) + 1

    def generate(self, messages, *, max_tokens, temperature=None, stop=None) -> Answer:
        max_tokens = int(max_tokens)
        if max_tokens < 16:
            raise ValueError("max_tokens must be >= 16")
        est_in = self.estimate_input_tokens(messages)
        if est_in >= LONG_CONTEXT_TOKENS:
            raise RuntimeError(f"prompt too long for the kit's price table (~{est_in} tokens >= {LONG_CONTEXT_TOKENS})")
        allowance = int(self.s["reasoning_allowance_tokens"])
        worst = self.cost(est_in, max_tokens + allowance)   # pre-send estimate: input + visible output + reasoning
        cap = float(self.s["max_cost_usd"])
        _reserve_run(worst, cap, reserve=False)     # early, clear refusal; checked again atomically before sending
        per_month = self.s["max_cost_usd_per_month"]
        self.ledger.check(worst, float(self.s["max_cost_usd_per_day"]), None if per_month is None else float(per_month))
        if not has_consent(self.cache):
            _ask_consent(self.cache, self.stdin)
        key = os.environ.get(KEY_ENV, "").strip()
        if not key:
            raise ValueError(f"set your xAI API key in the environment variable {KEY_ENV} (never in a config file)")
        payload = {"model": self.s["model"], "input": [{"role": m["role"], "content": str(m["content"])}
                                                        for m in messages],
                   "max_output_tokens": max_tokens, "store": False}
        temp = temperature if temperature is not None else self.s["temperature"]
        if temp is not None:
            payload["temperature"] = float(temp)
        if self.s["reasoning_effort"]:
            payload["reasoning"] = {"effort": self.s["reasoning_effort"]}
        t0 = time.perf_counter()
        _reserve_run(worst, cap)                   # run total: check + reserve atomically, before sending
        try:                                       # ledger: locked check + reserve of the worst case, before sending
            self.ledger.reserve(worst, float(self.s["max_cost_usd_per_day"]),
                                None if per_month is None else float(per_month))
        except BaseException:
            _adjust_run(-worst)                    # nothing was sent: release the run reservation
            raise
        resp = self.post(self.url, payload, {"Authorization": f"Bearer {key}"}, float(self.s["timeout_s"]))
        self.requests += 1
        usage = resp.get("usage") or {}
        tin = int(usage.get("input_tokens") or 0)
        tout = int(usage.get("output_tokens") or 0)
        reasoning = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
        # xAI's docs don't say clearly whether usage.output_tokens already includes reasoning_tokens (the field is
        # "Number of output tokens used"; the reference example has total = input + output + reasoning). To err high,
        # reasoning is always added on top of output_tokens. https://docs.x.ai/developers/rest-api-reference/inference/responses
        billed_out = tout + reasoning                                  # reasoning is billed as output
        if not usage:                                                  # no usage reported: assume the estimate
            tin, billed_out = est_in, max_tokens + allowance
        cost, cost_source = self.cost(tin, billed_out), "tokens" if usage else "estimate"
        ticks = usage.get("cost_in_usd_ticks")
        if isinstance(ticks, (int, float)) and not isinstance(ticks, bool) and ticks > 0:
            # "the actual amount billed" (https://docs.x.ai/developers/cost-tracking): preferred over the token math
            cost, cost_source = float(ticks) / 1e10, "cost_in_usd_ticks"
        overshoot = cost > worst + 1e-12
        self.spent_usd += cost
        _adjust_run(cost - worst)                  # run total ends at the REAL cost, even above the estimate
        self.ledger.add(cost - worst, overshoot=overshoot)   # same for the ledger; overshoots are counted
        if overshoot:
            sys.stderr.write(f"plm-kit: warning: this xAI answer cost ${cost:.4f}, more than the pre-send estimate "
                             f"${worst:.4f} (reasoning beyond generation.xai.reasoning_allowance_tokens={allowance}). "
                             "The real cost was recorded; consider raising the allowance.\n")
        text = _text_of(resp).strip()
        for s in stop or ():
            if s and s in text:
                text = text.split(s, 1)[0]
        if not text and resp.get("status") not in (None, "completed"):
            raise RuntimeError(f"xAI response status {resp.get('status')!r} with no text")
        return Answer(text, {"backend": "xai", "model": resp.get("model") or self.s["model"],
                             "usage": {"input_tokens": tin, "output_tokens": tout, "reasoning_tokens": reasoning},
                             "cost_usd": round(cost, 6), "cost_source": cost_source,
                             "estimate_usd": round(worst, 6),
                             "overshoot": overshoot, "overshoot_usd": round(max(0.0, cost - worst), 6),
                             "run_spent_usd": round(run_spent_usd(), 6),
                             "ms": round((time.perf_counter() - t0) * 1e3)})


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m plm_kit.answer_backends.xai")
    ap.add_argument("--consent", action="store_true", help="show the notice and record your consent")
    ap.add_argument("--revoke", action="store_true", help="delete the recorded consent")
    ap.add_argument("--cache")
    a = ap.parse_args(argv)
    if a.revoke:
        consent_file(a.cache).unlink(missing_ok=True)
        print("xai consent removed")
        return 0
    if a.consent:
        sys.stderr.write(NOTICE)
        sys.stderr.flush()
        if sys.stdin.readline().strip().lower() != "yes":
            print("not recorded")
            return 1
        print(f"recorded: {give_consent(a.cache)}")
        return 0
    print(f"consent recorded: {has_consent(a.cache)} ({consent_file(a.cache)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
