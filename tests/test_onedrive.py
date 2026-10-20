"""OneDrive adapter tests. No network and no real sign-in: MSAL and Graph are faked.
Includes the read-only guards (no write scope, no write HTTP method) and a no-hard-coded-GUID check."""
import ast
import json
import pathlib
import re
import shutil
import subprocess
import sys
import types
import typing

import pytest
import yaml

from plm_kit import cli, onedrive

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "plm_kit" / "onedrive.py"
# Fabricated at runtime so no GUID literal sits in the repo (the GUID scan below covers this file too).
FAKE_CLIENT = "-".join(["1" * 8, "2" * 4, "3" * 4, "4" * 4, "5" * 12])
FAKE_TENANT = "-".join(["a" * 8, "b" * 4, "c" * 4, "d" * 4, "e" * 12])
WRITE = "Read" + "Write"   # built at runtime, see test_no_write_scope_anywhere_in_code_or_config
GRAPH = onedrive.GRAPH


# ---------- fakes ----------
class FakeCache:
    def __init__(self):
        self.data, self.has_state_changed = "{}", False

    def deserialize(self, s):
        self.data = s

    def serialize(self):
        return self.data


class FakeApp:
    log: typing.ClassVar[list] = []
    token_scope = "Files.Read"
    signed_in = False

    def __init__(self, client_id, authority=None, token_cache=None, **kw):
        FakeApp.log.append(("init", client_id, authority, sorted(kw)))
        self.cache = token_cache

    def get_accounts(self):
        return [{"username": "user"}] if FakeApp.signed_in else []

    def _ok(self, scopes):
        FakeApp.log.append(("scopes", tuple(scopes)))
        FakeApp.signed_in = True
        self.cache.has_state_changed, self.cache.data = True, '{"fake": "cache"}'
        return {"access_token": "tok", "scope": FakeApp.token_scope}

    def acquire_token_silent(self, scopes, account=None):
        return self._ok(scopes)

    def initiate_device_flow(self, scopes=None):
        FakeApp.log.append(("device", tuple(scopes)))
        return {"user_code": "ABCD", "message": "To sign in, open the device login page and enter ABCD"}

    def acquire_token_by_device_flow(self, flow):
        return self._ok(list(onedrive.SCOPES))

    def acquire_token_interactive(self, scopes=None, **kw):
        return self._ok(scopes)

    def remove_account(self, a):
        FakeApp.signed_in = False


@pytest.fixture
def fake_msal(monkeypatch):
    mod = types.ModuleType("msal")
    mod.PublicClientApplication = FakeApp
    mod.SerializableTokenCache = FakeCache
    monkeypatch.setitem(sys.modules, "msal", mod)
    FakeApp.log, FakeApp.signed_in, FakeApp.token_scope = [], False, "Files.Read"
    return FakeApp


class Resp:
    def __init__(self, status=200, body=None, content=b"", headers=None):
        self.status_code, self._body, self.content, self.headers = status, body, content, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")


class FakeGraph:
    """In-memory drive: {path: bytes}; folders implied. Records every request (always via GET)."""

    def __init__(self, files):
        self.files, self.calls, self.etag = dict(files), [], {}

    def _items(self, folder):
        kids = {}
        for p, data in self.files.items():
            if not p.startswith(folder + "/"):
                continue
            rest = p[len(folder) + 1:]
            name = rest.split("/")[0]
            if "/" in rest:
                kids[name] = {"id": f"dir:{folder}/{name}", "name": name, "folder": {"childCount": 1}}
            else:
                kids[name] = {"id": f"file:{p}", "name": name, "size": len(data), "file": {},
                              "eTag": self.etag.get(p, "v1")}
        return list(kids.values())

    def __call__(self, url, headers, timeout):
        self.calls.append(("GET", url, headers.get("Authorization")))
        assert headers["Authorization"] == "Bearer tok"
        path = url.split("/v1.0/me/drive/", 1)[1]
        if path.startswith("root:/"):
            folder = re.sub(r"%2F", "/", path[len("root:/"):].split(":/children")[0])
            folder = folder.replace("%20", " ")
            if not any(p.startswith(folder + "/") for p in self.files):
                return Resp(404)
            return Resp(body={"value": self._items(folder)})
        item = re.sub(r"%3A", ":", path.split("/")[1]).replace("%2F", "/").replace("%20", " ")
        if path.endswith("/children?$select=" + onedrive.SELECT):
            return Resp(body={"value": self._items(item.split(":", 1)[1])})
        if path.endswith("/content"):
            return Resp(content=self.files[item.split(":", 1)[1]])
        return Resp(404)


def settings(tmp_path, **over):
    od = {"enabled": True, "client_id": FAKE_CLIENT, "tenant": "consumers", "folders": ["Notes"],
          "token_cache": str(tmp_path / "tc" / "cache.json"), "sync_dir": str(tmp_path / "sync")}
    od.update(over)
    return {"sources": {"onedrive": od}}


# ---------- settings ----------
def test_nothing_baked_in_default_config():
    od = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())["sources"]["onedrive"]
    assert od["enabled"] is False and od["client_id"] is None and od["tenant"] is None


def test_settings_from_config_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv(onedrive.ENV_CLIENT, raising=False)
    monkeypatch.delenv(onedrive.ENV_TENANT, raising=False)
    s = onedrive.settings(settings(tmp_path))
    assert s["client_id"] == FAKE_CLIENT and s["tenant"] == "consumers" and s["folders"] == ["Notes"]
    env = {onedrive.ENV_CLIENT: FAKE_CLIENT, onedrive.ENV_TENANT: FAKE_TENANT}
    s = onedrive.settings(settings(tmp_path, client_id=None, tenant=None), env=env)
    assert s["tenant"] == FAKE_TENANT
    for t in ("common", "organizations", "contoso.onmicrosoft.com"):
        assert onedrive.settings(settings(tmp_path, tenant=t), env={})["tenant"] == t


@pytest.mark.parametrize("over", [{"client_id": None}, {"client_id": "my-app"}, {"tenant": None},
                                  {"tenant": "not a tenant!"}, {"folders": ["../etc"]}, {"folders": ["/"]}])
def test_settings_rejects_missing_or_bad_values(tmp_path, over):
    with pytest.raises(ValueError):
        onedrive.settings(settings(tmp_path, **over), env={})


def test_token_cache_and_sync_dir_outside_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("PLM_KIT_TOKEN_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    p = onedrive.token_cache_path()
    if sys.platform not in ("win32", "darwin"):
        assert p == tmp_path / "xdg" / "plm-kit" / "onedrive-msal-cache.json"
    assert ROOT not in p.parents
    with pytest.raises(ValueError):
        onedrive.token_cache_path(ROOT / "tokens.json")
    with pytest.raises(ValueError):
        onedrive.sync_dir({"sync_dir": str(ROOT / "sync")})


# ---------- sign-in (MSAL faked) ----------
def test_device_code_login_requests_only_files_read_and_caches_0600(tmp_path, fake_msal, capsys):
    s = onedrive.settings(settings(tmp_path), env={})
    sess = onedrive.Session(s)
    assert sess.token(out=sys.stdout) == "tok"
    assert "enter ABCD" in capsys.readouterr().out
    init = next(x for x in fake_msal.log if x[0] == "init")
    assert init[1] == FAKE_CLIENT and init[2] == "https://login.microsoftonline.com/consumers"
    assert {x[1] for x in fake_msal.log if x[0] in ("scopes", "device")} == {("Files.Read",)}
    assert sess.cache_path.stat().st_mode & 0o777 == 0o600
    assert onedrive.Session(s).token(interactive_ok=False) == "tok"   # silent from cache next time


def test_interactive_login_and_logout(tmp_path, fake_msal):
    s = onedrive.settings(settings(tmp_path, auth="interactive"), env={})
    sess = onedrive.Session(s)
    sess.token()
    assert not any(x[0] == "device" for x in fake_msal.log)
    sess.logout()
    assert not sess.cache_path.exists()


def test_not_signed_in_without_interactive_raises(tmp_path, fake_msal):
    sess = onedrive.Session(onedrive.settings(settings(tmp_path), env={}))
    with pytest.raises(RuntimeError, match="onedrive login"):
        sess.token(interactive_ok=False)


def test_token_with_write_scope_is_refused(tmp_path, fake_msal):
    fake_msal.token_scope = f"Files.{WRITE} openid profile"
    with pytest.raises(RuntimeError, match="write scope"):
        onedrive.Session(onedrive.settings(settings(tmp_path), env={})).token()


# ---------- Graph (faked) + sync ----------
def test_sync_mirrors_selected_folder_get_only(tmp_path):
    g = FakeGraph({"Notes/a.md": b"# A\nalpha", "Notes/sub/b.txt": b"beta", "Notes/pic.png": b"x",
                   "Notes/big.md": b"x" * 2048, "Private/secret.md": b"nope"})
    s = onedrive.settings(settings(tmp_path, max_file_mb=0.001), env={})
    dest = tmp_path / "sync"
    st = onedrive.sync(onedrive.GraphReader(lambda: "tok", get=g), s, dest)
    assert st == {"downloaded": 2, "unchanged": 0, "skipped": 2, "removed": 0}
    assert (dest / "Notes/a.md").read_bytes() == b"# A\nalpha" and (dest / "Notes/sub/b.txt").is_file()
    assert not (dest / "Private").exists()                      # only selected folders
    assert {c[0] for c in g.calls} == {"GET"}
    assert all(c[1].startswith(f"{GRAPH}/me/drive/") for c in g.calls)
    # second run: unchanged by eTag; remote delete -> local delete; changed eTag -> re-download
    del g.files["Notes/sub/b.txt"]
    g.files["Notes/a.md"], g.etag["Notes/a.md"] = b"# A\nalpha v2", "v2"
    st = onedrive.sync(onedrive.GraphReader(lambda: "tok", get=g), s, dest)
    assert st["removed"] == 1 and st["downloaded"] == 1 and not (dest / "Notes/sub/b.txt").exists()
    assert (dest / "Notes/a.md").read_bytes().endswith(b"v2")
    assert (dest / onedrive.STATE).stat().st_mode & 0o777 == 0o600


def test_reader_refuses_urls_outside_graph_drive():
    r = onedrive.GraphReader(lambda: "tok", get=lambda *a: pytest.fail("token sent"))
    for url in ("https://evil.example/me/drive/x", f"{GRAPH}/me/messages", f"{GRAPH}/sites/x/drive/root"):
        with pytest.raises(ValueError):
            r._get(url)


def test_next_link_outside_graph_is_refused():
    pages = {"first": {"value": [], "@odata.nextLink": "https://elsewhere.example/page2"}}
    r = onedrive.GraphReader(lambda: "tok", get=lambda url, h, t: Resp(body=pages["first"]))
    with pytest.raises(ValueError):
        list(r.children(folder="Notes"))


def test_throttling_retries_then_succeeds():
    seq = [Resp(429, headers={"Retry-After": "1"}), Resp(body={"value": []})]
    slept = []
    r = onedrive.GraphReader(lambda: "tok", get=lambda *a: seq.pop(0), sleep=slept.append)
    assert list(r.children(folder="Notes")) == [] and slept == [1.0]


def test_unsafe_remote_names_are_skipped(tmp_path):
    class G(FakeGraph):
        def _items(self, folder):
            return [{"id": "x1", "name": "../escape.md", "file": {}, "size": 1, "eTag": "1"},
                    {"id": "x2", "name": "..", "folder": {}}]

    g = G({"Notes/x.md": b"x"})
    st = onedrive.sync(onedrive.GraphReader(lambda: "tok", get=g), onedrive.settings(settings(tmp_path), env={}),
                       tmp_path / "sync")
    assert st["downloaded"] == 0 and not (tmp_path / "escape.md").exists()


def test_cli_sync_feeds_the_local_index(tmp_path, fake_msal, monkeypatch, capsys):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv(onedrive.ENV_CLIENT, raising=False)
    monkeypatch.delenv(onedrive.ENV_TENANT, raising=False)
    g = FakeGraph({"Notes/zephyr.md": b"# Zephyr kite\nThe Zephyr kite needs a 12 m line and light wind."})
    monkeypatch.setattr(onedrive, "http_get", g)
    c = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    c["sources"] = settings(tmp_path)["sources"]
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(c))
    assert cli.main(["--config", str(f), "onedrive", "login"]) == 0
    assert cli.main(["--config", str(f), "onedrive", "sync"]) == 0
    out = capsys.readouterr().out
    rep = json.loads(out[out.index("{"):])
    assert rep["sync"]["downloaded"] == 1 and "onedrive/" in rep["index"]["sources"]
    assert cli.main(["--config", str(f), "ask", "Zephyr kite line length"]) == 0
    assert "onedrive/Notes/zephyr.md" in capsys.readouterr().out
    assert cli.main(["--config", str(f), "onedrive", "status"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["scopes"] == ["Files.Read"] and FAKE_CLIENT not in json.dumps(st)
    assert {c[0] for c in g.calls} == {"GET"}


def test_cli_onedrive_without_settings_fails_cleanly(monkeypatch, capsys):
    monkeypatch.delenv(onedrive.ENV_CLIENT, raising=False)
    assert cli.main(["onedrive", "status"]) == 2
    assert "client id" in capsys.readouterr().err


# ---------- read-only guards ----------
WRITE_SCOPE = re.compile(r"\b[A-Z][A-Za-z]+\.(Read" r"Write|Write|Manage|FullControl)(\.[A-Za-z]+)*\b")
WRITE_METHODS = {"put", "post", "patch", "delete"}


def _code_and_config_files():
    files = list((ROOT / "plm_kit").rglob("*.py")) + list((ROOT / "scripts").glob("*"))
    files += list((ROOT / "config").glob("*")) + [ROOT / "config.example.toml", ROOT / ".env.example"]
    return [f for f in files if f.is_file()]


def test_scopes_constant_is_files_read_only():
    assert onedrive.SCOPES == ("Files.Read",)


def test_no_write_scope_anywhere_in_code_or_config():
    hits = [f"{f.relative_to(ROOT)}: {m.group(0)}" for f in _code_and_config_files()
            for m in WRITE_SCOPE.finditer(f.read_text(errors="replace"))]
    assert hits == [], hits
    assert WRITE_SCOPE.search(f"Files.{WRITE}") and WRITE_SCOPE.search(f"Sites.{WRITE}.All")  # the guard works


def test_no_write_http_method_in_adapter():
    tree = ast.parse(SRC.read_text())
    bad = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr.lower() in WRITE_METHODS | {"request", "send", "urlopen"}):
            bad.append(f"line {node.lineno}: .{node.func.attr}()")
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and re.fullmatch(r"(?i)put|post|patch|delete", node.value.strip())):
            bad.append(f"line {node.lineno}: method string {node.value!r}")
        if isinstance(node, ast.keyword) and node.arg == "method":
            bad.append(f"line {node.lineno}: method= keyword")
    assert bad == [], bad
    assert not WRITE_METHODS & {m.lower() for m in dir(onedrive.GraphReader)}
    # the only HTTP call: requests.get
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "requests"}
    assert calls == {"get"}


def test_no_write_method_to_drive_endpoints_in_repo():
    pat = re.compile(r"(?i)\b(PUT|POST|PATCH|DELETE)\b[^\n]{0,80}(/me/drive|/drives?/|graph\.microsoft\.com)")
    hits = [str(f.relative_to(ROOT)) for f in _code_and_config_files() if pat.search(f.read_text(errors="replace"))]
    assert hits == []


GUID = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")


def test_no_guid_like_client_or_tenant_id_hard_coded():
    """gitleaks-style scan of every tracked + to-be-committed text file for GUIDs (client/tenant ids)."""
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                       capture_output=True, text=True, check=False) if shutil.which("git") else None
    if r is not None and r.returncode == 0:
        files = r.stdout.split()
    else:   # not a git checkout (e.g. an unpacked release): scan every file except virtualenvs and caches
        skip = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
        files = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                 if p.is_file() and not skip.intersection(p.relative_to(ROOT).parts)]
    hits = []
    for rel in files:
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        hits += [f"{rel}:{text.count(chr(10), 0, m.start()) + 1}" for m in GUID.finditer(text)]
    assert hits == [], hits
    assert GUID.search(FAKE_CLIENT)  # the scanner would catch one


@pytest.mark.skipif(not shutil.which("gitleaks"), reason="gitleaks not installed")
def test_gitleaks_rule_catches_hard_coded_entra_ids(tmp_path):
    (tmp_path / "leak.toml").write_text(f'client_id = "{FAKE_CLIENT}"\ntenant = "{FAKE_TENANT}"\n')
    r = subprocess.run(["gitleaks", "dir", str(tmp_path), "--config", str(ROOT / ".gitleaks.toml"), "--no-banner",
                        "--report-format", "json", "--report-path", str(tmp_path / "r.json")],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 1
    rules = {x["RuleID"] for x in json.loads((tmp_path / "r.json").read_text())}
    assert "entra-client-or-tenant-id" in rules


def test_pinned_msal_api_matches_adapter():
    msal = pytest.importorskip("msal")
    import inspect

    assert msal.__version__ == "1.39.0"
    pca = msal.PublicClientApplication
    for name in ("initiate_device_flow", "acquire_token_by_device_flow", "acquire_token_silent",
                 "acquire_token_interactive", "get_accounts", "remove_account"):
        assert callable(getattr(pca, name)), name
    assert "token_cache" in inspect.signature(msal.ClientApplication.__init__).parameters
    tc = msal.SerializableTokenCache()
    assert hasattr(tc, "has_state_changed") and callable(tc.serialize) and callable(tc.deserialize)
