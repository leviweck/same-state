"""Optional, read-only OneDrive source via Microsoft Graph (delegated permissions).

- You bring your own Entra app registration (docs/ONEDRIVE.md). The client id and tenant come from your config
  or env at runtime; the kit contains none.
- Sign-in: MSAL device-code flow (default) or interactive browser sign-in. Scope: Files.Read only. MSAL adds
  openid/profile/offline_access itself (they are reserved scopes; passing them explicitly is an error).
- HTTP: GET only, to https://graph.microsoft.com/v1.0/me/drive/... This module has no code path that writes to
  OneDrive (tests/test_onedrive.py enforces that on the source).
- Tokens: cached in your OS user cache dir, outside the repo, file mode 0600. `plm-kit onedrive logout` deletes it.
- Sync copies selected folders' .md/.txt files into a local folder in your cache, which `plm-kit index` reads."""
import json
import os
import pathlib
import re
import sys
import time
from urllib.parse import quote

from plm_kit import paths

GRAPH = "https://graph.microsoft.com/v1.0"
SCOPES = ("Files.Read",)
AUTHORITY_HOST = "https://login.microsoftonline.com"
TENANT_ALIASES = ("common", "consumers", "organizations")
GUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
DOMAIN = re.compile(r"^(?=.{4,253}$)([a-zA-Z0-9-]{1,63}\.)+[a-zA-Z]{2,63}$")
ENV_CLIENT, ENV_TENANT = "PLM_KIT_ONEDRIVE_CLIENT_ID", "PLM_KIT_ONEDRIVE_TENANT"
SELECT = "id,name,size,file,folder,eTag,lastModifiedDateTime"
STATE = ".plm-kit-sync.json"
WRITE_SCOPE = re.compile(r"ReadWrite|\.Write|FullControl|Manage", re.IGNORECASE)


def settings(cfg, env=None):
    """Resolve the user's OneDrive settings. env (PLM_KIT_ONEDRIVE_*) beats config. Raises ValueError if unusable."""
    env = os.environ if env is None else env
    od = dict((cfg.get("sources") or {}).get("onedrive") or {})
    client_id = (env.get(ENV_CLIENT) or od.get("client_id") or "").strip()
    tenant = (env.get(ENV_TENANT) or od.get("tenant") or "").strip()
    if not client_id:
        raise ValueError(f"OneDrive: no client id. Set sources.onedrive.client_id or {ENV_CLIENT} to YOUR app "
                         "registration's Application (client) ID (docs/ONEDRIVE.md)")
    if not GUID.match(client_id):
        raise ValueError("OneDrive: client id must be the Application (client) ID GUID of your app registration")
    if not tenant:
        raise ValueError(f"OneDrive: no tenant. Set sources.onedrive.tenant or {ENV_TENANT} to consumers "
                         "(personal accounts), organizations, common, or your tenant id/domain")
    if tenant.lower() not in TENANT_ALIASES and not GUID.match(tenant) and not DOMAIN.match(tenant):
        raise ValueError("OneDrive: tenant must be consumers, organizations, common, a tenant id, or a domain")
    folders = [normalize_folder(f) for f in (od.get("folders") or [])]
    return {"client_id": client_id, "tenant": tenant, "auth": od.get("auth") or "device_code", "folders": folders,
            "extensions": tuple(e.lower() for e in (od.get("extensions") or (".md", ".txt"))),
            "max_bytes": int(float(od.get("max_file_mb") or 5) * 1024 * 1024),
            "token_cache": od.get("token_cache"), "sync_dir": od.get("sync_dir")}


def normalize_folder(f):
    parts = [p for p in str(f).replace("\\", "/").split("/") if p]
    if not parts or any(p in (".", "..") for p in parts):
        raise ValueError(f"OneDrive: bad folder path {f!r} (use a path from your OneDrive root, e.g. Notes/Recipes)")
    return "/".join(parts)


def _outside_repo(p, what):
    p = pathlib.Path(p).expanduser().resolve()
    if p == paths.REPO or paths.REPO in p.parents:
        raise ValueError(f"refusing to put the {what} inside the kit repo: {p}")
    return p


def os_cache_dir():
    """The OS user cache dir for plm-kit (not the repo): Windows %LOCALAPPDATA%, macOS ~/Library/Caches,
    else $XDG_CACHE_HOME or ~/.cache."""
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        base = pathlib.Path(os.environ["LOCALAPPDATA"])
    elif sys.platform == "darwin":
        base = pathlib.Path("~/Library/Caches").expanduser()
    else:
        base = pathlib.Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    return base / "plm-kit"


def token_cache_path(override=None):
    return _outside_repo(override or os.environ.get("PLM_KIT_TOKEN_CACHE") or os_cache_dir() / "onedrive-msal-cache.json",
                         "token cache")


def sync_dir(s, cache=None):
    return _outside_repo(s.get("sync_dir") or paths.cache_dir(cache) / "sources" / "onedrive", "OneDrive sync folder")


def _write_private(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(path, 0o600)


class Session:
    """MSAL public-client app + persisted token cache. msal is imported only when OneDrive is used."""

    def __init__(self, s):
        import msal

        self.s = s
        self.cache_path = token_cache_path(s.get("token_cache"))
        self.cache = msal.SerializableTokenCache()
        if self.cache_path.is_file():
            self.cache.deserialize(self.cache_path.read_text())
        self.app = msal.PublicClientApplication(s["client_id"], authority=f"{AUTHORITY_HOST}/{s['tenant']}",
                                                token_cache=self.cache)

    def _save(self):
        if self.cache.has_state_changed:
            _write_private(self.cache_path, self.cache.serialize())

    def token(self, interactive_ok=True, out=None):
        """Access token for Files.Read: silent from cache first, then device code / interactive if allowed."""
        out = out or sys.stderr
        scopes = list(SCOPES)
        r = None
        accounts = self.app.get_accounts()
        if accounts:
            r = self.app.acquire_token_silent(scopes, account=accounts[0])
        if not (r and "access_token" in r):
            if not interactive_ok:
                raise RuntimeError("OneDrive: not signed in; run `plm-kit onedrive login`")
            if self.s.get("auth") == "interactive":
                r = self.app.acquire_token_interactive(scopes=scopes)
            else:
                flow = self.app.initiate_device_flow(scopes=scopes)
                if "user_code" not in flow:
                    raise RuntimeError(f"OneDrive: device code flow failed: {flow.get('error', 'unknown error')}")
                print(flow["message"], file=out)
                r = self.app.acquire_token_by_device_flow(flow)
        self._save()
        if not r or "access_token" not in r:
            raise RuntimeError(f"OneDrive: sign-in failed: {(r or {}).get('error', 'no token')}")
        granted = (r.get("scope") or "").split()
        if any(WRITE_SCOPE.search(g) for g in granted):
            raise RuntimeError("OneDrive: the token carries a write scope; refusing. Grant only Files.Read "
                               "(docs/ONEDRIVE.md)")
        return r["access_token"]

    def logout(self):
        for a in self.app.get_accounts():
            self.app.remove_account(a)
        if self.cache_path.exists():
            self.cache_path.unlink()


def http_get(url, headers, timeout):
    """The only HTTP call this module makes. requests drops the Authorization header on cross-host redirects
    (Graph /content answers with a 302 to a pre-authenticated download URL)."""
    import requests

    return requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)


class GraphReader:
    """Read-only Graph client: lists folder children and downloads file content. GET only."""

    def __init__(self, token_fn, get=None, timeout=30, retries=3, sleep=time.sleep):
        self.token_fn, self.get, self.timeout, self.retries, self.sleep = token_fn, get or http_get, timeout, retries, sleep

    def _get(self, url):
        if not url.startswith(f"{GRAPH}/me/drive/"):
            raise ValueError("OneDrive: refusing a request outside Graph /me/drive")
        for attempt in range(self.retries + 1):
            r = self.get(url, {"Authorization": f"Bearer {self.token_fn()}"}, self.timeout)
            if r.status_code in (429, 503, 504) and attempt < self.retries:
                self.sleep(min(float(r.headers.get("Retry-After", 2 ** attempt)), 60))
                continue
            if r.status_code == 404:
                raise FileNotFoundError(url.split("/me/drive/", 1)[1].split("?")[0])
            r.raise_for_status()
            return r
        raise RuntimeError("OneDrive: throttled; try again later")

    def children(self, folder=None, item_id=None):
        if item_id:
            url = f"{GRAPH}/me/drive/items/{quote(item_id, safe='')}/children?$select={SELECT}"
        else:
            url = f"{GRAPH}/me/drive/root:/{quote(folder)}:/children?$select={SELECT}"
        while url:
            data = self._get(url).json()
            yield from data.get("value", [])
            url = data.get("@odata.nextLink")

    def walk(self, folder):
        """Yield (relative_path, item) for every file under the folder (recursive)."""
        stack = [("", None)]
        while stack:
            prefix, item_id = stack.pop()
            for it in (self.children(item_id=item_id) if item_id else self.children(folder=folder)):
                name = it.get("name", "")
                if not name or name in (".", "..") or "/" in name or "\\" in name:
                    continue
                if "folder" in it:
                    stack.append((f"{prefix}{name}/", it["id"]))
                elif "file" in it:
                    yield f"{prefix}{name}", it

    def content(self, item_id):
        return self._get(f"{GRAPH}/me/drive/items/{quote(item_id, safe='')}/content").content


def sync(reader, s, dest, out=None):
    """Mirror the selected folders' matching files into `dest` (local only). Unchanged files (same eTag) are
    skipped; files removed remotely are removed locally (only files this sync created). Returns counts."""
    out = out or sys.stderr
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    state_p = dest / STATE
    try:
        state = json.loads(state_p.read_text())
    except (OSError, ValueError):
        state = {}
    new, stats = {}, {"downloaded": 0, "unchanged": 0, "skipped": 0, "removed": 0}
    for folder in s["folders"]:
        for rel, it in reader.walk(folder):
            key = f"{folder}/{rel}"
            if not key.lower().endswith(s["extensions"]) or int(it.get("size") or 0) > s["max_bytes"]:
                stats["skipped"] += 1
                continue
            local = (dest / key).resolve()
            if dest.resolve() not in local.parents:
                stats["skipped"] += 1
                continue
            prev = state.get(key)
            if prev and prev.get("eTag") == it.get("eTag") and local.is_file():
                stats["unchanged"] += 1
            else:
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(reader.content(it["id"]))
                stats["downloaded"] += 1
            new[key] = {"id": it["id"], "eTag": it.get("eTag")}
    for key in set(state) - set(new):
        p = (dest / key).resolve()
        if dest.resolve() in p.parents and p.is_file():
            p.unlink()
            stats["removed"] += 1
    _write_private(state_p, json.dumps(new, indent=1))
    return stats
