# OneDrive source (optional, read-only)

plm-kit can copy selected OneDrive folders' `.md` / `.txt` files to your machine and add them to your local index.
It uses Microsoft Graph with **delegated, read-only** access (`Files.Read`). It never writes, moves, shares or
deletes anything in OneDrive.

**Bring your own app registration.** The kit contains no client id, tenant id or secret. You create a free app
registration in your own Microsoft Entra tenant (or your personal Microsoft account's directory) and put its ids in
your config or environment. Nobody else's app is involved.

## Your responsibility

Only connect accounts and folders you're allowed to copy to your own computer. For a work or school OneDrive, check your employer's policies first. Synced files may contain other people's information; you're responsible for handling it lawfully.

## Third-party services

Microsoft's and Hugging Face's own terms and privacy policies govern those services. Your OneDrive app registration falls under the [Microsoft APIs Terms of Use](https://learn.microsoft.com/en-us/legal/microsoft-apis/terms-of-use).

## Plain-text storage

The index and any synced OneDrive copies are stored as plain text in the cache. Keep your device encrypted and locked (FileVault, BitLocker or LUKS), and delete the cache when you no longer need it (see PRIVACY.md, "Deleting your data").

## 1. Register an app (Microsoft Entra admin center)
1. Go to <https://entra.microsoft.com> > **Identity** > **Applications** > **App registrations** > **New registration**.
2. **Name:** anything, e.g. `my-notes-reader`.
3. **Supported account types:** pick the one that matches the OneDrive you want to read:
   - personal OneDrive: *Personal Microsoft accounts only* (tenant `consumers`), or *any organizational directory and personal Microsoft accounts* (tenant `common`)
   - work/school OneDrive: *Accounts in this organizational directory only* (tenant = your tenant id or domain), or *any organizational directory* (tenant `organizations`)
4. **Redirect URI:** platform *Public client/native (mobile & desktop)*, value `http://localhost`. Only needed for
   `auth = "interactive"`; device-code sign-in doesn't use it.
5. Register. On the **Overview** page, copy the **Application (client) ID** (and the **Directory (tenant) ID** if you
   chose a single-tenant app).
6. **Authentication** > **Advanced settings** > **Allow public client flows** = **Yes** > Save. Device-code sign-in
   needs this. Do **not** create a client secret or certificate; plm-kit is a public client and uses none.
7. **API permissions** > **Add a permission** > **Microsoft Graph** > **Delegated permissions** > **Files.Read** >
   Add. Remove any other permission you don't need (`User.Read` is added by default; you can remove it).
   `openid`, `profile` and `offline_access` are requested automatically by MSAL during sign-in (offline_access lets
   it refresh the token without asking you again). **Do not add any write permission** (anything named
   `...ReadWrite...`). plm-kit refuses a token that carries one.
8. Work/school tenants: your admin may need to allow user consent or grant consent for `Files.Read`.

Microsoft's reference: [Register an application](https://learn.microsoft.com/entra/identity-platform/quickstart-register-app)
and [desktop app registration](https://learn.microsoft.com/entra/identity-platform/scenario-desktop-app-registration).

## 2. Configure plm-kit
In your config (see `config.example.toml`), or via environment variables (env wins):

```toml
[sources.onedrive]
enabled = true
client_id = ""            # or env PLM_KIT_ONEDRIVE_CLIENT_ID = your Application (client) ID
tenant = "consumers"      # or env PLM_KIT_ONEDRIVE_TENANT: consumers | organizations | common | <tenant id or domain>
auth = "device_code"      # or "interactive"
folders = ["Notes"]       # paths from your OneDrive root
```

Keep your config outside the repo. Client and tenant ids aren't passwords, but they identify your registration and
must never be committed (CI and `tests/test_onedrive.py` fail on any GUID-like id in the repo).

## 3. Sign in and sync
```bash
plm-kit onedrive status     # shows what will be used (scope, folders, token cache path); no network
plm-kit onedrive login      # device code: open the URL it prints, enter the code, approve "Files.Read"
plm-kit onedrive sync       # GET-only download of the selected folders, then rebuilds the local index
plm-kit ask "..."           # synced notes show up as onedrive/<folder>/<file>
plm-kit onedrive logout     # removes the account from the token cache and deletes the cache file
```
`--folder Notes/Recipes` (repeatable) overrides `folders` for one sync.

## What it does, exactly
- Requests: `GET https://graph.microsoft.com/v1.0/me/drive/root:/<folder>:/children`, `.../items/<id>/children`
  and `.../items/<id>/content`. Nothing else, and no other host gets your token. Requests outside
  `graph.microsoft.com/v1.0/me/drive/` are refused, including paging links.
- Copies only files whose extension is listed (default `.md`, `.txt`) and whose size is under `max_file_mb`.
- Local copies go to `<cache>/sources/onedrive/` (outside the repo). Unchanged files (same eTag) aren't
  downloaded again. A file deleted in OneDrive is deleted from the local copy on the next sync (only files the
  sync itself created).
- Tokens: MSAL token cache in your OS user cache dir (Linux `~/.cache/plm-kit/`, macOS `~/Library/Caches/plm-kit/`,
  Windows `%LOCALAPPDATA%\plm-kit\`), file mode 0600, never in the repo.
- To revoke access completely: `plm-kit onedrive logout`, then remove the app's consent at
  <https://account.live.com/consent/Manage> (personal) or **My Apps** (work/school), or delete the app registration.
