# User-run ls20 asset acquisition

Prepared, statically reviewed, **not executed**:
[arc3_download_ls20_user.py](arc3_download_ls20_user.py).
No `.env` content or credential was read, displayed or transmitted during this
preparation. Existing `.env` and security settings are untouched.

Run this yourself in PowerShell from `D:\Github\CellForge`:

```powershell
& '.\.venv-arc3\Scripts\python.exe' -I -B '.\reports\arc3_download_ls20_user.py'
```

Only when you execute it, the script reads `ARC_API_KEY` from this repository's
`.env` and sends it in `X-Api-Key` to `https://three.arcprize.org`. It does not
put the key in arguments, URLs, logs, output, receipts or saved files. No need
to paste the key into chat or modify `.env`.

The two successful GETs are `/api/games/ls20` and
`/api/games/ls20-<returned-version>/source`, matching installed official
`arc-agi==0.9.9` source. Downloads stop on redirects, non-200 status, oversized
responses, malformed/mismatched metadata or invalid Python syntax. Source is
parsed only, never executed. Response bodies/errors and request headers are not
printed. TLS verification stays enabled, and inherited `.netrc` credentials,
proxies and custom CA environment settings are not used by this script.

Output is exclusively under `data/arc3-public-ls20/`:

- `environment_files/ls20/<version>/metadata.json` — original metadata bytes.
- `environment_files/ls20/<version>/<class-name>.py` — original source bytes.
- `DOWNLOAD.json` — exact ID/version, source URLs, lengths and SHA-256 hashes.

An existing destination stops the script before credential loading or requests.
If writing is interrupted, preserve the partial directory for inspection rather
than deleting or replacing evidence. The script does not instantiate `Arcade`,
fetch an anonymous key, list other games, open a scorecard, reset/play a game,
submit anything or import the downloaded source. `Arcade()`/`make()` are avoided
because their convenience workflow has those discovery/scoring/reset effects.

Access/terms caveat: a valid key must be entitled to the public `ls20` download.
If the official service requires login, consent, license acceptance or another
access step, the script stops; it cannot accept terms for you. No standalone
game license has yet been verified. Preserve any license/terms returned in
metadata or source; the engine's MIT license is not proof of the game's license.

After the command succeeds, offline review can verify hashes, the exact version,
asset license and import safety before the separately bounded action trace.
Downloading alone produces no game score or evidence of agent capability.

Official references: [API-key setup](https://docs.arcprize.org/api-keys),
[public example games](https://docs.arcprize.org/available-games),
[offline execution](https://docs.arcprize.org/local-vs-online).
