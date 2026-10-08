"""USER-RUN ONLY: download one public ls20 asset; never instantiate/play it.

API paths/header match installed arc-agi 0.9.9 base.py. This deliberately does
not import arc_agi: Arcade() can fetch a key/list games, and make() starts a
scorecard and resets the game. Nothing here runs on import.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
DESTINATION = ROOT / "data" / "arc3-public-ls20"
ORIGIN = "https://three.arcprize.org"


class DownloadError(Exception):
    """Only controlled, credential-free messages may be used here."""


def main() -> None:
    if DESTINATION.exists():
        raise DownloadError("Destination already exists; no requests made. Preserve it for offline inspection.")

    import requests
    from dotenv import dotenv_values

    # Explicit user-named file only; no parent search, env fallback or expansion.
    logging.getLogger("dotenv.main").disabled = True
    key = dotenv_values(ROOT / ".env", interpolate=False).get("ARC_API_KEY")
    if not key or not key.strip() or any(char in key for char in "\r\n"):
        raise DownloadError("ARC_API_KEY is missing/invalid in the repository .env; no requests made.")

    def fetch(session, url: str, limit: int) -> bytes:
        try:
            with session.get(url, headers={"X-Api-Key": key, "Accept": "application/json"},
                             timeout=(10, 30), allow_redirects=False, stream=True,
                             verify=True) as response:
                if response.status_code != 200:
                    raise DownloadError(f"Official API returned HTTP {response.status_code}; stopped without following redirects or printing its body. Check access/terms on the official platform if needed.")
                chunks = []
                size = 0
                for chunk in response.iter_content(chunk_size=65536):
                    size += len(chunk)
                    if size > limit:
                        raise DownloadError("Official response exceeds the bounded download size; stopped.")
                    chunks.append(chunk)
        except requests.RequestException:
            raise DownloadError("Network/TLS request failed; response details and credentials are not logged.") from None
        body = b"".join(chunks)
        if key.encode("utf-8") in body:
            raise DownloadError("Response echoed the credential; nothing from that response was saved.")
        return body

    with requests.Session() as session:
        # Do not inherit .netrc authentication, proxies or custom CA env settings.
        # TLS certificate verification remains enabled; no retries or redirects.
        session.trust_env = False
        metadata_url = ORIGIN + "/api/games/ls20"
        raw_metadata = fetch(session, metadata_url, 256 * 1024)
        try:
            metadata = json.loads(raw_metadata)
        except (ValueError, UnicodeError):
            raise DownloadError("Metadata was not valid JSON; no assets saved.") from None
        if not isinstance(metadata, dict):
            raise DownloadError("Metadata was not a JSON object; no assets saved.")
        game_id = metadata.get("game_id")
        if not isinstance(game_id, str) or not re.fullmatch(r"ls20-[A-Za-z0-9][A-Za-z0-9._-]{0,63}", game_id):
            raise DownloadError("Metadata did not identify a versioned ls20 game; no assets saved.")
        version = game_id[len("ls20-"):]
        if metadata.get("version") not in (None, version):
            raise DownloadError("Metadata version fields disagree; no assets saved.")
        class_name = metadata.get("class_name") or "Ls20"
        if not isinstance(class_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", class_name):
            raise DownloadError("Metadata class name is invalid; no assets saved.")
        source_url = ORIGIN + "/api/games/" + game_id + "/source"
        source = fetch(session, source_url, 8 * 1024 * 1024)
    # Parse only, never compile/execute/import the downloaded game.
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, UnicodeError):
        raise DownloadError("Source was not parseable Python; no assets saved.") from None
    if not any(isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body):
        raise DownloadError("Source lacks its declared game class; no assets saved.")

    # Create an exclusive root after both downloads validate. Never overwrite.
    DESTINATION.mkdir()
    game_dir = DESTINATION / "environment_files" / "ls20" / version
    game_dir.mkdir(parents=True)
    files = []
    for name, body, url in (("metadata.json", raw_metadata, metadata_url),
                            (class_name.lower() + ".py", source, source_url)):
        path = game_dir / name
        with path.open("xb") as handle:
            handle.write(body)
        files.append({"path": path.relative_to(DESTINATION).as_posix(), "url": url,
                      "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    receipt = {
        "game_id": game_id, "version": version, "class_name": class_name,
        "classification": "Official public demonstration game ls20; not private evaluation",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "download_actor": "user-run handoff", "successful_get_requests": 2,
        "game_instances_created": 0, "game_actions": 0, "scorecards_created": 0,
        "license_status": "Unverified; preserve any terms/license in metadata/source. Do not infer the engine MIT license applies to game assets.",
        "files": files,
    }
    with (DESTINATION / "DOWNLOAD.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(f"Downloaded {game_id} to {DESTINATION}")
    print("Metadata/source hashes saved. No game loaded, reset, scored or played. Ready for offline asset/license inspection.")


if __name__ == "__main__":
    try:
        main()
    except DownloadError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    except Exception:
        # Do not print arbitrary library exceptions, locals, headers or bodies.
        print("Download preparation/write failed; preserve any partial destination for inspection. No exception details logged.", file=sys.stderr)
        raise SystemExit(1)
