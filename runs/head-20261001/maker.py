"""Budget-metered maker route (server). One place for every maker call from the orchestrator regime on (2026-10-02).

The maker is the owner-configured route: server `codex exec -m gpt-6.1-sol -c model_reasoning_effort=low`, ChatGPT subscription login, read-only sandbox, no
API key. That login shares ONE weekly limit with the Codex desktop app (measured 2026-10-02: both report the same `used_percent` and `resets_at`).
This module makes that limit visible and enforceable:
  - the call is NOT ephemeral, so Codex writes a rollout file whose `token_count` events carry `rate_limits.primary.used_percent`; we read it back;
  - every call appends one line to BUDGET.jsonl (utc, role, track, used_percent, resets_at, seconds, prompt sha);
  - `ask` refuses (BudgetExhausted) when the last recorded used_percent is at or above the cap in BUDGET_CAP.json (default 85);
  - the tool-use guard of head_arc_cells.ask_maker is kept: a response that ran a command or an MCP tool is never released.
Nothing here writes weights; it only calls the maker and meters it.

  python maker.py status            # last meter reading, cap, calls today
  python maker.py probe             # one tiny call (about 15k cached input tokens) to refresh the meter
"""
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

CODEX = os.environ.get("CODEX_BIN", "/root/.local/bin/codex")
HOME = pathlib.Path(os.environ.get("HEAD_HOME", "/srv/cellforge-inbox/head-20261001"))
LEDGER = pathlib.Path(os.environ.get("MAKER_LEDGER", str(HOME / "BUDGET.jsonl")))
CAP_FILE = pathlib.Path(os.environ.get("MAKER_CAP", str(HOME / "BUDGET_CAP.json")))
SESSIONS = pathlib.Path(os.path.expanduser("~/.codex/sessions"))
DEFAULT_CAP = 85.0
_lock = threading.Lock()


class BudgetExhausted(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def cap():
    try:
        return float(json.loads(CAP_FILE.read_text())["stop_at_percent"])
    except Exception:  # noqa: BLE001
        return DEFAULT_CAP


def readings():
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def last_used():
    """Most recent used_percent in the ledger, or None when nothing has been metered yet."""
    for row in reversed(readings()):
        if row.get("used_percent") is not None:
            return row["used_percent"], row.get("resets_at")
    return None, None


def _thread_id(events_text):
    for line in events_text.splitlines():
        if '"thread.started"' in line:
            try:
                return json.loads(line)["thread_id"]
            except (ValueError, KeyError):
                return None
    return None


def meter_from_rollout(thread_id):
    """(used_percent, resets_at) from the last token_count event of this call's rollout file, or (None, None)."""
    if not thread_id or not SESSIONS.exists():
        return None, None
    for _ in range(10):
        files = list(SESSIONS.glob("*/*/*/rollout-*-%s.jsonl" % thread_id))
        if files:
            used = resets = None
            for line in files[0].read_text(errors="replace").splitlines():
                if '"rate_limits"' not in line:
                    continue
                try:
                    primary = json.loads(line)["payload"]["rate_limits"]["primary"]
                    used, resets = primary["used_percent"], primary["resets_at"]
                except (ValueError, KeyError, TypeError):
                    continue
            return used, resets
        time.sleep(0.5)
    return None, None


def log(row):
    with _lock:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as f:
            f.write(json.dumps(row) + "\n")


def ask(text, work, role="writer", track="", timeout=900, model="gpt-6.1-sol", effort="low"):
    """Run one maker call. Returns the parsed JSON of the response. Raises BudgetExhausted before the call when the meter is at the cap,
    RuntimeError when the maker failed or used a tool (nothing is released then), ValueError when the response is not JSON."""
    used, _ = last_used()
    if used is not None and used >= cap():
        raise BudgetExhausted("weekly meter %.1f%% >= cap %.1f%%" % (used, cap()))
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")}
    cmd = [CODEX, "exec", "--ignore-user-config", "--skip-git-repo-check", "--sandbox", "read-only", "-m", model,
           "-c", 'model_reasoning_effort="%s"' % effort, "--json", "-C", str(work), "-o", str(work / "response.json"), "-"]
    import hashlib
    sha = hashlib.sha256(text.encode()).hexdigest()
    (work / "START.json").write_text(json.dumps({"utc": now(), "model": model, "reasoning": effort, "auth": "ChatGPT subscription", "prompt_sha256": sha,
                                                  "role": role, "track": track}))
    started = time.monotonic()
    with open(work / "events.jsonl", "w") as sink:
        done = subprocess.run(cmd, input=text, text=True, stdout=sink, stderr=sink, env=env, timeout=timeout)
    seconds = time.monotonic() - started
    events = (work / "events.jsonl").read_text()
    (work / "EXIT.json").write_text(json.dumps({"returncode": done.returncode, "seconds": seconds}))
    used, resets = meter_from_rollout(_thread_id(events))
    log({"utc": now(), "role": role, "track": track, "work": work.name, "seconds": round(seconds, 1), "returncode": done.returncode,
         "used_percent": used, "resets_at": resets, "prompt_sha256": sha, "prompt_chars": len(text)})
    if done.returncode or '"type":"command_execution"' in events or '"type":"mcp_tool_call"' in events:
        raise RuntimeError("maker failed or invoked tools; no release")
    raw = (work / "response.json").read_text().strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(raw)


def status():
    rows = readings()
    used, resets = last_used()
    today = [r for r in rows if r["utc"][:10] == now()[:10]]
    by_role = {}
    for r in rows:
        by_role[r["role"]] = by_role.get(r["role"], 0) + 1
    print(json.dumps({"used_percent": used, "resets_at": resets, "cap_percent": cap(), "calls_logged": len(rows), "calls_today": len(today),
                      "calls_by_role": by_role}, indent=1))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "probe":
        probe = pathlib.Path("/tmp/maker_probe")
        print(ask("Reply with the JSON array [1].", probe, role="probe", track="meter"))
    status()
