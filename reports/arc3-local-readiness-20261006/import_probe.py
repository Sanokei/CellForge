"""Import-only diagnosis with the same socket denial, no games or actions."""
import json
import sys
import traceback

events = []


def audit(event, args):
    if event.startswith("socket.") or event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"):
        events.append({"event": event, "stack": [{"file": item.filename, "line": item.lineno, "function": item.name} for item in traceback.extract_stack(limit=8)]})
        raise PermissionError("Import-only probe forbids network and child processes")


sys.addaudithook(audit)
import arc_agi
import arcengine

assert len(events) == 1 and events[0]["event"] == "socket.__new__"
assert any(frame["function"] == "_has_ipv6" and frame["file"].replace("\\", "/").endswith("urllib3/util/connection.py") for frame in events[0]["stack"])
print(json.dumps({"imports": "ok", "game_instances": 0, "actions": 0, "blocked_events": events}, indent=2))
