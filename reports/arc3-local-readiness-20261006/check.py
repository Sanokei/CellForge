"""One bounded trusted-engine check driven by a frozen JSON action queue.

No candidate policy is loaded. The Python audit hook is defense in depth for
trusted code, not a security sandbox for arbitrary Python/native code.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ASSETS = ROOT / "data/arc3-official-sdk-fixture-20261006"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(output):
    denied_events = []

    def audit(event, args):
        if event.startswith("socket.") or event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"):
            denied_events.append(event)  # No addresses, arguments or credentials.
            raise PermissionError("Readiness child forbids network and child processes")

    sys.addaudithook(audit)
    result = {"status": "error", "denied_events": denied_events}
    trace_head = "0" * 64
    event_count = 0
    steps = 0
    started = time.monotonic()
    try:
        protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
        manifest = json.loads((ASSETS / "MANIFEST.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            assert digest(ASSETS / entry["path"]) == entry["sha256"], entry["path"]
        assert protocol["max_action_steps"] == len(protocol["actions"]) == 16
        assert os.environ["PYTHON_DOTENV_DISABLED"] == "1"
        assert os.environ["OPERATION_MODE"] == "offline"
        assert "ARC_API_KEY" not in os.environ

        import arcengine
        from arc_agi import Arcade, OperationMode

        result["versions"] = {name: importlib.metadata.version(name) for name in ("arc-agi", "arcengine", "numpy", "pydantic")}
        result["python"] = sys.version
        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(ASSETS / "environment_files"),
            recordings_dir=str(output / "sdk-recordings"),
        )
        assert arcade.operation_mode is OperationMode.OFFLINE
        assert not arcade.arc_api_key
        env = arcade.make(protocol["game_id"], seed=protocol["seed"], save_recording=False, render_mode=None)
        if env is None:
            raise RuntimeError("Official fixture unavailable in offline environment scan")

        with (output / "trace.jsonl").open("x", encoding="utf-8", newline="\n") as trace:
            def record(kind, action_id, observation):
                nonlocal trace_head, event_count
                if observation is None or observation.is_empty():
                    raise RuntimeError("Engine returned missing or empty observation")
                data = observation.model_dump(mode="json")
                data["state"] = observation.state.name
                # Frames are a private Pydantic attribute; model_dump omits them.
                data["frame"] = [layer.tolist() for layer in observation.frame]
                event = {"sequence": event_count, "kind": kind, "action": {"id": action_id, "data": {}}, "observation": data, "previous_sha256": trace_head}
                encoded = canonical(event)
                if len(encoded.encode("utf-8")) > 2_000_000:
                    raise RuntimeError("Observation exceeds readiness receipt bound")
                trace_head = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                trace.write(canonical({**event, "sha256": trace_head}) + "\n")
                trace.flush()
                event_count += 1

            observation = env.observation_space  # make() performs one reset.
            record("automatic_initial_reset", 0, observation)
            for action_id in protocol["actions"]:
                if observation.state.name in protocol["stop_on"]:
                    break
                if time.monotonic() - started >= protocol["wall_seconds"]:
                    raise TimeoutError("Readiness deadline")
                if action_id not in observation.available_actions:
                    raise RuntimeError("Frozen action unavailable; no substitute allowed")
                action = arcengine.GameAction.from_id(action_id)
                observation = env.step(action, data={})
                steps += 1
                record("action", action_id, observation)
        # Closing an OFFLINE scorecard is a local operation only.
        arcade.close_scorecard()
        for entry in manifest["files"]:
            assert digest(ASSETS / entry["path"]) == entry["sha256"]
        assert not denied_events, "Unexpected denied operation"
        result.update(status="ok", game_id=protocol["game_id"], final_state=observation.state.name, levels_completed=observation.levels_completed, stop_reason="terminal" if observation.state.name in protocol["stop_on"] else "action_cap", initial_resets=1, action_steps=steps, total_engine_actions=1 + steps)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        result.update(trace_events=event_count, trace_head_sha256=trace_head, elapsed_seconds=time.monotonic() - started)
        write_json(output / "result.json", result)
    return 0 if result["status"] == "ok" else 1


def launch():
    # Exclusive directory creation prevents silently overwriting/rerunning evidence.
    output = HERE / "attempt01"
    output.mkdir()
    (output / "mpl-config").mkdir()
    child_env = {name: os.environ[name] for name in ("SystemRoot", "WINDIR", "TEMP", "TMP") if name in os.environ}
    child_env.update(PYTHON_DOTENV_DISABLED="1", OPERATION_MODE="offline", MPLBACKEND="Agg", MPLCONFIGDIR=str(output / "mpl-config"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [str(ROOT / ".venv-arc3/Scripts/python.exe"), "-I", "-B", str(Path(__file__).resolve()), "--child", str(output)]
    receipt = {"command": command, "protocol_sha256": digest(HERE / "protocol.json"), "runner_sha256": digest(Path(__file__)), "asset_manifest_sha256": digest(ASSETS / "MANIFEST.json"), "external_wall_limit_seconds": 30, "execution": "default restricted sandbox", "credential_environment_inherited": False}
    started = time.monotonic()
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        try:
            completed = subprocess.run(command, cwd=output, env=child_env, stdout=stdout, stderr=stderr, timeout=30, check=False)
            receipt["returncode"] = completed.returncode
            receipt["timed_out"] = False
        except subprocess.TimeoutExpired:
            receipt.update(returncode=None, timed_out=True)
    receipt["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "process.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["returncode"] == 0 else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        raise SystemExit(worker(Path(sys.argv[2])))
    raise SystemExit(launch())
