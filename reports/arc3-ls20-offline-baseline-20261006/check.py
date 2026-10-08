"""Bounded host harness; consumes action DATA and trusted publisher game only."""
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
ASSETS = ROOT / "data/arc3-public-ls20"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(output):
    denied_import = []
    denied_runtime = []
    importing = True

    def audit(event, args):
        if event.startswith("socket.") or event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"):
            (denied_import if importing else denied_runtime).append(event)
            raise PermissionError("Offline baseline forbids sockets and child processes")

    sys.addaudithook(audit)
    result = {"status": "error", "denied_import_events": denied_import, "denied_runtime_events": denied_runtime}
    trace_head = "0" * 64
    event_count = steps = 0
    started = time.monotonic()
    try:
        protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
        manifest = json.loads((ASSETS / "DOWNLOAD.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            assert digest(ASSETS / entry["path"]) == entry["sha256"]
        assert protocol["max_action_steps"] == len(protocol["actions"]) == 149
        assert os.environ["PYTHON_DOTENV_DISABLED"] == "1"
        assert os.environ["OPERATION_MODE"] == "offline"
        assert "ARC_API_KEY" not in os.environ

        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction

        # The same guard stays active. urllib3 catches the denied IPv6 import
        # probe. No socket is allowed; only the reporting phase changes here.
        assert denied_import in ([], ["socket.__new__"]), "Unexpected import operation"
        importing = False
        result["versions"] = {name: importlib.metadata.version(name) for name in ("arc-agi", "arcengine", "numpy", "pydantic")}
        arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                        environments_dir=str(ASSETS / "environment_files"),
                        recordings_dir=str(output / "local-scorecards"))
        assert arcade.operation_mode is OperationMode.OFFLINE and not arcade.arc_api_key
        env = arcade.make(protocol["game_id"], seed=protocol["seed"], save_recording=False, render_mode=None)
        if env is None:
            raise RuntimeError("Official public game unavailable offline")

        with (output / "trace.jsonl").open("x", encoding="utf-8", newline="\n") as trace:
            def record(kind, action_id, observation):
                nonlocal trace_head, event_count
                if observation is None or observation.is_empty():
                    raise RuntimeError("Engine returned a missing/empty observation")
                data = observation.model_dump(mode="json")
                data["state"] = observation.state.name
                data["frame"] = [layer.tolist() for layer in observation.frame]
                event = {"sequence": event_count, "kind": kind,
                         "action": {"id": action_id, "data": {}},
                         "observation": data, "previous_sha256": trace_head}
                encoded = canonical(event)
                if len(encoded.encode("utf-8")) > 8_000_000:
                    raise RuntimeError("Observation exceeds receipt bound")
                trace_head = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                trace.write(canonical({**event, "sha256": trace_head}) + "\n")
                trace.flush()
                event_count += 1
                return data

            observation = env.observation_space  # make() performs one reset.
            initial = record("automatic_initial_reset", 0, observation)
            write_json(output / "INITIAL_OBSERVATION.json", initial)
            schema = {"game_id": protocol["game_id"], "initial_available_actions": initial["available_actions"],
                      "actions": [{"id": action.value, "name": action.name,
                                   "data_schema": action.action_type.model_json_schema()}
                                  for action in env.action_space],
                      "model_contract": {"id": "integer from latest available_actions", "data": "object matching that action schema"},
                      "initialization_reset_id": 0, "reset_not_in_baseline_queue": True}
            write_json(output / "ACTION_SCHEMA.json", schema)
            print(json.dumps({"initial_state": initial["state"], "available_actions": initial["available_actions"],
                              "levels_completed": initial["levels_completed"], "win_levels": initial["win_levels"],
                              "observation_file": str(output / "INITIAL_OBSERVATION.json")}), flush=True)
            for action_id in protocol["actions"]:
                if observation.state.name in protocol["stop_on"]:
                    break
                if time.monotonic() - started >= protocol["wall_seconds"]:
                    raise TimeoutError("Baseline wall deadline")
                if action_id not in observation.available_actions:
                    raise RuntimeError("Frozen action unavailable; no substitution")
                observation = env.step(GameAction.from_id(action_id), data={})
                steps += 1
                record("action", action_id, observation)
        scorecard = arcade.close_scorecard()  # OFFLINE: local bookkeeping only.
        if scorecard is None:
            raise RuntimeError("Missing local scorecard")
        write_json(output / "LOCAL_SCORECARD.json", scorecard.model_dump(mode="json"))
        for entry in manifest["files"]:
            assert digest(ASSETS / entry["path"]) == entry["sha256"]
        assert not denied_runtime, "Unexpected runtime operation"
        result.update(status="ok", game_id=protocol["game_id"], final_state=observation.state.name,
                      levels_completed=observation.levels_completed, win_levels=observation.win_levels,
                      stop_reason="terminal" if observation.state.name in protocol["stop_on"] else "action_cap",
                      initial_resets=1, action_steps=steps, total_engine_actions=1 + steps)
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        result.update(trace_events=event_count, trace_head_sha256=trace_head, elapsed_seconds=time.monotonic() - started)
        write_json(output / "result.json", result)
    return 0 if result["status"] == "ok" else 1


def launch():
    output = HERE / "attempt01"
    output.mkdir()  # Exclusive: preserve evidence, prohibit accidental rerun.
    (output / "mpl-config").mkdir()
    child_env = {name: os.environ[name] for name in ("SystemRoot", "WINDIR", "TEMP", "TMP") if name in os.environ}
    child_env.update(PYTHON_DOTENV_DISABLED="1", OPERATION_MODE="offline", MPLBACKEND="Agg",
                     MPLCONFIGDIR=str(output / "mpl-config"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [str(ROOT / ".venv-arc3/Scripts/python.exe"), "-I", "-B", str(Path(__file__).resolve()), "--child", str(output)]
    receipt = {"command": command, "protocol_sha256": digest(HERE / "protocol.json"),
               "runner_sha256": digest(Path(__file__)), "asset_receipt_sha256": digest(ASSETS / "DOWNLOAD.json"),
               "external_wall_limit_seconds": 30, "execution": "default restricted sandbox", "credential_environment_inherited": False}
    started = time.monotonic()
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        try:
            completed = subprocess.run(command, cwd=output, env=child_env, stdout=stdout, stderr=stderr, timeout=30, check=False)
            receipt.update(returncode=completed.returncode, timed_out=False)
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
