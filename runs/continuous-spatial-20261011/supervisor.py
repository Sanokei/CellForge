"""Durable sequential CellForge queue, fixed systemd cgroup and global flock.
No arbitrary queue commands. All outcomes are persisted before another job.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import pathlib
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone

UNIT = "cellforge-continuous-job.service"
LOCK = "/run/cellforge-continuous.lock"
ACTIVE = {"active", "activating", "deactivating"}
PYTHON = "/srv/arc3-claude/venv/bin/python"
LIMITS = {"cpu_quota": "100%", "cpu_affinity": "0", "memory_max": 1073741824,
          "runtime_max_seconds": 600, "disk_floor": 5 * 1024 ** 3,
          "run_bytes_max": 64 * 1024 ** 2, "total_bytes_max": 256 * 1024 ** 2}
CODE = ["supervisor.py", "worker.py", "policy_v2.py", "arc3_env.py", "arc3cell.py", "structcode.py"]

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()

def filehash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(canonical(obj) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)

def read(path):
    return json.loads(path.read_text())

def tree_size(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())

def fingerprint(config):
    return digest({k: v for k, v in config.items() if k != "fingerprint"})

def validate(config, plan):
    if config["fingerprint"] != fingerprint(config):
        raise ValueError("experiment fingerprint mismatch")
    if config["games"] != ["ar25", "lp85", "sc25"] or config["actions"] != 1500:
        raise ValueError("unapproved development scope")
    if config["remote_maker_calls"] != 0:
        raise ValueError("remote model calls are disabled; fresh quota adapter required")
    if type(config["seed"]) is not int or config["seed"] not in (0, 2, 3):
        raise ValueError("unapproved environment seed")
    if type(config["control_seed"]) is not int:
        raise ValueError("invalid control seed")
    if set(config["params"]) != {"action", "pixel", "rates"}:
        raise ValueError("unknown tensor key")
    for key, size in (("action", 8), ("pixel", 6), ("rates", 2)):
        values = config["params"][key]
        if len(values) != size or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 20 for v in values):
            raise ValueError("invalid finite numeric tensor")
    if config["code_sha256"] != plan["code_sha256"] or config["dataset_sha256"] != plan["dataset_sha256"]:
        raise ValueError("experiment source or dataset binding differs")
    if config["hypothesis"] not in plan["hypotheses"]:
        raise ValueError("unapproved hypothesis")

def command(argv):
    return subprocess.run(argv, text=True, capture_output=True, timeout=15)

class Systemd:
    def show(self):
        result = command(["systemctl", "show", UNIT, "--no-pager",
                          "--property=LoadState,ActiveState,SubState,Result,ExecMainStatus,MainPID,ControlGroup,CPUUsageNSec,MemoryPeak,RuntimeMaxUSec,MemoryMax,CPUQuotaPerSecUSec,InvocationID"])
        return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

    def empty(self, info):
        group = info.get("ControlGroup", "")
        if not group:
            return True
        if not group.startswith("/") or ".." in pathlib.PurePosixPath(group).parts:
            return False
        path = pathlib.Path("/sys/fs/cgroup") / group.lstrip("/")
        if not path.exists():
            return True
        events = dict(line.split() for line in (path / "cgroup.events").read_text().splitlines())
        return events.get("populated") == "0"

    def stop(self):
        result = command(["systemctl", "stop", UNIT])
        info = self.show()
        if result.returncode and info.get("LoadState") != "not-found":
            raise RuntimeError("job stop failed: " + result.stderr[:200])
        if info.get("ActiveState") in ACTIVE or not self.empty(info):
            raise RuntimeError("job tree still populated; refusing next job")
        return info

    def start(self, root, run):
        info = self.show()
        if info.get("ActiveState") in ACTIVE or not self.empty(info):
            raise RuntimeError("existing job or orphan cgroup prevents launch")
        command(["systemctl", "reset-failed", UNIT])
        argv = ["systemd-run", "--quiet", "--unit=" + UNIT, "--service-type=exec",
                "--property=RemainAfterExit=yes", "--property=Restart=no",
                "--property=CPUQuota=100%", "--property=CPUAffinity=0",
                "--property=MemoryMax=1G", "--property=MemorySwapMax=0",
                "--property=MemoryAccounting=yes", "--property=CPUAccounting=yes",
                "--property=TasksMax=64", "--property=RuntimeMaxSec=600",
                "--property=TimeoutStopSec=5", "--property=KillMode=control-group",
                "--property=SendSIGKILL=yes", "--property=OOMPolicy=kill",
                "--property=StandardOutput=null", "--property=StandardError=null",
                "--property=PrivateNetwork=yes", "--property=PrivateTmp=yes",
                "--property=ProtectSystem=strict", "--property=ProtectHome=yes",
                "--property=ReadWritePaths=" + str(run),
                "--property=InaccessiblePaths=/srv/cellforge-private",
                "--property=LimitFSIZE=16M", "--property=LimitCORE=0",
                "--property=WorkingDirectory=" + str(root)]
        for key, value in {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                           "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                           "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(run / "scratch"),
                           "ARC3_DIR": "/srv/arc3-claude"}.items():
            argv.append("--setenv=" + key + "=" + value)
        argv += [PYTHON, str(root / "worker.py"), str(run)]
        result = command(argv)
        if result.returncode:
            raise RuntimeError("systemd launch failed: " + result.stderr[:500])
        return self.show()

class Queue:
    def __init__(self, root, backend=None):
        self.root = pathlib.Path(root)
        self.backend = backend or Systemd()
        self.plan = read(self.root / "PLAN.json")
        self.state = read(self.root / "STATE.json") if (self.root / "STATE.json").exists() else {
            "schema": "cellforge-queue/1", "outcomes": {}, "active": None, "consecutive_failures": 0,
            "not_before": 0, "status": "ready"}
        self.halt = False
        self.guard_cache = None

    def persist(self, status, reason=None):
        self.state.update(status=status, reason=reason, updated_utc=now())
        atomic(self.root / "STATE.json", self.state)

    def event(self, kind, **values):
        path = self.root / "EVENTS.jsonl"
        if path.exists() and path.stat().st_size > 1024 * 1024:
            os.replace(path, self.root / "EVENTS.previous.jsonl")
        with path.open("ab") as stream:
            stream.write(canonical(dict(utc=now(), kind=kind, **values)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def cancelled(self, fp):
        return (self.root / "cancel" / fp).exists()

    def gates(self):
        for filename, expected in self.plan["code_sha256"].items():
            if filehash(self.root / filename) != expected:
                return "code_changed:" + filename
        # Protected STOPs are checked on every tick; no lane is revived by this queue.
        for path, expected in self.plan["protected_stops"].items():
            p = pathlib.Path(path)
            if not p.exists() or filehash(p) != expected:
                return "protected_stop_changed:" + path
        for path, expected in self.plan["dataset_files"].items():
            if filehash(pathlib.Path(path)) != expected:
                return "development_dataset_changed"
        if shutil.disk_usage(self.root).free < LIMITS["disk_floor"]:
            return "disk_floor"
        if tree_size(self.root) > LIMITS["total_bytes_max"]:
            return "artifact_cap"
        return None

    def terminal(self, fp, info, override=None):
        run = self.root / "runs" / fp
        # Metrics are read before stopping/resetting the transient unit.
        summary = read(run / "SUMMARY.json") if (run / "SUMMARY.json").exists() else None
        result = override or ("completed" if info.get("Result") == "success" and
                  info.get("ExecMainStatus") == "0" and summary else "failed")
        stopped = self.backend.stop()
        if not self.backend.empty(stopped):
            raise RuntimeError("cannot issue terminal receipt while job tree is alive")
        receipt = {"fingerprint": fp, "run_id": "spatial-" + fp[:16],
                   "terminal_utc": now(), "outcome": result, "systemd": info,
                   "limits": LIMITS, "job_tree_empty": True,
                   "config_sha256": filehash(run / "CONFIG.json"),
                   "results_sha256": filehash(run / "RESULTS.json") if (run / "RESULTS.json").exists() else None,
                   "summary": summary}
        target = run / "TERMINAL.json"
        if target.exists():
            if read(target)["fingerprint"] != fp:
                raise RuntimeError("conflicting terminal receipt")
        else:
            atomic(target, receipt)
        self.state["outcomes"][fp] = read(target)["outcome"]
        self.state["active"] = None
        if result == "failed":
            self.state["consecutive_failures"] += 1
            self.state["not_before"] = time.time() + min(300, 30 * 2 ** (self.state["consecutive_failures"] - 1))
        elif result == "completed":
            self.state["consecutive_failures"] = 0
        self.persist("ready")
        self.event("terminal", fingerprint=fp, outcome=result)
        # Scratch is explicitly disposable evaluator recording workspace, never evidence.
        scratch = run / "scratch"
        if scratch.exists() and scratch.resolve().parent == run.resolve():
            shutil.rmtree(scratch)

    def tick(self):
        info = self.backend.show()
        fp = self.state.get("active")
        stopped = (self.root / "STOP").exists() or self.halt
        # Crash recovery: terminal receipt dominates incomplete STATE commits.
        if fp and (self.root / "runs" / fp / "TERMINAL.json").exists():
            self.backend.stop()
            self.state["outcomes"][fp] = read(self.root / "runs" / fp / "TERMINAL.json")["outcome"]
            self.state["active"] = None
            self.persist("ready")
            return
        if stopped:
            if fp:
                self.terminal(fp, info, "cancelled")
            elif info.get("ActiveState") in ACTIVE:
                self.backend.stop()
            self.persist("paused_stop", "STOP or service shutdown")
            return
        if fp and self.cancelled(fp):
            self.terminal(fp, info, "cancelled")
            return
        gate = self.gates()
        if gate:
            if fp:
                self.terminal(fp, info, "resource_blocked")
            self.persist("blocked_gate", gate)
            return
        if fp:
            run = self.root / "runs" / fp
            if tree_size(run) > LIMITS["run_bytes_max"]:
                self.terminal(fp, info, "resource_blocked")
                self.persist("blocked_gate", "run_artifact_cap")
            elif info.get("SubState") == "running" and info.get("ActiveState") in ACTIVE:
                self.persist("running", fp)
            elif info.get("SubState") == "exited" or info.get("ActiveState") == "failed":
                self.terminal(fp, info)
            elif info.get("LoadState") == "not-found" or info.get("ActiveState") == "inactive":
                self.terminal(fp, info, "interrupted")
            else:
                self.persist("recovering", str(info))
            return
        if info.get("ActiveState") in ACTIVE or not self.backend.empty(info):
            self.persist("blocked_orphan", "unowned fixed job unit; operator intervention required")
            return
        if self.state["consecutive_failures"] >= 3:
            self.persist("blocked_failure_circuit", "three consecutive failed experiments; no automatic retry")
            return
        if time.time() < self.state["not_before"]:
            self.persist("failure_backoff", "bounded failure backoff")
            return
        configs = self.plan["experiments"]
        fingerprints = [c["fingerprint"] for c in configs]
        if len(fingerprints) != len(set(fingerprints)):
            self.persist("blocked_invalid_queue", "duplicate scientific fingerprint")
            return
        for config in configs:
            fp = config["fingerprint"]
            if fp in self.state["outcomes"]:
                continue
            try:
                validate(config, self.plan)
            except Exception as exc:
                self.persist("blocked_invalid_queue", str(exc))
                return
            if self.cancelled(fp):
                self.state["outcomes"][fp] = "cancelled_before_start"
                self.persist("ready")
                continue
            run = self.root / "runs" / fp
            if run.exists():
                # Never overwrite/repeat work after a partially committed launch.
                if (run / "TERMINAL.json").exists():
                    self.state["outcomes"][fp] = read(run / "TERMINAL.json")["outcome"]
                    self.persist("ready")
                    continue
                self.state["active"] = fp
                self.persist("recovering")
                self.terminal(fp, info, "interrupted")
                return
            run.mkdir()
            (run / "scratch").mkdir()
            atomic(run / "CONFIG.json", config)
            atomic(run / "START.json", {"utc": now(), "fingerprint": fp,
                   "run_id": "spatial-" + fp[:16], "limits": LIMITS,
                   "scientific_context": self.plan["scientific_context"]})
            self.state["active"] = fp
            self.persist("launching", fp)  # durable intent before service creation
            self.event("launch_intent", fingerprint=fp)
            try:
                unit = self.backend.start(self.root, run)
                atomic(run / "UNIT.json", unit)
                self.persist("running", fp)
                self.event("launched", fingerprint=fp, invocation=unit.get("InvocationID"))
            except Exception as exc:
                # Inspect the fixed unit before deciding that a failed RPC did not start it.
                fresh = self.backend.show()
                if fresh.get("SubState") == "running":
                    self.persist("recovering", str(exc))
                else:
                    self.terminal(fp, fresh, "failed")
            return
        self.persist("paused_exhausted", "nine preregistered experiments consumed; next hypothesis/audit required")

def acquire_lock(path=LOCK):
    stream = open(path, "a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    if args.stop:
        # Automatic crash restart must adopt the existing job, not synthesize cancellation.
        # An actual STOP marker always wins, including while the supervisor failed.
        if os.environ.get("SERVICE_RESULT", "success") != "success" and not (args.root / "STOP").exists():
            return 0
        atomic(args.root / "STOP", {"utc": now(), "reason": "explicit service stop"})
        Systemd().stop()
        return 0
    lock = acquire_lock()
    if lock is None:
        print("duplicate supervisor refused: global lock held", flush=True)
        return 0
    queue = Queue(args.root)
    def halt(*_):
        queue.halt = True
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    try:
        while True:
            queue.tick()
            if queue.halt:
                return 0
            time.sleep(2)
    finally:
        lock.close()

if __name__ == "__main__":
    raise SystemExit(main())

