"""One command for the whole lab (runs on the laptop, talks to the server over ssh `cellforge-1`). Orchestrator regime 2026-10-02; see LAB.md.

  python scripts/lab.py status     # budget meter, live processes, ARC-3 managed run, ARC-1/2 final score
  python scripts/lab.py sync       # copy the server's status files into reports/ (small files only)
  python scripts/lab.py stop       # put STOP files on every driver this regime started (cooperative, resumable)
  python scripts/lab.py budget N   # set the cap on the shared weekly limit (percent, default 85)
"""
import pathlib
import subprocess
import sys

HOST = "cellforge-1"
HEAD = "/srv/cellforge-inbox/head-20261001"
ARC3 = HEAD + "/runs/weights-arc3-managed1"
REPO = pathlib.Path(__file__).resolve().parent.parent


def ssh(command, timeout=120):
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, command], capture_output=True, text=True, timeout=timeout)
    return (done.stdout + done.stderr).strip()


def status():
    print(ssh("date -u +'server time %FT%TZ'; uptime | sed 's/.*load/load/'"))
    print("\n== maker budget (ONE weekly pool shared with the Codex desktop app)")
    print(ssh("cd %s && python3 maker.py status" % HEAD))
    print("\n== live lab processes")
    print(ssh("ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E 'lab_manager|make_arc_score|struct_solve|head_driver|arena|tournament|continue_hard|transfer_audit' | grep -v grep | cut -c1-170") or "none")
    print("\n== ARC-3 managed run")
    print(ssh("cat %s/STATUS.md 2>/dev/null || echo 'no STATUS.md yet'" % ARC3))
    print("\n== ARC-1/2 final score (report split, recomputed)")
    probe = "import json;s=json.load(open('ARC_SCORE.json'));print(s['claimed']['solved'],'of',s['claimed']['total'],'controls',s['controls'])"
    print(ssh("cd %s/runs/weights-head-structbank1 && (test -f ARC_SCORE.json && python3 -c \"%s\" || echo 'not scored yet (make_arc_score.py running or not started)')" % (HEAD, probe)))


def sync():
    out = REPO / "reports" / "arc3"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("STATUS.md", "STATE.json", "VERIFY.json", "ATTEMPTS.jsonl", "ORDERS.jsonl"):
        subprocess.run(["scp", "-q", "%s:%s/%s" % (HOST, ARC3, name), str(out / name)], capture_output=True)
    subprocess.run(["scp", "-q", "%s:%s/BUDGET.jsonl" % (HOST, HEAD), str(REPO / "reports" / "BUDGET.jsonl")], capture_output=True)
    print("synced to", out)


def stop():
    print(ssh("touch %s/STOP && echo 'STOP placed in the ARC-3 run (finishes the current cycle, resumable by deleting it)'" % ARC3))


def budget(value):
    print(ssh("cd %s && echo '{\"stop_at_percent\": %s}' > BUDGET_CAP.json && cat BUDGET_CAP.json" % (HEAD, float(value))))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"status": status, "sync": sync, "stop": stop}.get(cmd, lambda: budget(sys.argv[2]) if cmd == "budget" else print(__doc__))()
