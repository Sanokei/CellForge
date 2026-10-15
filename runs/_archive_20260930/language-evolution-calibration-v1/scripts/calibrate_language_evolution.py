"""Gen0 calibration and operator-private pilot split; no gradient operations."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.runtime.lm import score_texts


def content_fitness(graph, items):
    texts = [item["text"] for item in items]
    scores, info = score_texts(graph, texts)
    lengths = [len(text.encode("utf-8")) for text in texts]
    penalty = max(8., float(np.log2(len(graph.meta["vocab"])) + 1))
    bits = sum(penalty*length if scores["failed"][i] else float(info["per_pos_bits"][i,:length].sum()) for i,length in enumerate(lengths))
    return {"fitness_bits_per_byte": bits/sum(lengths), "eligible_for_selection": not bool(scores["failed"].any()),
            "next_byte": {"content_bits": bits, "content_bytes": sum(lengths), "failed_documents": int(scores["failed"].sum()), "EOS": "excluded"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--public", required=True)
    args = parser.parse_args()
    private, public = Path(args.private).resolve(), Path(args.public)
    if private.parent != Path("/srv/cellforge-private") or private.exists() or public.exists():
        raise SystemExit("fresh private pilot required")
    private.mkdir(mode=0o700)
    raw = Path(args.source).read_bytes()
    unique = {hashlib.sha256(text.encode()).hexdigest(): text for text in json.loads(raw) if 150 <= len(text.encode()) <= 256}
    ordered = sorted(unique)
    if len(ordered) < 2:
        raise SystemExit("insufficient whole-family pilot data")
    selection = [{"id":ordered[0],"family":ordered[0],"kind":"next_byte","text":unique[ordered[0]]}]
    report = [{"id":ordered[1],"family":ordered[1],"kind":"next_byte","text":unique[ordered[1]]}]
    selection_raw, report_raw = json.dumps(selection,sort_keys=True).encode(), json.dumps(report,sort_keys=True).encode()
    (private/"selection.json").write_bytes(selection_raw)
    (private/"report-only.json").write_bytes(report_raw)
    graph = Graph.load(args.artifact)
    policy = {"protocol":"evolution-pilot-partition/1","source_sha256":hashlib.sha256(raw).hexdigest(),
        "selection_sha256":hashlib.sha256(selection_raw).hexdigest(),"report_only_sha256":hashlib.sha256(report_raw).hexdigest(),
        "partition":"first two lexicographic SHA256 whole-story families of150..256UTF8bytes; disjoint families",
        "exposure":"source is public dev_train; report-only pilot split, not unexposed final data",
        "report_policy":"never loaded into fitness; curves released only after all generations/winner hashes frozen",
        "founder_hash":graph.hash(),"metric":"UTF8 content-only bits per byte; EOS excluded; not MC likelihood accuracy",
        "source_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (private/"partition-policy.json").write_text(json.dumps(policy,sort_keys=True))
    started = time.monotonic()
    score = content_fitness(graph,selection)
    receipt = {"protocol":"evolution-gen0-calibration/1","policy":policy,"model_stats":graph.stats(),
               "generation":0,"fitness":score,"elapsed_seconds":time.monotonic()-started,
               "mutations":0,"gradient_steps":0,"report_evaluations":0,"final_evaluations":0}
    public.write_text(json.dumps(receipt,indent=2)+"\n")
    public.chmod(0o644)


if __name__ == "__main__":
    main()
