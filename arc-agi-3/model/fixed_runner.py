"""CellForge ARC-3 FIXED RUNNER (weights-only arena, orchestrator-owned).

ARC-3 sibling of /opt/cellforge/code/src/cellforge/reference/evaluator.py.
Same philosophy, adapted to interactive games:

  REFERENCE PATTERN (reused):                          ARC-3 ADAPTATION:
  - versioned evaluator (ref-eval/1)                  -> EVALUATOR_VERSION = "arc3-ref-eval/1";
                                                       protocol changes => new version, separate result epoch
  - weights as DATA: Graph registry + param store,    -> submission = fixed-arch checkpoint ("Graph equivalent"):
    hashed (sha256), versioned FORMAT                    pure tensors, sha256-hashed, versioned format id
  - model_hash in every result                        -> checkpoint sha256 in every RHAE_RESULT
  - dataset manifest hash                             -> game-list manifest hash
  - per-case failure reasons (categorized)            -> categorized failure reasons (parse_fail, budget_exhausted, ...)
  - latency accounting (total + ms/case)              -> wall-clock total + ms/decision (Kaggle 9h budget)
  - MACs accounting                                   -> param count + decisions (compute proxy)
  - contracts: unit-test components in isolation,    -> submission contract: shapes match fixed spec, all finite,
    batch-2 determinism                                 deterministic re-hash after load
  - protected surface (protocols.py): "Final-split   -> eval seeds drawn AFTER staking closes, never in researcher
    seeds are never stored here: the operator           context (same rule, noted lineage)
    supplies them on the evaluation server
    after an artifact is frozen."

Researchers submit ONLY weight files. This runner loads them into the FIXED
tiny architecture (4L/4H/d128, vocab 2209) and scores true RHAE.
No researcher Python ever executes.

RMT EXTENSION (Track 3, 2026-10-16): if the checkpoint contains mem params
("mem_init"), the runner auto-detects RMT mode (--rmt forces it): the model
is RMTGPT with n_mem=8 recurrent memory tokens. Memory is carried across
decisions WITHIN a level and reset to mem_init at each level start. The
prompt/CALL/OBS flow is otherwise byte-identical to the plain runner, so
RMT vs plain comparisons are apples-to-apples. --zero-mem-at K zeroes the
memory at decision K of each level (the memory load-bearing / anti-cheat
standard: a model that ignores its memory is unaffected; one that depends
on it degrades).

Usage:
  fixed_runner.py --checkpoint model.pt --games lf52,s5i5 --seeds 0 [--maxdec 60]
  fixed_runner.py --lora deltas.json --base lm/router_v3.pt --games ... --seeds ...
  fixed_runner.py --checkpoint lm/router_rmt.pt --games ... --seeds 0 --rmt
  fixed_runner.py --checkpoint lm/router_rmt.pt --games ... --seeds 0 --zero-mem-at 30
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm"))
from cells import GameState, call_cell, REGISTRY
from tok import load_vocab, encode, decode
from gpt import GPT
from gpt_rmt import RMTGPT, rmt_generate
from rhae_scorer import environment_score
import torch
from arc_agi import Arcade
from arcengine import GameState as GState

EVALUATOR_VERSION = "arc3-ref-eval/1"
FORMAT = "cellforge-arc3-ckpt/1"
BASE = "/srv/arc3-claude"
HERE = os.path.dirname(os.path.abspath(__file__))
CALL_RE = re.compile(r"CALL\s*(\{)")

# FIXED architecture — the canonical base researchers start from.
# gen2-gen2 (Sano directive): researchers may GROW the architecture through
# tensors alone. The runner builds the model from the submission's declared
# cfg, validated against GROWTH_BOUNDS below. Tokenizer is fixed, so
# vocab_size is fixed. Everything else (load-bearing, finiteness,
# determinism, param cap) is still enforced.
FIXED_CFG = {"vocab_size": 2209, "n_layer": 4, "n_head": 4, "n_embd": 128,
             "block_size": 192, "dropout": 0.0}
# RMT variant: same transformer, plus 8 recurrent memory tokens.
N_MEM = 8
RMT_CFG = dict(FIXED_CFG, n_mem=N_MEM)
# Architectural growth walls: submissions may declare any cfg inside these.
GROWTH_BOUNDS = {
    "n_layer": (1, 8),
    "n_head": (1, 8),
    "n_embd": (32, 512),
    "block_size": (64, 512),
    "n_mem": (1, 16),   # RMT checkpoints only
}
MAX_PARAMS = 10_000_000
CFG_KEYS = ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")
CFG_KEYS_RMT = CFG_KEYS + ("n_mem",)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_cfg(cfg, rmt):
    """A submission's declared cfg: exact keys, bounded, self-consistent.
    Returns (clean_cfg, msg); clean_cfg None on rejection."""
    want = CFG_KEYS_RMT if rmt else CFG_KEYS
    if not isinstance(cfg, dict):
        return None, "cfg not a dict"
    if set(cfg.keys()) != set(want):
        return None, f"cfg keys mismatch (got {sorted(cfg.keys())})"
    if cfg.get("vocab_size") != 2209:
        return None, "vocab_size must be 2209 (fixed tokenizer)"
    for k in ("n_layer", "n_head", "n_embd", "block_size"):
        v = cfg[k]
        lo, hi = GROWTH_BOUNDS[k]
        if not isinstance(v, int) or not (lo <= v <= hi):
            return None, f"cfg.{k}={v} out of bounds [{lo},{hi}]"
    if cfg["n_embd"] % cfg["n_head"] != 0:
        return None, "n_embd not divisible by n_head"
    dr = cfg.get("dropout", 0.0)
    if not isinstance(dr, (int, float)) or not (0.0 <= dr <= 0.5):
        return None, f"dropout {dr} out of bounds [0, 0.5]"
    if rmt:
        v = cfg["n_mem"]
        lo, hi = GROWTH_BOUNDS["n_mem"]
        if not isinstance(v, int) or not (lo <= v <= hi):
            return None, f"cfg.n_mem={v} out of bounds [{lo},{hi}]"
    return {k: cfg[k] for k in want}, "cfg OK"


def expected_shapes_for(cfg, rmt):
    m = RMTGPT(**cfg) if rmt else GPT(**cfg)
    return {k: tuple(v.shape) for k, v in m.state_dict().items()}


def submission_contract(sd, cfg, rmt=False):
    """The ARC-3 component contract: declared cfg validated and bounded;
    state shapes match the DECLARED cfg; all finite; param cap; deterministic."""
    clean, msg = validate_cfg(cfg, rmt)
    if clean is None:
        return False, msg
    exp = expected_shapes_for(clean, rmt)
    if set(sd.keys()) != set(exp.keys()):
        return False, f"keys mismatch (got {len(sd)}, want {len(exp)})"
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            return False, f"non-tensor at {k}"
        if tuple(v.shape) != exp[k]:
            return False, f"shape mismatch at {k}"
        if not bool(torch.isfinite(v).all()):
            return False, f"non-finite at {k}"
    n = sum(v.numel() for v in sd.values())
    if n > MAX_PARAMS:
        return False, f"param cap exceeded: {n} > {MAX_PARAMS}"
    return True, f"contract OK ({n} params)"


def purity_check_checkpoint(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        return None, None, f"REJECT: load failed: {e}", False
    if not isinstance(ckpt, dict) or "state" not in ckpt:
        return None, None, "REJECT: not a {cfg, state} checkpoint", False
    sd = ckpt["state"]
    is_rmt = "mem_init" in sd
    cfg = ckpt.get("cfg") or (dict(RMT_CFG) if is_rmt else dict(FIXED_CFG))
    ok, msg = submission_contract(sd, cfg, is_rmt)
    if not ok:
        return None, None, f"REJECT: {msg}", False
    clean, _ = validate_cfg(cfg, is_rmt)
    return sd, clean, "OK", is_rmt


def purity_check_lora(path, base_sd, base_cfg, base_rmt=False):
    try:
        deltas = json.load(open(path))
    except Exception as e:
        return None, f"REJECT: bad JSON: {e}"
    if not isinstance(deltas, dict):
        return None, "REJECT: LoRA must be a JSON object"
    exp = {k: tuple(v.shape) for k, v in base_sd.items()}
    sd = {k: v.clone() for k, v in base_sd.items()}
    for k, ab in deltas.items():
        if k not in exp:
            return None, f"REJECT: unknown tensor {k}"
        try:
            A = torch.tensor(ab["A"], dtype=torch.float32)
            B = torch.tensor(ab["B"], dtype=torch.float32)
        except Exception as e:
            return None, f"REJECT: bad A/B at {k}: {e}"
        if A.dim() != 2 or B.dim() != 2 or A.shape[1] != B.shape[0]:
            return None, f"REJECT: rank mismatch at {k}"
        if A.shape[1] > 8:
            return None, f"REJECT: rank {A.shape[1]} > 8 at {k}"
        d = A @ B
        if tuple(d.shape) != exp[k]:
            return None, f"REJECT: delta shape mismatch at {k}"
        sd[k] = sd[k] + d
    ok, msg = submission_contract(sd, base_cfg, base_rmt)
    if not ok:
        return None, f"REJECT: {msg}"
    return sd, "OK"


def extract_balanced(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_call(text):
    m = CALL_RE.search(text)
    if not m:
        return None, None
    blob = extract_balanced(text, m.start(1))
    if not blob:
        return None, None
    try:
        d = json.loads(re.sub(r"\s+", "", blob))
        name = d.get("cell")
        if name not in REGISTRY:
            return None, None
        return name, d.get("args", {})
    except Exception:
        return None, None


def run_game(model, game, seed, budget, maxdec, vocab, stoi, arc,
             rmt=False, zero_mem_at=None, rulebook=None):
    torch.manual_seed(seed)
    info = next(e for e in arc.get_environments()
                if e.game_id.split("-")[0] == game)
    baselines = [int(x) for x in info.baseline_actions]
    card = arc.create_scorecard()
    env = arc.make(info.game_id, seed=seed, scorecard_id=card)
    st = GameState(env, a_budget := budget)
    hist, ndec = [], 0
    failures = {}
    per_level, prev_lvl, prev_steps = {}, 0, 0
    # RMT state: memory carried across decisions, reset at each level start.
    mem, cur_lvl, dec_in_level = None, 0, 0
    t0 = time.perf_counter()
    while ndec < maxdec and st.steps_used < budget:
        if st.resp.state == GState.WIN:
            break
        if rmt:
            lvl_now = int(st.resp.levels_completed)
            if lvl_now != cur_lvl:
                mem = None            # level start: back to mem_init
                cur_lvl, dec_in_level = lvl_now, 0
        state_txt = call_cell(st, "perceive.describe", {})
        prompt = "".join(b + "\n" for b in hist[-2:])
        if rulebook:
            prompt += rulebook + "\n"
        prompt += "STATE " + state_txt + "\nTHINK "
        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        if idx.shape[1] > model.block_size:
            idx = idx[:, -model.block_size:]
        if rmt:
            if zero_mem_at is not None and dec_in_level == zero_mem_at:
                mem_in = torch.zeros(1, model.n_mem,
                                     model.wte.embedding_dim)
            else:
                mem_in = mem
            out, mem = rmt_generate(model, mem_in, idx, max_new_tokens=64,
                                    temperature=0.7, top_k=40)
        else:
            with torch.no_grad():
                out = model.generate(idx, max_new_tokens=64, temperature=0.7,
                                     top_k=40)
        gen = decode(out[0].tolist()[len(idx[0]):], vocab)
        name, args = parse_call(gen)
        if name is None:
            failures["parse_fail"] = failures.get("parse_fail", 0) + 1
            name, args = "perceive.describe", {}
        call_cell(st, name, args)
        lvl = int(st.resp.levels_completed)
        if lvl > prev_lvl:
            for L in range(prev_lvl + 1, lvl + 1):
                per_level[L] = st.steps_used - prev_steps
            prev_lvl, prev_steps = lvl, st.steps_used
        think_txt = gen.split("CALL")[0].strip()
        hist.append("STATE " + state_txt + "\nTHINK " + think_txt +
                    "\nCALL " + json.dumps({"cell": name, "args": args}))
        ndec += 1
        dec_in_level += 1
    if st.steps_used >= budget:
        failures["budget_exhausted"] = failures.get("budget_exhausted", 0) + 1
    dt = time.perf_counter() - t0
    env_score, level_scores = environment_score(baselines, per_level)
    return {"game": game, "seed": seed, "baselines": baselines,
            "per_level_actions": per_level,
            "level_scores": {str(k): round(v, 4) for k, v in level_scores.items()},
            "rhae_pct": round(env_score * 100, 4),
            "levels_completed": prev_lvl, "steps_used": st.steps_used,
            "failure_reasons": failures,
            "rmt": rmt, "zero_mem_at": zero_mem_at,
            "latency_s_total": round(dt, 3),
            "latency_ms_per_decision": round(1000 * dt / max(1, ndec), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--base", default=os.path.join(HERE, "lm", "router_v3.pt"))
    ap.add_argument("--games", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--maxdec", type=int, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rmt", action="store_true",
                    help="force RMT mode (auto-detected from checkpoint keys)")
    ap.add_argument("--zero-mem-at", type=int, default=None,
                    help="zero memory at decision K of each level (RMT only)")
    ap.add_argument("--rulebook", default=None,
                    help="rulebook v1 text file: injected into prompt context "
                         "(runner template, versioned; submission stays pure tensors)")
    a = ap.parse_args()

    rulebook_txt, rulebook_hash = None, None
    if a.rulebook:
        rulebook_txt = open(a.rulebook).read().strip()
        rulebook_hash = hashlib.sha256(rulebook_txt.encode()).hexdigest()[:16]
        print(f"RULEBOOK sha={rulebook_hash} chars={len(rulebook_txt)}",
              flush=True)
    base_sd, base_cfg, msg, base_rmt = purity_check_checkpoint(a.base)
    assert base_sd is not None, f"base {msg}"
    if a.lora:
        assert not base_rmt, "LoRA on an RMT base is not supported"
        sd, msg = purity_check_lora(a.lora, base_sd, base_cfg)
        src = f"lora:{os.path.basename(a.lora)}"
        model_hash = hashlib.sha256(open(a.lora, "rb").read()).hexdigest()[:16]
        rmt_mode = False
        model_cfg = base_cfg
    elif a.checkpoint:
        sd, ckpt_cfg, msg, is_rmt = purity_check_checkpoint(a.checkpoint)
        src = f"ckpt:{os.path.basename(a.checkpoint)}"
        model_hash = sha256_file(a.checkpoint)[:16]
        rmt_mode = bool(a.rmt or is_rmt)
        model_cfg = ckpt_cfg
        if a.rmt and not is_rmt:
            print("REJECT: --rmt forced but checkpoint has no mem params",
                  file=sys.stderr)
            sys.exit(3)
        if is_rmt:
            src += "+rmt"
    else:
        print("need --checkpoint or --lora", file=sys.stderr)
        sys.exit(2)
    if sd is None:
        print(msg, file=sys.stderr)
        sys.exit(3)
    print(f"PURITY {msg} src={src} model_hash={model_hash} rmt={rmt_mode}",
          flush=True)

    model = RMTGPT(**model_cfg) if rmt_mode else GPT(**model_cfg)
    model.load_state_dict(sd)
    model.eval()
    n_params = sum(p.numel() for p in sd.values())
    vocab, stoi = load_vocab(os.path.join(HERE, "lm", "vocab.json"))
    lg = logging.getLogger("q")
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    arc = Arcade(logger=lg, environments_dir=BASE + "/environment_files",
                 recordings_dir=BASE + "/recordings")

    games = [g.strip() for g in a.games.split(",")]
    game_manifest = hashlib.sha256(",".join(sorted(games)).encode()).hexdigest()[:16]
    results = []
    for game in games:
        for seed in [int(s) for s in a.seeds.split(",")]:
            r = run_game(model, game, seed, a.budget, a.maxdec, vocab, stoi,
                         arc, rmt=rmt_mode, zero_mem_at=a.zero_mem_at,
                         rulebook=rulebook_txt)
            r.update({"agent": src, "params": n_params, "model_hash": model_hash,
                      "evaluator": EVALUATOR_VERSION, "format": FORMAT,
                      "game_manifest": game_manifest, "rulebook": rulebook_hash})
            results.append(r)
            print("RHAE_RESULT " + json.dumps(r), flush=True)
    mean_rhae = sum(r["rhae_pct"] for r in results) / max(1, len(results))
    levels = sum(r["levels_completed"] for r in results)
    total_s = round(sum(r["latency_s_total"] for r in results), 1)
    summary = {"agent": src, "runs": len(results),
               "mean_rhae": round(mean_rhae, 4), "levels": levels,
               "evaluator": EVALUATOR_VERSION, "model_hash": model_hash,
               "wallclock_s_total": total_s, "results": results,
               "rulebook": rulebook_hash}
    print(f"SUMMARY runs={len(results)} mean_RHAE={mean_rhae:.2f} levels={levels} "
          f"wallclock={total_s}s", flush=True)
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)
        print(f"WROTE {a.out}", flush=True)


if __name__ == "__main__":
    main()
