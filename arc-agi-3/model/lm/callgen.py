"""Constrained CALL decoding for the blackboard router.

Phase 1: free THINK generation until the model emits CALL (max 40 tokens).
Phase 2: masked-logit generation over the CALL grammar trie — the emitted
CALL is ALWAYS a valid act.* cell with schema-correct args. Parse-fail
rate is ~0 by construction.

Only act.* cells are reachable (perception moved to passive intake).
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tok import tokenize

import torch
import torch.nn.functional as F

# per-cell arg schemas: (key, lo, hi)
SCHEMAS = {
    "act.try": [("action", 0, 9)],
    "act.sweep": [],
    "act.click_at": [("x", 0, 63), ("y", 0, 63)],
    "act.pick_drop": [("x1", 0, 63), ("y1", 0, 63),
                      ("x2", 0, 63), ("y2", 0, 63)],
    "act.repeat_click": [("x", 0, 63), ("y", 0, 63), ("n", 1, 20)],
    "act.ghost_follow": [("x", 0, 63), ("y", 0, 63)],
    "act.reset": [],
}
CELLS = sorted(SCHEMAS)


class CallGrammar:
    """Precompiled token-id trie for the CALL line.

    Only cells whose name tokens ALL exist in the vocab are offered: the
    decoder never proposes a cell the model wasn't trained to use.
    """

    def __init__(self, vocab, stoi, cells=None):
        self.vocab, self.stoi = vocab, stoi

        def tid(tok):
            if tok not in stoi:
                raise ValueError(f"grammar token missing from vocab: {tok!r}")
            return stoi[tok]

        def tids(s):
            return [tid(t) for t in tokenize(s)]

        want = cells or sorted(SCHEMAS)
        self.cells = []
        for c in want:
            try:
                tids(c)
                for key, _lo, _hi in SCHEMAS[c]:
                    tids(f'"{key}"')
                self.cells.append(c)
            except ValueError as e:
                print(f"grammar: skipping {c} ({e})", flush=True)
        if not self.cells:
            raise ValueError("no usable cells in grammar")

        self.t_lbrace = tids("{")
        self.t_cell_key = tids('"cell"')
        self.t_args_key = tids('"args"')
        self.t_colon = tids(":")
        self.t_quote = tids('"')
        self.t_comma = tids(",")
        self.t_rbrace = tids("}")
        self.num_ids = {i: tid(str(i)) for i in range(64)}
        self.key_ids = {}
        need_keys = set()
        for c in self.cells:
            for key, _lo, _hi in SCHEMAS[c]:
                need_keys.add(key)
        for key in need_keys:
            self.key_ids[key] = tids(f'"{key}"')
        # cell-name trie over token-id sequences (only usable cells)
        self.trie = {}
        self.cell_of = {}
        for c in self.cells:
            ids = tids(c)
            node = self.trie
            for i in ids:
                node = node.setdefault(i, {})
            node["$"] = c
        print(f"grammar cells: {self.cells}", flush=True)

    def _sample_masked(self, model, idx, allowed, temperature, bonus=None):
        with torch.no_grad():
            logits, _ = model(idx[:, -model.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            mask = torch.full_like(logits, float("-inf"))
            mask[:, list(allowed)] = 0.0
            logits = logits + mask
            if bonus:
                for tid_, b in bonus.items():
                    if tid_ in allowed:
                        logits[:, tid_] += b
            if len(allowed) == 1 and not bonus:
                return next(iter(allowed))
            probs = F.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())

    def _emit(self, model, idx, id_list, temperature):
        for i in id_list:
            nxt = self._sample_masked(model, idx, {i}, temperature)
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
        return idx

    def generate_call(self, model, idx, temperature=0.7, copy_xy=None,
                      copy_bonus=2.0):
        """idx ends right after the CALL marker. Returns (idx, name, args).
        copy_xy: optional (x, y) from the novelty intake. When the chosen
        cell is act.ghost_follow (whose job is "click the marker"), the
        observed marker coordinates get a logit bonus -- a copy mechanism
        that makes the router read its intake instead of hallucinating
        memorized positions."""
        idx = self._emit(model, idx, self.t_lbrace, temperature)
        idx = self._emit(model, idx, self.t_cell_key, temperature)
        idx = self._emit(model, idx, self.t_colon, temperature)
        idx = self._emit(model, idx, self.t_quote, temperature)
        # walk the cell-name trie with masked sampling
        # (no cell name is a prefix of another, so the first terminal wins)
        node, name = self.trie, None
        while name is None:
            allowed = {k for k in node if k != "$"}
            nxt = self._sample_masked(model, idx, allowed, temperature)
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
            node = node[nxt]
            kids = [k for k in node if k != "$"]
            if "$" in node and not kids:
                name = node["$"]
        idx = self._emit(model, idx, self.t_quote, temperature)
        idx = self._emit(model, idx, self.t_comma, temperature)
        idx = self._emit(model, idx, self.t_args_key, temperature)
        idx = self._emit(model, idx, self.t_colon, temperature)
        idx = self._emit(model, idx, self.t_lbrace, temperature)
        args = {}
        schema = SCHEMAS[name]
        use_copy = (copy_xy is not None and name == "act.ghost_follow")
        for j, (key, lo, hi) in enumerate(schema):
            if j:
                idx = self._emit(model, idx, self.t_comma, temperature)
            idx = self._emit(model, idx, self.key_ids[key], temperature)
            idx = self._emit(model, idx, self.t_colon, temperature)
            allowed = {self.num_ids[i] for i in range(lo, hi + 1)}
            bonus = None
            if use_copy and key in ("x", "y"):
                want = copy_xy[0] if key == "x" else copy_xy[1]
                if want in self.num_ids:
                    bonus = {self.num_ids[want]: copy_bonus}
            nxt = self._sample_masked(model, idx, allowed, temperature,
                                      bonus=bonus)
            val = self.vocab[nxt]
            args[key] = int(val)
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
        idx = self._emit(model, idx, self.t_rbrace, temperature)
        idx = self._emit(model, idx, self.t_rbrace, temperature)
        return idx, name, args


CALL_END_RE = re.compile(r"CALL\s*$")


def generate_think_call(model, idx, vocab, stoi, grammar,
                        temperature=0.7, max_think=40, copy_xy=None):
    """Phase 1: free THINK until CALL appears. Phase 2: constrained CALL.
    Returns (full_idx, think_text, name, args, n_gen)."""
    n0 = idx.shape[1]
    for _ in range(max_think):
        idx_c = idx[:, -model.block_size:]
        with torch.no_grad():
            logits, _ = model(idx_c)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            v, _ = torch.topk(logits, 40)
            logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
        txt = " ".join(vocab[i] for i in idx[0, n0:].tolist())
        if CALL_END_RE.search(txt):
            break
    else:
        # model never emitted CALL: force it
        call_id = stoi.get("CALL")
        if call_id is not None:
            idx = torch.cat(
                [idx, torch.tensor([[call_id]], dtype=torch.long)], dim=1)
    txt = " ".join(vocab[i] for i in idx[0, n0:].tolist())
    m = CALL_END_RE.search(txt)
    think_text = txt[: m.start()].strip() if m else txt.strip()
    idx, name, args = grammar.generate_call(model, idx, temperature,
                                            copy_xy=copy_xy)
    return idx, think_text, name, args, idx.shape[1] - n0
