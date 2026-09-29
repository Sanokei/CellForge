"""Expand my hand-written grammar (grammar.py) into a CellForge proposal.

Architecture (all tables are deterministic expansions of the grammar in grammar.py):
  state = [g (one-hot grammar belief state, G) | alive (token mask, Wn) | pos (one-hot char position, P)]
  consume prev:  alive &= spell[pos] matches prev ; pos += 1
                 if a token completed: g <- T[g, class(token)] (binary-coded lookup + decoder),
                 alive <- all, pos <- 0
  emit:          v[c] = sum_{alive tokens w with next char c} P(w | g);  logits = PWL-log(v)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import grammar as gr  # noqa: E402

VOCAB = ["<pad>", "<bos>", "<eos>", " ", ",", "."] + [chr(ord("a") + i) for i in range(26)]
CIDX = {c: i for i, c in enumerate(VOCAB)}
CIDX["\x02"] = 2
P = 11

# PWL log: knots k0 * RATIO^i ; floor = log(k0)
K0 = 1e-6
RATIO = 1.5
KMAX = 1e3
# end-of-document rule (my reading: a new sentence is started while the text is shorter than ~150 chars)
EOS_L = 151.0     # centre of the length gate
EOS_K = 1.0       # log-odds per character
EOS_A = 6.0       # clip of log-odds


def sparse(a, tol=0.0):
    idx = np.argwhere(np.abs(a) > tol)
    return {"sparse": [[*map(int, ix), float(a[tuple(ix)])] for ix in idx], "shape": list(a.shape)}


def build(out_path, author="r1-O1", rationale=""):
    states, trans = gr.build()
    part, mt = gr.minimize(trans)
    toks, classes = gr.token_classes(mt)
    G, Wn, C = len(mt), len(toks), len(classes)
    NB = max(1, math.ceil(math.log2(G)))
    tix = {t: i for i, t in enumerate(toks)}
    g0 = part[0]
    S = G + Wn + P
    assert max(len(t) for t in toks) <= P - 1

    spell = np.zeros((Wn * 32, P))
    lenm = np.zeros((Wn, P))
    for w, t in enumerate(toks):
        for p, ch in enumerate(t):
            spell[w * 32 + CIDX[ch], p] = 1.0
        lenm[w, len(t) - 1] = 1.0
    cls = np.zeros((C, Wn))
    tclass = {}
    for c, ts in enumerate(classes):
        for t in ts:
            cls[c, tix[t]] = 1.0
            tclass[t] = c
    Tcode = np.zeros((NB, G * C))
    Pw = np.zeros((Wn, G))
    for gi, row in enumerate(mt):
        for t, (p, nj) in row.items():
            Pw[tix[t], gi] = p
            if nj is not None:
                for b in range(NB):
                    if (nj >> b) & 1:
                        Tcode[b, gi * C + tclass[t]] = 1.0
    dec = np.zeros((G, NB))
    decb = np.zeros(G)
    for j in range(G):
        bits = [(j >> b) & 1 for b in range(NB)]
        dec[j] = [1.0 if x else -1.0 for x in bits]
        decb[j] = -(sum(bits) - 0.5)
    init = np.zeros(S)
    init[g0] = 1.0
    init[G:G + Wn] = 1.0
    init[G + Wn] = 1.0
    sepv = np.zeros((1, 32))
    for ch in " ,.":
        sepv[0, CIDX[ch]] = 1.0
    knots = [K0 * RATIO ** i for i in range(int(math.ceil(math.log(KMAX / K0) / math.log(RATIO))) + 1)]
    slopes = [(math.log(knots[i + 1]) - math.log(knots[i])) / (knots[i + 1] - knots[i])
              for i in range(len(knots) - 1)]
    H = len(slopes)
    c = [slopes[0]] + [slopes[i] - slopes[i - 1] for i in range(1, H)]

    ops = [{"op": "delete_param", "name": "boot_logits"}]

    def sp(name, arr, dense=False):
        arr = np.asarray(arr, dtype=np.float64)
        if dense:
            ops.append({"op": "set_param", "name": name, "value": arr.tolist()})
        else:
            ops.append({"op": "set_param", "name": name, "pattern": sparse(arr)})

    def zeros(name, n):
        ops.append({"op": "set_param", "name": name, "pattern": {"fill": 0.0, "shape": [n]}})

    sp("spell.W", spell); zeros("spell.b", Wn * 32)
    sp("len.W", lenm); zeros("len.b", Wn)
    sp("cls.W", cls); zeros("cls.b", C)
    sp("trans.W", Tcode); zeros("trans.b", NB)
    sp("dec.W", dec, dense=True); sp("dec.b", decb, dense=True)
    sp("pw.W", Pw); zeros("pw.b", Wn)
    sp("sep.W", sepv); zeros("sep.b", 1)
    sp("init", init)
    ops.append({"op": "set_param", "name": "ones_w", "pattern": {"fill": 1.0, "shape": [Wn]}})
    ops.append({"op": "set_param", "name": "one", "pattern": {"fill": 1.0, "shape": [1]}})
    e0 = np.zeros(P); e0[0] = 1.0
    sp("e0", e0, dense=True)
    sp("eosg.W1", [[EOS_K], [EOS_K]], dense=True)
    sp("eosg.b1", [-EOS_K * EOS_L + EOS_A, -EOS_K * EOS_L - EOS_A], dense=True)
    sp("eosg.W2", [[1.0, -1.0]], dense=True)
    sp("eosg.b2", [-EOS_A], dense=True)
    eosv = np.zeros(32); eosv[2] = 1.0
    sp("eosv", eosv, dense=True)
    ops.append({"op": "set_param", "name": "log.W1", "pattern": {"fill": 1.0, "shape": [H, 1]}})
    sp("log.b1", [-k for k in knots[:H]], dense=True)
    sp("log.W2", [c], dense=True)
    sp("log.b2", [math.log(K0)], dense=True)

    def cell(name, prm, act="identity", two=False):
        d = {"kind": "cell", "act": act, "params": {"W1": prm + ".W1" if two else prm + ".W",
                                                    "b1": prm + ".b1" if two else prm + ".b"}}
        if two:
            d["params"].update({"W2": prm + ".W2", "b2": prm + ".b2"})
        ops.append({"op": "put_def", "name": name, "def": d})

    cell("c_spell", "spell"); cell("c_len", "len"); cell("c_cls", "cls"); cell("c_trans", "trans")
    cell("c_dec", "dec", act="step"); cell("c_pw", "pw"); cell("c_sep", "sep"); cell("c_log", "log", act="relu", two=True)
    cell("c_eosg", "eosg", act="relu", two=True)

    def n(id_, use, ins=None, **args):
        d = {"id": id_, "use": use}
        if ins:
            d["in"] = ins
        if args:
            d["args"] = args
        return d

    # --- bundle: read the next-char table for a position
    ops.append({"op": "put_def", "name": "b_chartable", "def": {"kind": "bundle", "inputs": ["pos"], "nodes": [
        n("t", "c_spell", {"x": "$pos"}), n("r", "op:reshape", {"x": "t"}, shape=[Wn, 32])],
        "outputs": {"table": "r"}}})
    # --- bundle: grammar transition g, completed token mask -> next g
    ops.append({"op": "put_def", "name": "b_grammar", "def": {"kind": "bundle", "inputs": ["g", "tok"], "nodes": [
        n("cl", "c_cls", {"x": "$tok"}),
        n("gc", "op:reshape", {"x": "$g"}, shape=[G, 1]),
        n("cc", "op:reshape", {"x": "cl"}, shape=[1, C]),
        n("pr", "op:matmul", {"a": "gc", "b": "cc"}),
        n("pf", "op:reshape", {"x": "pr"}, shape=[G * C]),
        n("code", "c_trans", {"x": "pf"}),
        n("gn", "c_dec", {"x": "code"})], "outputs": {"g": "gn"}}})
    # --- bundle: consume prev, update state
    ops.append({"op": "put_def", "name": "b_consume", "def": {"kind": "bundle", "inputs": ["state", "prev"], "nodes": [
        n("g", "op:slice", {"x": "$state"}, axis=0, start=0, stop=G),
        n("al", "op:slice", {"x": "$state"}, axis=0, start=G, stop=G + Wn),
        n("ps", "op:slice", {"x": "$state"}, axis=0, start=G + Wn, stop=S),
        n("tab", "b_chartable", {"pos": "ps"}),
        n("pc", "op:reshape", {"x": "$prev"}, shape=[32, 1]),
        n("m", "op:matmul", {"a": "tab.table", "b": "pc"}),
        n("m2", "op:reshape", {"x": "m"}, shape=[Wn]),
        n("al1", "op:mul", {"a": "al", "b": "m2"}),
        n("ps1", "op:shift", {"x": "ps"}, axis=0, k=1),
        n("lv", "c_len", {"x": "ps"}),
        n("dv", "op:mul", {"a": "al1", "b": "lv"}),
        n("done", "op:sum", {"x": "dv"}, axis=0, keepdims=True),
        n("gr", "b_grammar", {"g": "g", "tok": "al1"}),
        n("dg", "op:sub", {"a": "gr.g", "b": "g"}),
        n("dg2", "op:mul", {"a": "dg", "b": "done"}),
        n("g1", "op:add", {"a": "g", "b": "dg2"}),
        n("tot", "op:sum", {"x": "al1"}, axis=0, keepdims=True),
        n("any", "op:step", {"x": "tot"}),
        n("onep", "param:one"),
        n("na", "op:sub", {"a": "onep", "b": "any"}),
        n("sep", "c_sep", {"x": "$prev"}),
        n("dsep", "op:mul", {"a": "na", "b": "sep"}),
        n("rs", "op:add", {"a": "done", "b": "dsep"}),
        n("onesw", "param:ones_w"),
        n("da", "op:sub", {"a": "onesw", "b": "al1"}),
        n("da2", "op:mul", {"a": "da", "b": "rs"}),
        n("al2", "op:add", {"a": "al1", "b": "da2"}),
        n("e0", "param:e0"),
        n("dp", "op:sub", {"a": "e0", "b": "ps1"}),
        n("dp2", "op:mul", {"a": "dp", "b": "rs"}),
        n("ps2", "op:add", {"a": "ps1", "b": "dp2"}),
        n("new", "op:concat", {"x0": "g1", "x1": "al2", "x2": "ps2"}, axis=0),
        n("bos", "op:slice", {"x": "$prev"}, axis=0, start=1, stop=2),
        n("ini", "param:init"),
        n("bi", "op:mul", {"a": "ini", "b": "bos"}),
        n("ns", "op:add", {"a": "new", "b": "bi"})], "outputs": {"state": "ns"}}})
    # --- bundle: emit logits from the new state
    ops.append({"op": "put_def", "name": "b_eos", "def": {"kind": "bundle", "inputs": ["v", "step"], "nodes": [
        n("z", "c_eosg", {"x": "$step"}),
        n("r", "op:exp", {"x": "z"}),
        n("onep", "param:one"),
        n("rm", "op:sub", {"a": "r", "b": "onep"}),
        n("ev", "param:eosv"),
        n("ve0", "op:mul", {"a": "$v", "b": "ev"}),
        n("ve", "op:sum", {"x": "ve0"}, axis=0, keepdims=True),
        n("k", "op:mul", {"a": "ve", "b": "rm"}),
        n("ad", "op:mul", {"a": "ev", "b": "k"}),
        n("v2", "op:add", {"a": "$v", "b": "ad"})], "outputs": {"v": "v2"}}})
    ops.append({"op": "put_def", "name": "b_emit", "def": {"kind": "bundle", "inputs": ["state", "step"], "nodes": [
        n("g", "op:slice", {"x": "$state"}, axis=0, start=0, stop=G),
        n("al", "op:slice", {"x": "$state"}, axis=0, start=G, stop=G + Wn),
        n("ps", "op:slice", {"x": "$state"}, axis=0, start=G + Wn, stop=S),
        n("tab", "b_chartable", {"pos": "ps"}),
        n("pw", "c_pw", {"x": "g"}),
        n("u", "op:mul", {"a": "al", "b": "pw"}),
        n("u2", "op:reshape", {"x": "u"}, shape=[1, Wn]),
        n("v", "op:matmul", {"a": "u2", "b": "tab.table"}),
        n("v1", "op:reshape", {"x": "v"}, shape=[32]),
        n("vg", "b_eos", {"v": "v1", "step": "$step"}),
        n("v2", "op:reshape", {"x": "vg.v"}, shape=[32, 1]),
        n("lg", "c_log", {"x": "v2"}),
        n("lg2", "op:reshape", {"x": "lg"}, shape=[32])], "outputs": {"logits": "lg2"}}})
    ops.append({"op": "put_def", "name": "step_root", "def": {"kind": "bundle",
        "inputs": ["ctx", "prev", "step", "state"], "nodes": [
            n("c", "b_consume", {"state": "$state", "prev": "$prev"}),
            n("e", "b_emit", {"state": "c.state", "step": "$step"})],
        "outputs": {"logits": "e.logits", "state": "c.state"}}})
    ops.append({"op": "set_state_dim", "value": S})
    prop = {"author": author, "rationale": rationale, "ops": ops}
    Path(out_path).write_text(json.dumps(prop))
    print(json.dumps({"G": G, "Wn": Wn, "C": C, "NB": NB, "S": S, "H": H, "ops": len(ops)}))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "proposal.json",
          rationale=sys.argv[2] if len(sys.argv) > 2 else "")
