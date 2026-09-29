"""Turn my hand-chosen values into proposal JSON (no dataset access)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from templates import T

CFG = dict(
    tops=[6, 7, 8, 9, 10],     # candidate glyph top rows
    band_in=1.0, band_out=-2.0, vtemp=0.5,
    edge_thr=1.5,              # column-ink threshold for left edge
    pitch=12.04, cell_off=-2.0,  # cell start = x0 + off + pitch*step
    win=3.5,                   # +-3 px alignment search
    eos_lo=2.5, eos_hi=8.5,    # central cell region for ink test
    eos_gain=2.0, eos_thr=5.0,
    wmap={"X": 1.0, ".": -1.0, "o": 0.0},
)


def build(cfg=CFG, out=None):
    ops = []
    P = lambda n, v: ops.append({"op": "set_param", "name": n, "value": v})
    # vertical band detector
    Wv = []
    for t in cfg["tops"]:
        r = [0.0] * 32
        for y in range(t, t + 15):
            r[y] = cfg["band_in"]
        if t - 1 >= 0:
            r[t - 1] = cfg["band_out"]
        if t + 15 < 32:
            r[t + 15] = cfg["band_out"]
        Wv.append(r)
    P("vband.W1", Wv); P("vband.b1", [0.0] * len(cfg["tops"]))
    # glyph templates: score = T.(2I-1)/N
    W, b = [], []
    for d in "0123456789":
        flat = [cfg["wmap"][ch] for row in T[d] for ch in row]
        n = sum(abs(v) for v in flat)
        W.append([2 * v / n for v in flat]); b.append(-sum(flat) / n)
    P("glyph.W1", W); P("glyph.b1", b)
    P("locate.W1", [[1.0, cfg["pitch"]]]); P("locate.b1", [cfg["cell_off"]])
    P("eos.W1", [[-cfg["eos_gain"]]]); P("eos.b1", [cfg["eos_gain"] * cfg["eos_thr"]])
    P("special_logits", [-50.0, -50.0])
    ops.append({"op": "delete_param", "name": "boot_logits"})

    D = lambda n, d: ops.append({"op": "put_def", "name": n, "def": d})
    D("vband_cell", {"kind": "cell", "act": "identity", "params": {"W1": "vband.W1", "b1": "vband.b1"}})
    D("glyph_cell", {"kind": "cell", "act": "identity", "params": {"W1": "glyph.W1", "b1": "glyph.b1"}})
    D("locate_cell", {"kind": "cell", "act": "identity", "params": {"W1": "locate.W1", "b1": "locate.b1"}})
    D("eos_cell", {"kind": "cell", "act": "identity", "params": {"W1": "eos.W1", "b1": "eos.b1"}})

    K = len(cfg["tops"])
    nodes = [{"id": "rows", "use": "op:sum", "in": {"x": "$image"}, "args": {"axis": 1}},
             {"id": "vs", "use": "vband_cell", "in": {"x": "rows"}},
             {"id": "vw", "use": "op:softmax", "in": {"x": "vs"}, "args": {"axis": 0, "temp": cfg["vtemp"]}},
             {"id": "vw3", "use": "op:reshape", "in": {"x": "vw"}, "args": {"shape": [K, 1, 1]}}]
    cat = {}
    for i, t in enumerate(cfg["tops"]):
        nodes.append({"id": f"s{i}", "use": "op:slice", "in": {"x": "$image"}, "args": {"axis": 0, "start": t, "stop": t + 15}})
        nodes.append({"id": f"e{i}", "use": "op:expand", "in": {"x": f"s{i}"}, "args": {"axis": 0}})
        cat[f"x{i}"] = f"e{i}"
    nodes += [{"id": "stack", "use": "op:concat", "in": cat, "args": {"axis": 0}},
              {"id": "wt", "use": "op:mul", "in": {"a": "stack", "b": "vw3"}},
              {"id": "al", "use": "op:sum", "in": {"x": "wt"}, "args": {"axis": 0}}]
    D("vert_align", {"kind": "bundle", "inputs": ["image"], "nodes": nodes, "outputs": {"aligned": "al"}})

    nodes, cat = [], {}
    for dx in range(12):
        nodes.append({"id": f"h{dx}", "use": "op:shift", "in": {"x": "$aligned"}, "args": {"axis": 1, "k": -dx, "value": 0.0}})
        nodes.append({"id": f"g{dx}", "use": "op:expand", "in": {"x": f"h{dx}"}, "args": {"axis": 0}})
        cat[f"x{dx}"] = f"g{dx}"
    nodes += [{"id": "st", "use": "op:concat", "in": cat, "args": {"axis": 0}},
              {"id": "tr", "use": "op:transpose", "in": {"x": "st"}, "args": {"perm": [2, 1, 0]}},
              {"id": "fl", "use": "op:reshape", "in": {"x": "tr"}, "args": {"shape": [128, 180]}}]
    D("col_windows", {"kind": "bundle", "inputs": ["aligned"], "nodes": nodes, "outputs": {"win": "fl"}})

    D("left_edge", {"kind": "bundle", "inputs": ["image"], "nodes": [
        {"id": "ci", "use": "op:sum", "in": {"x": "$image"}, "args": {"axis": 0}},
        {"id": "cb", "use": "op:bias", "in": {"x": "ci"}, "args": {"c": -cfg["edge_thr"]}},
        {"id": "bin", "use": "op:step", "in": {"x": "cb"}},
        {"id": "cm", "use": "op:cummax", "in": {"x": "bin"}, "args": {"axis": 0}},
        {"id": "nm", "use": "op:neg", "in": {"x": "cm"}},
        {"id": "before", "use": "op:bias", "in": {"x": "nm"}, "args": {"c": 1.0}},
        {"id": "x0", "use": "op:sum", "in": {"x": "before"}, "args": {"axis": 0, "keepdims": True}}],
        "outputs": {"x0": "x0", "colink": "ci"}})

    D("encode_root", {"kind": "bundle", "inputs": ["image"], "nodes": [
        {"id": "va", "use": "vert_align", "in": {"image": "$image"}},
        {"id": "cw", "use": "col_windows", "in": {"aligned": "va.aligned"}},
        {"id": "gm", "use": "glyph_cell", "in": {"x": "cw.win"}},
        {"id": "le", "use": "left_edge", "in": {"image": "$image"}}],
        "outputs": {"S": "gm", "colink": "le.colink", "x0": "le.x0"}})

    D("cell_locator", {"kind": "bundle", "inputs": ["x0", "step"], "nodes": [
        {"id": "xs", "use": "op:concat", "in": {"x0": "$x0", "x1": "$step"}, "args": {"axis": 0}},
        {"id": "t", "use": "locate_cell", "in": {"x": "xs"}},
        {"id": "pos", "use": "op:coords", "args": {"n": 128, "start": 0.0, "step": 1.0}},
        {"id": "d", "use": "op:sub", "in": {"a": "pos", "b": "t"}}],
        "outputs": {"d": "d"}})
    D("char_head", {"kind": "bundle", "inputs": ["S", "d"], "nodes": [
        {"id": "ad", "use": "op:abs", "in": {"x": "$d"}},
        {"id": "na", "use": "op:neg", "in": {"x": "ad"}},
        {"id": "wb", "use": "op:bias", "in": {"x": "na"}, "args": {"c": cfg["win"]}},
        {"id": "w", "use": "op:step", "in": {"x": "wb"}},
        {"id": "w1", "use": "op:bias", "in": {"x": "w"}, "args": {"c": -1.0}},
        {"id": "pen", "use": "op:scale", "in": {"x": "w1"}, "args": {"c": 100.0}},
        {"id": "pe", "use": "op:expand", "in": {"x": "pen"}, "args": {"axis": 1}},
        {"id": "sp", "use": "op:add", "in": {"a": "$S", "b": "pe"}},
        {"id": "mx", "use": "op:max", "in": {"x": "sp"}, "args": {"axis": 0}}],
        "outputs": {"scores": "mx"}})
    D("eos_head", {"kind": "bundle", "inputs": ["colink", "d"], "nodes": [
        {"id": "a", "use": "op:bias", "in": {"x": "$d"}, "args": {"c": -cfg["eos_lo"]}},
        {"id": "sa", "use": "op:step", "in": {"x": "a"}},
        {"id": "nd", "use": "op:neg", "in": {"x": "$d"}},
        {"id": "b", "use": "op:bias", "in": {"x": "nd"}, "args": {"c": cfg["eos_hi"]}},
        {"id": "sb", "use": "op:step", "in": {"x": "b"}},
        {"id": "m", "use": "op:mul", "in": {"a": "sa", "b": "sb"}},
        {"id": "mi", "use": "op:mul", "in": {"a": "m", "b": "$colink"}},
        {"id": "ink", "use": "op:sum", "in": {"x": "mi"}, "args": {"axis": 0, "keepdims": True}},
        {"id": "e", "use": "eos_cell", "in": {"x": "ink"}}],
        "outputs": {"eos": "e"}})
    D("step_root", {"kind": "bundle", "inputs": ["S", "colink", "x0", "step", "prev", "state"], "nodes": [
        {"id": "loc", "use": "cell_locator", "in": {"x0": "$x0", "step": "$step"}},
        {"id": "ch", "use": "char_head", "in": {"S": "$S", "d": "loc.d"}},
        {"id": "eo", "use": "eos_head", "in": {"colink": "$colink", "d": "loc.d"}},
        {"id": "sp", "use": "param:special_logits"},
        {"id": "lg", "use": "op:concat", "in": {"x0": "sp", "x1": "eo.eos", "x2": "ch.scores"}, "args": {"axis": 0}}],
        "outputs": {"logits": "lg", "state": "$state"}})

    prop = {"author": "r1-iter-a",
            "rationale": "Hand-built template reader: vertical band alignment (softmax over 5 top rows), 12x15 sliding-window Hamming-style glyph templates at every column, left-edge detection plus fixed 12.04px pitch to locate cell k with +-3px max search, and an ink test in the cell centre for EOS.",
            "ops": ops}
    if out:
        with open(out, "w") as f:
            json.dump(prop, f)
    return prop


if __name__ == "__main__":
    build(out=sys.argv[1])
