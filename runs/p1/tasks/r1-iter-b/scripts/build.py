"""Build proposal.json from hand-written glyph templates and hand-chosen constants.

Nothing here reads dataset images or labels. Templates are ascii drawings typed by
the maker after viewing dev PNGs; all numbers are chosen by the maker.
"""
import json, sys, os

# ---- hand-chosen constants
ROW_THR = 0.5      # rowmax threshold to find the glyph top row
COL_THR = 0.5      # colmax threshold to find the leftmost ink column
GH, CW = 15, 12    # glyph rows, cell pitch (px)
NCELL = 8
MARGIN = 4         # aligned column index of leftmost ink
NOFF = 4           # candidate left bearings 0..3
LSPAN = 24         # leftmost-ink search span (columns)
AW = 104           # aligned width
PRES_THR = 4.0     # presence threshold on mid-cell ink sum
BOOST = 1000.0
EOS_LOGIT = 500.0
MAPV = {'#': 1.0, '+': 1.0, ':': 0.0, '.': 0.0, ' ': -1.0}

G = {}
G['0'] = ["   .+##+.   ","  .######.  ","  ##+  +##  ","  ##    ##. "," :#:    :#: "," :#:    .#+ "," +#.    .#+ "," +#. ## .#+ "," +#. ## .#+ "," :#:    .#+ "," :#:    :#: "," .##    ##. ","  ##+  +##  ","  .######.  ","   .+##+.   "]
G['1'] = ["   .+##+    ","  +####+    ","  :+..#+    ","     .#+    ","     .#+    ","     .#+    ","     .#+    ","     .#+    ","     .#+    ","     .#+    ","     .#+    ","     .#+    ","     .#+    ","  .#######+ ","  .#######+ "]
G['2'] = ["  .+###+    "," .#######.  "," .+.   +##  ","        ##  ","        ##. ","        ##  ","       :#:  ","      .#+   ","      ##    ","     ##     ","    ##      ","   ##       ","  ##        "," :########. "," :########. "]
G['3'] = ["  .+###+.   "," .#######.  "," .+.   :##  ","        ##  ","        ##  ","       +#+  ","    ####+   ","    ####+   ","       +##  ","        +#. ","        :#: ","        +#: "," ::.   +##  "," +#######:  ","  :+###+.   "]
G['4'] = ["      :##   ","      ###   ","     +###   ","    .#.##   ","    ++ ##   ","   .#  ##   ","   #:  ##   ","  :#   ##   ","  #:   ##   "," +#    ##   "," ########## "," ########## ","       ##   ","       ##   ","       ##   "]
G['5'] = ["  #######:  ","  #######:  ","  ##        ","  ##        ","  ##        ","  #####+    ","  #######.  ","  +.  .###  ","        ##. ","        +#. ","        +#. ","        ##. "," ::.  .+##  "," :#######.  ","  :####:    "]
G['6'] = ["    :###+   ","   ######+  ","  +##.  .:  ","  ##        "," .#+        "," :#.        "," +#..###:   "," +#:######  "," +##:  .##: "," +#+    .#+ "," :#:    .#+ "," .#+    :#+ ","  ##:  .##. ","  .######+  ","   .+###:   "]
G['7'] = [" +########: "," +########. ","        ##  ","       .#:  ","       +#   ","       ##   ","      :#:   ","      ##    ","      #+    ","     :#.    ","     ##     ","    .#+     ","    +#.     ","    ##      ","   :#:      "]
G['8'] = ["   .###+.   ","  +######+  ","  ##.  .##. "," .#+    +#: "," .#+    +#. ","  +#.  .#+  ","   :####+   ","   +####+.  ","  ##.  .##  "," :#:    :#: "," +#.    .#+ "," +#:    :#+ "," :##.  .##: ","  +######+  ","   :####:   "]
G['9'] = ["   :###+    ","  +######.  "," :##.  :##  "," +#.    +#. "," +#     :#: "," +#.    +#+ "," :##.  :##+ ","  ######:#+ ","   :###..#+ ","        :#: ","        +#. ","       .##  ","  :.  .##:  ","  +#####+   ","   +###:    "]

W1, b1 = [], []
for d in '0123456789':
    rows = G[d]
    assert len(rows) == GH and all(len(r) == CW for r in rows), d
    v = [MAPV[c] for r in rows for c in r]
    W1.append(v)
    n_ink = sum(1 for x in v if x > 0); n_bg = sum(1 for x in v if x < 0)
    b1.append(0.5 * (n_bg - n_ink))   # score = sum_ink (x-0.5) + sum_bg (0.5-x)
# presence row: ones over cell columns 2..9, all rows
W1.append([1.0 if 2 <= c <= 9 else 0.0 for r in range(GH) for c in range(CW)])
b1.append(0.0)

# vertical crop basis: Pv[t, r*32+y] = 1 iff y == t + r
pv = [[t, r * 32 + t + r, 1.0] for t in range(32) for r in range(GH) if t + r < 32]
# horizontal align basis: Ph[t, x*AW+j] = 1 iff x == t + j - MARGIN
ph = []
for t in range(LSPAN):
    for j in range(AW):
        x = t + j - MARGIN
        if 0 <= x < 128:
            ph.append([t, x * AW + j, 1.0])


def n(id_, use, inp=None, **args):
    d = {"id": id_, "use": use}
    if inp:
        d["in"] = inp
    if args:
        d["args"] = args
    return d


defs = {}
defs["cell_templates"] = {"kind": "cell", "act": "identity",
                          "params": {"W1": "tmpl.W1", "b1": "tmpl.b1"}}
# first-index one-hot of a thresholded profile
defs["first_on"] = {"kind": "bundle", "inputs": ["p"], "nodes": [
    n("ind", "op:step", {"x": "$p"}),
    n("cs", "op:cumsum", {"x": "ind"}, axis=0),
    n("neg", "op:neg", {"x": "cs"}),
    n("lim", "op:bias", {"x": "neg"}, c=1.5),
    n("first", "op:step", {"x": "lim"}),
    n("oh", "op:mul", {"a": "ind", "b": "first"})],
    "outputs": {"oh": "oh"}}
defs["align"] = {"kind": "bundle", "inputs": ["image"], "nodes": [
    n("rmax", "op:max", {"x": "$image"}, axis=1),
    n("rthr", "op:bias", {"x": "rmax"}, c=-ROW_THR),
    n("top", "first_on", {"p": "rthr"}),
    n("top2", "op:expand", {"x": "top.oh"}, axis=0),
    n("pv", "param:align.Pv"),
    n("vsel", "op:matmul", {"a": "top2", "b": "pv"}),
    n("vsel2", "op:reshape", {"x": "vsel"}, shape=[GH, 32]),
    n("crop", "op:matmul", {"a": "vsel2", "b": "$image"}),
    n("cmax", "op:max", {"x": "crop"}, axis=0),
    n("cthr", "op:bias", {"x": "cmax"}, c=-COL_THR),
    n("left", "first_on", {"p": "cthr"}),
    n("left24", "op:slice", {"x": "left.oh"}, axis=0, start=0, stop=LSPAN),
    n("left2", "op:expand", {"x": "left24"}, axis=0),
    n("ph", "param:align.Ph"),
    n("hsel", "op:matmul", {"a": "left2", "b": "ph"}),
    n("hsel2", "op:reshape", {"x": "hsel"}, shape=[128, AW]),
    n("al", "op:matmul", {"a": "crop", "b": "hsel2"})],
    "outputs": {"aligned": "al"}}
defs["read_cells"] = {"kind": "bundle", "inputs": ["x"], "nodes": [
    n("sl", "op:slice", {"x": "$x"}, axis=1, start=MARGIN, stop=MARGIN + CW * NCELL),
    n("r1", "op:reshape", {"x": "sl"}, shape=[GH, NCELL, CW]),
    n("tr", "op:transpose", {"x": "r1"}, perm=[1, 0, 2]),
    n("fl", "op:reshape", {"x": "tr"}, shape=[NCELL, GH * CW]),
    n("sc", "cell_templates", {"x": "fl"}),
    n("ex", "op:expand", {"x": "sc.y"}, axis=0)],
    "outputs": {"scores": "ex"}}
rn = []
for o in range(NOFF):
    src = "$aligned"
    if o:
        rn.append(n(f"sh{o}", "op:shift", {"x": "$aligned"}, axis=1, k=o, value=0.0))
        src = f"sh{o}"
    rn.append(n(f"rd{o}", "read_cells", {"x": src}))
rn.append(n("all", "op:concat", {f"x{o}": f"rd{o}.scores" for o in range(NOFF)}, axis=0))
rn += [
    n("c0", "op:slice", {"x": "all"}, axis=1, start=0, stop=1),
    n("c0d", "op:slice", {"x": "c0"}, axis=2, start=0, stop=10),
    n("c0m", "op:max", {"x": "c0d"}, axis=2),
    n("c0f", "op:reshape", {"x": "c0m"}, shape=[NOFF]),
    n("w", "op:onehot_argmax", {"x": "c0f"}, axis=0),
    n("w2", "op:expand", {"x": "w"}, axis=0),
    n("allf", "op:reshape", {"x": "all"}, shape=[NOFF, NCELL * 11]),
    n("sel", "op:matmul", {"a": "w2", "b": "allf"}),
    n("selr", "op:reshape", {"x": "sel"}, shape=[NCELL, 11])]
defs["reader"] = {"kind": "bundle", "inputs": ["aligned"], "nodes": rn, "outputs": {"cells": "selr"}}
defs["to_table"] = {"kind": "bundle", "inputs": ["cells"], "nodes": [
    n("dig", "op:slice", {"x": "$cells"}, axis=1, start=0, stop=10),
    n("pres", "op:slice", {"x": "$cells"}, axis=1, start=10, stop=11),
    n("pb", "op:bias", {"x": "pres"}, c=-PRES_THR),
    n("pr", "op:step", {"x": "pb"}),
    n("boost", "op:scale", {"x": "pr"}, c=BOOST),
    n("dl", "op:add", {"a": "dig", "b": "boost"}),
    n("dlp", "op:pad", {"x": "dl"}, axis=0, before=0, after=12 - NCELL, value=0.0),
    n("eos", "param:table.eos"),
    n("pbos", "param:table.padbos"),
    n("tab", "op:concat", {"x0": "pbos", "x1": "eos", "x2": "dlp"}, axis=1)],
    "outputs": {"table": "tab"}}
defs["encode_root"] = {"kind": "bundle", "inputs": ["image"], "nodes": [
    n("al", "align", {"image": "$image"}),
    n("rd", "reader", {"aligned": "al.aligned"}),
    n("tb", "to_table", {"cells": "rd.cells"})],
    "outputs": {"table": "tb.table"}}
defs["step_root"] = {"kind": "bundle", "inputs": ["table", "step", "prev", "state"], "nodes": [
    n("pos", "op:coords", {}, n=12, start=0.0, step=1.0),
    n("d", "op:sub", {"a": "pos", "b": "$step"}),
    n("ad", "op:abs", {"x": "d"}),
    n("nd", "op:neg", {"x": "ad"}),
    n("nb", "op:bias", {"x": "nd"}, c=0.5),
    n("oh", "op:step", {"x": "nb"}),
    n("oh2", "op:expand", {"x": "oh"}, axis=0),
    n("lg", "op:matmul", {"a": "oh2", "b": "$table"}),
    n("lgr", "op:reshape", {"x": "lg"}, shape=[13])],
    "outputs": {"logits": "lgr", "state": "$state"}}

ops = [
    {"op": "set_param", "name": "tmpl.W1", "value": W1},
    {"op": "set_param", "name": "tmpl.b1", "value": b1},
    {"op": "set_param", "name": "align.Pv", "pattern": {"sparse": pv, "shape": [32, GH * 32]}},
    {"op": "set_param", "name": "align.Ph", "pattern": {"sparse": ph, "shape": [LSPAN, 128 * AW]}},
    {"op": "set_param", "name": "table.eos", "pattern": {"fill": EOS_LOGIT, "shape": [12, 1]}},
    {"op": "set_param", "name": "table.padbos", "pattern": {"fill": -1e4, "shape": [12, 2]}},
]
for k in ["cell_templates", "first_on", "align", "read_cells", "reader", "to_table", "step_root", "encode_root"]:
    ops.append({"op": "put_def", "name": k, "def": defs[k]})
ops.append({"op": "delete_param", "name": "boot_logits"})

prop = {"author": "r1-iter-b",
        "rationale": ("Encoder aligns the line (first ink row -> vertical crop via a one-hot shift basis; "
                      "first ink column -> horizontal alignment), cuts 8 fixed-pitch 12px cells, scores each "
                      "with 10 hand-drawn 15x12 glyph templates plus a presence row, picks the left-bearing "
                      "offset (0..3) that best explains cell 0, and writes a 12x13 logit table (EOS where a "
                      "cell has no ink). The step bundle selects row `step` of the table."),
        "ops": ops}
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "proposal.json")
json.dump(prop, open(out, "w"))
print("wrote", out)
