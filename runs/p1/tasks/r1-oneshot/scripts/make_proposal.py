"""Emit proposal.json from hand-transcribed glyph templates.

Templates below were typed by hand (by eye, from a 5-level ASCII view of a few
dev images).  This script only converts them to IR ops and computes the
per-template bias -0.5*|H|^2 from the values chosen here.  It reads no data.
"""
import json, os

MAP = {"_": 0.0, ".": 0.3, ":": 0.5, "+": 0.7, "#": 0.95}

T = {}
T["0"] = ["   .+##+.   ", "  .######.  ", "  ##+  +##  ", "  ##    ##. ", " :#:    :#: ",
          " :#.    .#+ ", " +#.    .#+ ", " +#. ## .#+ ", " +#. ## .#+ ", " +#.    .#+ ",
          " :#:    :#: ", "  ##    ##. ", "  ##+  +##  ", "  .######.  ", "   .+##+.   "]
T["1"] = ["   .+##+    ", "  +####+    ", "  :+..#+    "] + ["     .#+    "] * 10 + \
         ["  .#######+ ", "  .#######+ "]
T["2"] = ["  .+###+    ", " .#######.  ", " .+.   +##  ", "        ##. ", "        ##  ",
          "        ##  ", "       :#:  ", "      .#+   ", "      ##    ", "     ##     ",
          "    ##      ", "   ##       ", "  ##        ", " :########. ", " :########. "]
T["3"] = ["  .+###+.   ", " .#######.  ", " .+.   +##  ", "        ##  ", "        ##  ",
          "       +#+  ", "    ####:   ", "    ####+   ", "       +##  ", "        +#. ",
          "        :#: ", "        +#: ", " ::.   :##  ", " +#######:  ", "  :+###+.   "]
T["4"] = ["      :##   ", "      ###   ", "     +###   ", "    .#.##   ", "    ++ ##   ",
          "   .#  ##   ", "   #:  ##   ", "  :#   ##   ", "  #:   ##   ", " +#    ##   ",
          " ########## ", " ########## ", "       ##   ", "       ##   ", "       ##   "]
T["5"] = ["  #######:  ", "  #######:  ", "  ##        ", "  ##        ", "  ##        ",
          "  #####+    ", "  #######   ", "  +.  .###  ", "        ##. ", "        :#: ",
          "        +#: ", "        ##. ", " ::.  .+##  ", " +#######.  ", "  :+###+    "]
T["6"] = ["    :###+   ", "   ######+  ", "  +##.  .:  ", "  ##.       ", " :#+        ",
          " +#.        ", " +#..###:   ", " +#:######  ", " +##:  .##. ", " +#+    .#+ ",
          " :#:    .#+ ", " .#+    :#+ ", "  ##:  .##. ", "  .######+  ", "   .+###:   "]
T["7"] = [" +########: ", " +########. ", "        ##  ", "       .#:  ", "       +#   ",
          "       ##   ", "      :#:   ", "      ##    ", "     .#+    ", "     +#.    ",
          "     ##     ", "    .#+     ", "    ##.     ", "    ##      ", "   :#:      "]
T["8"] = ["   .###+.   ", "  +######+  ", "  ##.  .##. ", " .#+    +#. ", " .#+    +#. ",
          "  +#.  .##  ", "   :####+   ", "  .#####+.  ", "  ##:  .##  ", " :#:    :#+ ",
          " +#.    .#+ ", " +#:    :#+ ", " :##.  .##: ", "  +######+  ", "   :+###:   "]
T["9"] = ["   :###+.   ", "  #######.  ", " :##.  :##  ", " +#.    +#. ", " +#.    :#: ",
          " +#.    +#: ", " .##.  :##+ ", "  +#####:#+ ", "   :###:.#+ ", "        :#: ",
          "        +#. ", "       .##  ", "  ::  .##:  ", "  +#####+   ", "   +##+:    "]

H, W = 15, 12
rows = {}
bias = []
for d in "0123456789":
    g = T[d]
    assert len(g) == H and all(len(r) == W for r in g), d
    g = [r.replace(" ", "_") for r in g]
    rows[d] = g
    bias.append(-0.5 * sum(MAP[c] ** 2 for r in g for c in r))


def n(id_, use, inp=None, **args):
    x = {"id": id_, "use": use}
    if inp:
        x["in"] = inp
    if args:
        x["args"] = args
    return x


enc = []
# --- vertical alignment: find top ink row t, gather rows t..t+14 ---
enc += [n("rmax", "op:max", {"x": "$image"}, axis=1),            # [32]
        n("rb", "op:bias", {"x": "rmax"}, c=-0.5),
        n("rs", "op:step", {"x": "rb"}),
        n("ra", "op:cummax", {"x": "rs"}, axis=0),
        n("ra1", "op:shift", {"x": "ra"}, axis=0, k=1, value=0.0),
        n("top", "op:sub", {"a": "ra", "b": "ra1"})]             # one-hot of t
cat = {}
for j in range(H):
    enc += [n(f"ts{j}", "op:shift", {"x": "top"}, axis=0, k=j, value=0.0),
            n(f"te{j}", "op:expand", {"x": f"ts{j}"}, axis=0)]
    cat[f"x{j}"] = f"te{j}"
enc += [n("sel", "op:concat", cat, axis=0),                        # [15,32]
        n("band", "op:matmul", {"a": "sel", "b": "$image"}),       # [15,128]
        n("bandT", "op:transpose", {"x": "band"}, perm=[1, 0])]    # [128,15]
# --- sliding 12-column windows: F[c] = band[:, c:c+12] flattened row-major ---
cat = {}
for j in range(W):
    enc += [n(f"ws{j}", "op:shift", {"x": "bandT"}, axis=0, k=-j, value=0.0),
            n(f"we{j}", "op:expand", {"x": f"ws{j}"}, axis=1)]
    cat[f"x{j}"] = f"we{j}"
enc += [n("win", "op:concat", cat, axis=1),                        # [128,12,15]
        n("winT", "op:transpose", {"x": "win"}, perm=[0, 2, 1]),   # [128,15,12]
        n("feat", "op:reshape", {"x": "winT"}, shape=[128, H * W]),
        n("sq", "op:square", {"x": "feat"}),
        n("energy", "op:sum", {"x": "sq"}, axis=1, keepdims=True), # [128,1]
        n("nhe", "op:scale", {"x": "energy"}, c=-0.5),
        n("corr", "tmpl_cell", {"x": "feat"}),                     # [128,10]
        n("dig", "op:add", {"a": "corr.y", "b": "nhe"}),           # -0.5||I-H||^2
        n("scores", "op:concat", {"x0": "nhe", "x1": "dig"}, axis=1)]  # [128,11]: blank,0..9
# --- first ink column f -> base = f - 2 (centre of candidate cell starts) ---
enc += [n("cmax", "op:max", {"x": "$image"}, axis=0),             # [128]
        n("cb", "op:bias", {"x": "cmax"}, c=-0.3),
        n("cs", "op:step", {"x": "cb"}),
        n("ca", "op:cummax", {"x": "cs"}, axis=0),
        n("ca1", "op:shift", {"x": "ca"}, axis=0, k=1, value=0.0),
        n("first", "op:sub", {"a": "ca", "b": "ca1"}),
        n("pos", "op:coords", None, n=128, start=0.0, step=1.0),
        n("fp", "op:mul", {"a": "first", "b": "pos"}),
        n("f", "op:sum", {"x": "fp"}, axis=0, keepdims=True),      # [1]
        n("base", "op:bias", {"x": "f"}, c=-2.0)]

stp = [n("pitch", "param:pitch"),
       n("adv", "op:mul", {"a": "$step", "b": "pitch"}),
       n("ctr", "op:add", {"a": "$base", "b": "adv"}),             # [1]
       n("pos", "op:coords", None, n=128, start=0.0, step=1.0),
       n("dx", "op:sub", {"a": "pos", "b": "ctr"}),
       n("adx", "op:abs", {"x": "dx"}),
       n("nadx", "op:neg", {"x": "adx"}),
       n("inw", "op:bias", {"x": "nadx"}, c=1.5),
       n("mask", "op:step", {"x": "inw"}),                         # 1 on 3 columns
       n("m1", "op:bias", {"x": "mask"}, c=-1.0),
       n("pen", "op:scale", {"x": "m1"}, c=1000.0),
       n("pene", "op:expand", {"x": "pen"}, axis=1),               # [128,1]
       n("ms", "op:add", {"a": "$scores", "b": "pene"}),
       n("best", "op:max", {"x": "ms"}, axis=0),                   # [11]: eos,0..9
       n("lg", "op:pad", {"x": "best"}, axis=0, before=2, after=0, value=-1000.0)]

ops = [
    {"op": "set_param", "name": "tmpl.W", "pattern": {"stack": [
        {"flatten": {"ascii": rows[d], "map": MAP}} for d in "0123456789"]}},
    {"op": "set_param", "name": "tmpl.b", "value": [round(b, 6) for b in bias]},
    {"op": "set_param", "name": "pitch", "value": [12.0]},
    {"op": "put_def", "name": "tmpl_cell", "def": {"kind": "cell", "act": "identity",
        "params": {"W1": "tmpl.W", "b1": "tmpl.b"}}},
    {"op": "put_def", "name": "encode_root", "def": {"kind": "bundle", "inputs": ["image"],
        "nodes": enc, "outputs": {"scores": "scores", "base": "base"}}},
    {"op": "put_def", "name": "step_root", "def": {"kind": "bundle",
        "inputs": ["scores", "base", "step", "prev", "state"],
        "nodes": stp, "outputs": {"logits": "lg", "state": "$state"}}},
    {"op": "delete_param", "name": "boot_logits"},
]
prop = {"author": "r1-oneshot",
        "rationale": ("Deterministic template recognizer: align the 15-row glyph band via the first "
                      "bright row, score every 12-column window against 10 hand-transcribed "
                      "DejaVuSansMono-20 glyph templates as -0.5*||I-H||^2 (plus a blank template "
                      "= EOS), and at decode step k take the best class over the 3 window starts "
                      "centred at firstInkCol-2+12k."),
        "ops": ops}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "proposal.json")
with open(out, "w") as fh:
    json.dump(prop, fh, indent=1)
print("bias", [round(b, 2) for b in bias])
