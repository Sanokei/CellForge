"""Build proposal.json for r1-flat from hand-typed glyph templates.

All numbers here are chosen by hand. The glyph drawings below were typed by
the maker after looking at dev images (15 rows x 12 cell columns, cell col 0 =
the left edge of the 12-px monospace advance cell). No dataset access here.

Usage: python make_proposal.py <out.json> [key=value ...]
"""
import json
import sys

# '#' strong ink, '+' medium, '.' faint, ' ' background (inside glyph box)
GLYPHS = {
    "0": [
        "   .####.   ",
        "  .######.  ",
        "  ###  +##  ",
        " .##    ##. ",
        " +#+    +#+ ",
        " +#+    +## ",
        " ##.    .## ",
        " ##. ##..## ",
        " ##..##..## ",
        " +#+    +## ",
        " +#+    +#+ ",
        " .##    ##. ",
        "  ###  +##  ",
        "  .######.  ",
        "   .####.   ",
    ],
    "1": [
        "   +####    ",
        "  +#####    ",
        "  +#++##    ",
        "     .##    ",
        "     .##    ",
        "     .##    ",
        "     .##    ",
        "     .##    ",
        "     .##    ",
        "     .##    ",
        "     .##    ",
        "     .##    ",
        "     .##    ",
        "  +######## ",
        "  +######## ",
    ],
    "2": [
        "  +#####.   ",
        " +#######.  ",
        " +#+. .###  ",
        "        ##. ",
        "        ##. ",
        "        ##  ",
        "       +#+  ",
        "      .##   ",
        "     .##    ",
        "    .##     ",
        "   .##.     ",
        "  .##.      ",
        " .##.       ",
        " +########+ ",
        " +########+ ",
    ],
    "3": [
        "  .#####.   ",
        " .#######+  ",
        " .#+. .+##  ",
        "        ##. ",
        "        ##  ",
        "      .+##  ",
        "   .####+   ",
        "   .#####   ",
        "      .+##  ",
        "        ##+ ",
        "        +#+ ",
        "        ##+ ",
        " ++.  .+##. ",
        " ########+  ",
        "  +#####.   ",
    ],
    "4": [
        "      +##.  ",
        "      ###.  ",
        "     +###.  ",
        "    .#.##.  ",
        "    ## ##.  ",
        "   +#. ##.  ",
        "   #+  ##.  ",
        "  +#   ##.  ",
        " .#+   ##.  ",
        " ##    ##.  ",
        " ########## ",
        " ########## ",
        "       ##.  ",
        "       ##.  ",
        "       ##.  ",
    ],
    "5": [
        "  #######+  ",
        "  #######+  ",
        "  ##        ",
        "  ##        ",
        "  ##        ",
        "  ######.   ",
        "  #######.  ",
        "  #.  .###  ",
        "        ##. ",
        "        ##+ ",
        "        ##+ ",
        "        ##. ",
        " ++.  .###  ",
        " +#######.  ",
        "  +####+.   ",
    ],
    "6": [
        "    +###+.  ",
        "   #######  ",
        "  +##+  ++  ",
        "  ##.       ",
        " +##        ",
        " +#+        ",
        " ##.+###+   ",
        " ##+######  ",
        " ###+  .##+ ",
        " +##    +## ",
        " +#+    .## ",
        " .##    +## ",
        "  ##+  .##+ ",
        "  .#######  ",
        "   .####+   ",
    ],
    "7": [
        " #########+ ",
        " #########. ",
        "        ##  ",
        "       .#+  ",
        "       ##.  ",
        "       ##   ",
        "      +#+   ",
        "      ##    ",
        "     .##    ",
        "     +#+    ",
        "     ##     ",
        "    +#+     ",
        "    ##.     ",
        "   .##      ",
        "   +#+      ",
    ],
    "8": [
        "   +####+   ",
        "  ########  ",
        " .##+  +##. ",
        " +##    ##+ ",
        " .##    ##. ",
        "  ##+  +##  ",
        "   +#####   ",
        "  .######.  ",
        "  ##+  +##. ",
        " +#+    +#+ ",
        " ##.    .## ",
        " ##+    +## ",
        " +##+  +##+ ",
        "  ########  ",
        "   +####+   ",
    ],
    "9": [
        "   +####.   ",
        "  #######.  ",
        " +##.  +##  ",
        " ##+    ##. ",
        " ##.    +#+ ",
        " ##+    ##+ ",
        " +##.  +### ",
        "  ######+## ",
        "   +###+.## ",
        "        +#+ ",
        "        ##+ ",
        "       .##  ",
        "  ++  +##+  ",
        "  #######   ",
        "  .+###+    ",
    ],
}

CFG = {
    "ink": 1.0,      # weight for '#'
    "mid": 0.6,      # weight for '+'
    "faint": 0.2,    # weight for '.'
    "bg": -0.6,      # weight for ' ' inside the 12x15 box
    "bias_frac": 0.5,  # template bias = -bias_frac * self-match
    "eos": 0.0,      # constant EOS logit
    "row0": 6,       # first image row of the 19-row band
    "nrows": 19,
    "win_lo": 0,     # slot window in rel3 units: [12k+win_lo, 12k+win_hi]
    "win_hi": 4,
    "shift3": 3,     # seen-mask left shift
    "nslots": 12,
    "thr": 0.25,     # ink threshold for leftmost-ink detection
}

VAL = {"#": 1.0, "+": 0.45, ".": 0.15, " ": 0.0}  # approx pixel value of each class


def weights(cfg):
    wmap = {"#": cfg["ink"], "+": cfg["mid"], ".": cfg["faint"], " ": cfg["bg"]}
    rows, bias = [], []
    for jit in range(5):          # y_jitter = jit-2 -> template top at band row jit
        for d in "0123456789":
            g = GLYPHS[d]
            assert len(g) == 15 and all(len(r) == 12 for r in g), d
            w = [[0.0] * 12 for _ in range(cfg["nrows"])]
            selfm = 0.0
            for r in range(15):
                for c in range(12):
                    ch = g[r][c]
                    w[jit + r][c] = wmap[ch]
                    selfm += max(wmap[ch], 0) * VAL[ch]
            rows.append([v for row in w for v in row])
            bias.append(-cfg["bias_frac"] * selfm)
    return rows, bias


def node(i, use, inp=None, args=None):
    n = {"id": i, "use": use}
    if inp:
        n["in"] = inp
    if args:
        n["args"] = args
    return n


def build(cfg):
    W, b = weights(cfg)
    nr = cfg["nrows"]
    enc = [node("band", "op:slice", {"x": "$image"}, {"axis": 0, "start": cfg["row0"], "stop": cfg["row0"] + nr})]
    cat_in = {}
    for j in range(12):
        src = "band"
        if j:
            enc.append(node(f"sh{j}", "op:shift", {"x": "band"}, {"axis": 1, "k": -j, "value": 0.0}))
            src = f"sh{j}"
        enc.append(node(f"ex{j}", "op:expand", {"x": src}, {"axis": 0}))
        cat_in[f"x{j}"] = f"ex{j}"
    enc += [
        node("win", "op:concat", cat_in, {"axis": 0}),                 # [12,19,128]
        node("winT", "op:transpose", {"x": "win"}, {"perm": [2, 1, 0]}),  # [128,19,12]
        node("patch", "op:reshape", {"x": "winT"}, {"shape": [128, nr * 12]}),
        node("tm", "tm_cell", {"x": "patch"}),                          # [128,50]
        node("tm5", "op:reshape", {"x": "tm.y"}, {"shape": [128, 5, 10]}),
        node("S", "op:max", {"x": "tm5"}, {"axis": 1}),                 # [128,10]
        # leftmost ink column
        node("thr", "op:bias", {"x": "band"}, {"c": -cfg["thr"]}),
        node("thrr", "op:relu", {"x": "thr"}),
        node("colsum", "op:sum", {"x": "thrr"}, {"axis": 0}),          # [128]
        node("colb", "op:bias", {"x": "colsum"}, {"c": -0.05}),
        node("ink", "op:step", {"x": "colb"}),
        node("seen", "op:cummax", {"x": "ink"}, {"axis": 0}),
        node("seen3", "op:shift", {"x": "seen"}, {"axis": 0, "k": -cfg["shift3"], "value": 1.0}),
        node("cs", "op:cumsum", {"x": "seen3"}, {"axis": 0}),
        node("rel3", "op:bias", {"x": "cs"}, {"c": -1.0}),
        node("rel3e", "op:expand", {"x": "rel3"}, {"axis": 0}),        # [1,128]
        node("cent", "op:coords", None, {"n": cfg["nslots"], "start": 0.0, "step": 12.0}),
        node("cente", "op:expand", {"x": "cent"}, {"axis": 1}),        # [12,1]
        node("D", "op:sub", {"a": "rel3e", "b": "cente"}),             # [12,128]
        node("lo", "op:bias", {"x": "D"}, {"c": 0.5 - cfg["win_lo"]}),
        node("lok", "op:step", {"x": "lo"}),
        node("hin", "op:neg", {"x": "D"}),
        node("hi", "op:bias", {"x": "hin"}, {"c": cfg["win_hi"] + 0.5}),
        node("hik", "op:step", {"x": "hi"}),
        node("inw", "op:mul", {"a": "lok", "b": "hik"}),
        node("seen3e", "op:expand", {"x": "seen3"}, {"axis": 0}),
        node("inside", "op:mul", {"a": "inw", "b": "seen3e"}),
        node("m1", "op:bias", {"x": "inside"}, {"c": -1.0}),
        node("madd", "op:scale", {"x": "m1"}, {"c": 1000.0}),
        node("madde", "op:expand", {"x": "madd"}, {"axis": 2}),        # [12,128,1]
        node("Se", "op:expand", {"x": "S"}, {"axis": 0}),              # [1,128,10]
        node("SM", "op:add", {"a": "Se", "b": "madde"}),               # [12,128,10]
        node("slotsc", "op:max", {"x": "SM"}, {"axis": 1}),            # [12,10]
        node("head", "head_cell", {"x": "slotsc"}),                    # [12,13]
    ]
    encode_root = {"kind": "bundle", "inputs": ["image"], "nodes": enc, "outputs": {"slots": "head.y"}}
    step = [
        node("pos", "op:coords", None, {"n": cfg["nslots"], "start": 0.0, "step": 1.0}),
        node("dp", "op:sub", {"a": "pos", "b": "$step"}),
        node("adp", "op:abs", {"x": "dp"}),
        node("nadp", "op:neg", {"x": "adp"}),
        node("sel0", "op:bias", {"x": "nadp"}, {"c": 1.0}),
        node("sel", "op:relu", {"x": "sel0"}),                         # [12]
        node("sele", "op:expand", {"x": "sel"}, {"axis": 1}),          # [12,1]
        node("pick", "op:mul", {"a": "sele", "b": "$slots"}),          # [12,13]
        node("logits", "op:sum", {"x": "pick"}, {"axis": 0}),          # [13]
    ]
    step_root = {"kind": "bundle", "inputs": ["slots", "step", "prev", "state"], "nodes": step,
                 "outputs": {"logits": "logits", "state": "$state"}}
    headW = [[0.0] * 10 for _ in range(13)]
    for d in range(10):
        headW[3 + d][d] = 1.0
    headb = [-100.0, -100.0, cfg["eos"]] + [0.0] * 10
    ops = [
        {"op": "set_param", "name": "tm.W1", "value": W},
        {"op": "set_param", "name": "tm.b1", "value": b},
        {"op": "set_param", "name": "head.W1", "value": headW},
        {"op": "set_param", "name": "head.b1", "value": headb},
        {"op": "put_def", "name": "tm_cell", "def": {"kind": "cell", "act": "identity",
                                                      "params": {"W1": "tm.W1", "b1": "tm.b1"}}},
        {"op": "put_def", "name": "head_cell", "def": {"kind": "cell", "act": "identity",
                                                        "params": {"W1": "head.W1", "b1": "head.b1"}}},
        {"op": "put_def", "name": "encode_root", "def": encode_root},
        {"op": "put_def", "name": "step_root", "def": step_root},
        {"op": "delete_param", "name": "boot_logits"},
    ]
    return {"author": "r1-flat",
            "rationale": ("Flat recognizer: dense template matching of 10 hand-typed 15x12 digit glyphs "
                          "at 5 vertical offsets over every column (one cell on 228-dim column windows), "
                          "max over offsets; leftmost-ink detection gives 12 slot windows at a 12-px pitch; "
                          "per-slot max of template scores becomes slot logits with a constant EOS logit; "
                          "the step bundle selects slot[step] with a one-hot built from the step index."),
            "ops": ops}


if __name__ == "__main__":
    out = sys.argv[1]
    cfg = dict(CFG)
    for kv in sys.argv[2:]:
        k, v = kv.split("=")
        cfg[k] = type(CFG[k])(float(v)) if isinstance(CFG[k], float) else int(v)
    json.dump(build(cfg), open(out, "w"))
    print("wrote", out, cfg)
