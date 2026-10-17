"""ALPHA cell proposal (gen3): act.target{which}
Coordinate-grounding action cell. Generic infrastructure, no per-game logic.

Problem: the model emits click coordinates as text (act.click_at{x,y}), but on
novel states its clicks are ungrounded (audit-2: transfer clicks collapse to
memorized positions; 50% malformed CALLs on novel states). Copying bbox
arithmetic ((x0+x1)/2) through a 1.4M-param LM is unreliable.

Mechanism: the model names an object by its rank in the STATE text's OBJ list
(a small int, vocab-safe); the cell resolves LIVE frame coordinates at call
time and clicks the bbox center. Grounding happens in code, not in the LM's
arithmetic. Object order matches perceive.describe (area-descending), so the
STATE text the model already sees and this cell agree on indices.

Name "act.target" chosen for vocab-emittability: encodes as
['act', '.', 'target'] with no <unk> (verified against vocab.json, 2209).
Args: which (int, default 0), clamped to [0, len(objs)-1].
"""


def c_target(st, args):
    f = st.frame()
    objs = _objs(f)  # area-descending, same order as perceive.describe
    if not objs:
        return "target: no objects visible; no click issued."
    try:
        which = int(args.get("which", 0))
    except Exception:
        return "target invalid: which must be an int"
    which = max(0, min(which, len(objs) - 1))
    o = objs[which]
    cx = (o["x0"] + o["x1"]) // 2
    cy = (o["y0"] + o["y1"]) // 2
    note = st._step(GameAction.ACTION6, data={"x": cx, "y": cy})
    base = _after(st, note + f"TARGET obj{which} c{o['c']}@({cx},{cy})", 6)
    return base


# REGISTRY addition (for coordinator merge into lm/cells.py):
#   "act.target": c_target,
# CELL_DOCS addition:
#   act.target{which} -> click center of Nth perceived object (area-descending,
#                        same order as STATE OBJ list; which=0 is largest)
