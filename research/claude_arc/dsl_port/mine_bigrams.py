"""STATIC frequency mining of Hodel's solvers.py -- aggregate primitive unigram/bigram counts over DEVELOPMENT-split
task ids only (crc32(task)%5 != 0). Nothing in solvers.py is ever imported/executed; the source is parsed with `ast`
and only counts are kept (no per-task output is written or printed). Output: mined_bigrams_dev.json."""
import ast, json, os, sys, zlib, collections
SOLVERS = os.environ.get("ARC_DSL_SOLVERS", "C:/Users/wkeif/AppData/Local/Temp/claude/C--Users-wkeif-Documents-GitHub-CellForge/"
          "48e6a80e-d82c-46b5-a3b4-36d3aad29197/scratchpad/arc-dsl/solvers.py")

def is_dev(task):
    return zlib.crc32(task.encode()) % 5 != 0

def callee(n):
    return n.func.id if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) else None

def main():
    tree = ast.parse(open(SOLVERS).read())
    uni, bi = collections.Counter(), collections.Counter()
    n_used = 0
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("solve_")):
            continue
        task = fn.name[len("solve_"):]
        if not is_dev(task):
            continue
        n_used += 1
        prod = {}
        for st in fn.body:
            if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name):
                val = st.value
                for node in ast.walk(val):
                    c = callee(node)
                    if c:
                        uni[c] += 1
                        for a in node.args:
                            ca = callee(a)
                            if ca is None and isinstance(a, ast.Name):
                                ca = prod.get(a.id)
                            if ca:
                                bi[(ca, c)] += 1
                top = callee(val)
                prod[st.targets[0].id] = top
            elif isinstance(st, ast.Return):
                for node in ast.walk(st.value):
                    c = callee(node)
                    if c:
                        uni[c] += 1
    out = {"n_dev_tasks_mined": n_used, "unigrams": uni.most_common(80),
           "bigrams_producer_consumer": [[a, b, n] for (a, b), n in bi.most_common(120)]}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mined_bigrams_dev.json"), "w"), indent=1)
    print("dev tasks mined:", n_used)
    print("top unigrams:", uni.most_common(40))
    print("top bigrams:", [(f"{a}>{b}", n) for (a, b), n in bi.most_common(50)])
main()
