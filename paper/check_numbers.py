"""Compare numeric tokens of paper/cellforge_clean.tex with the original draft and notes,
and run static structure checks (environments, braces, refs, cites)."""
import pathlib
import re

root = pathlib.Path(__file__).resolve().parent.parent
BS = "\\"


def toks(text):
    text = text.replace("{,}", "").replace(BS + "%", "")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    out = set()
    for m in re.finditer(r"\d+(?:\.\d+)?", text):
        t = m.group(0)
        out.add(t)
        if "." in t:
            out.add(t.rstrip("0").rstrip("."))
    return out


clean = (root / "paper/cellforge_clean.tex").read_text(encoding="utf-8")
orig_text = (root / "paper/cellforge.tex").read_text(encoding="utf-8")
orig = toks(orig_text)
allsrc = set(orig)
for s in ["notes/CLAUDE_RESEARCH.md", "README.md", "docs/ARC_PRIZE_2026_PLAN.md",
          "notes/PAPER_DRAFT_CLAUDE.md", "notes/PAPER_OUTLINE_CLAUDE.md"]:
    p = root / s
    if p.exists():
        allsrc |= toks(p.read_text(encoding="utf-8"))
c = toks(clean)
print("Numbers in clean but NOT in original tex:", sorted(c - orig, key=float))
print("Numbers in clean but NOT in original+notes:", sorted(c - allsrc, key=float))
print("Numbers in original but absent from clean:", sorted(orig - c, key=float))

begin = re.findall(re.escape(BS + "begin{") + r"(\w+)\}", clean)
end = re.findall(re.escape(BS + "end{") + r"(\w+)\}", clean)
print("environments balanced:", sorted(begin) == sorted(end), "| brace diff:", clean.count("{") - clean.count("}"))
labels = set(re.findall(re.escape(BS + "label{") + r"([^}]+)\}", clean))
refs = set(re.findall(re.escape(BS + "ref{") + r"([^}]+)\}", clean))
print("undefined refs:", refs - labels, "| unreferenced labels:", labels - refs)
bib = set(re.findall(re.escape(BS + "bibitem{") + r"([^}]+)\}", clean))
cites = {x.strip() for g in re.findall(re.escape(BS + "cite{") + r"([^}]+)\}", clean) for x in g.split(",")}
print("undefined cites:", cites - bib, "| uncited bib items:", bib - cites)
