# Cleanup notes for paper/cellforge.tex (diagnosis, 30 September 2026)

Cleaned version: `paper/cellforge_clean.tex` (original untouched). Build: `latexmk -pdf` on server
`cellforge-1` (TeX Live 2023, no build script in paper/; `build.log` is from a stale 5-page build).
Number check: `python paper/check_numbers.py`.

## Structural problems found
1. Sections appended chronologically. Lines 1-727 are a coherent 4-experiment paper; lines 735-1311 are a
   running lab log (CPU-only graph breeding, ~30 subsections/paragraphs) added after the bibliography and
   Limitations/Roadmap, so Limitations and Roadmap sit in the middle and the bibliography is mid-document.
2. `\section{Experiment 4}` became a catch-all (benchmarks, tools, documents, evolution, count models,
   comparator plan, language-core options) with ~25 untitled `\paragraph`s.
3. A second, unrelated stream ("Design-only pretrained composition") was followed by ~20 more ARC
   `\paragraph`s (still inside that section heading), so ARC results are filed under "Design-only".
4. Title, abstract, contributions and Scope still describe only the digit OCR round ("first results on
   gradient-free construction of an OCR model"); they omit LM, MNIST-1D, ARC, evolution results.
5. Duplicated/overlapping content: frontier MC probes reported twice (first sweep vs `82ebe5e5` probe);
   MoE failed-mutation result filed under the MNIST-1D section; owner-policy statements (no gradients, Hedge
   exclusion, count-seed authorization) repeated ~8 times; "No gradients, public evaluation, audit or final
   access occurred" repeated ~25 times; "Compilation remains unverified" repeated ~20 times; DEV200
   exposure caveat repeated; count-reference numbers told in three places; maker accounting in two places.
6. Stale claims: "PDF compilation remains unavailable/unverified" (server has pdflatex; clean version builds,
   16 pages); "pilot being prepared, not reported" for evolution (completed later in same text); "run is
   pending" for corrected population runner (completed in `grid-population-matched-v1`); "Round B not run"
   and "visual retriever not run" kept as limitations (still true, kept); "owner now requests..." framing
   (process narration) replaced by results.
7. Placeholder: author line `[name to be supplied]` (kept, owner must fill).
8. Unreferenced tables/figures: original had all 3 tables referenced; none unreferenced. Section labels
   `sec:exp2`, `sec:mnist1d`, `sec:frontier` existed; clean adds labels (several unreferenced, harmless).
9. Typographic corruption in appended text: missing spaces around numbers (`over152.68`, `about34`,
   `loss1.3816 on165`, `89.416,139.494,118.018` -> tokenizes as `89.416139`), `{,}` thousands separators
   dropped in places. Fixed with spaces; values unchanged.
10. Claims stronger than evidence: abstract "establish that LLM authors can construct working visual
    recognizers" is fine for digits but was the only headline; clean abstract adds the negative results.
    "Prize-winning odds assessed qualitatively as very low" kept as stated. "Wider prior-art review required"
    kept. Evolved-NCA and co-researcher numbers flagged as not reproduced.
11. Missing limitations: none of the later results (chance MC scores, 3/200 ARC, ties) fed back into
    Limitations; clean Discussion adds: exposed public evaluation, single small runs, no private-set score,
    not an LLM, no AGI claim, no final evaluation outside digits/tinylm/MNIST-1D.
12. Citations: only 5 bibitems, while text cites NCA arXiv:2506.15746 and arXiv:2609.27242 by URL/id inline.
    Clean adds bibitems ONLY for ids recorded in notes/CLAUDE_RESEARCH.md: 2506.15746 (Xu & Miikkulainen),
    2301.05062 (Tracr, DeepMind), 2609.27242 (title only; authors not recorded). No authors/years invented
    beyond what the ids and notes imply (Tracr and NCA years are implied by arXiv id / notes).

## Conflicts between passages / sources (flagged, not silently resolved)
- **arXiv 2609.27242**: tex says it "could not be retrieved from primary arXiv endpoints" and treats its
  claims as unverified; notes/CLAUDE_RESEARCH.md says "fetched" with 2/60 vs 6/60 numbers. Clean keeps the
  tex's stance (cited, numbers not quoted or compared) - owner should decide.
- **MMLongBench-Doc annotations**: tex says pinned source has 1,082 annotated questions, README says 1,091.
  Kept 1,082 (pinned receipt), flagged in Reproducibility.
- **Co-researcher library counts**: "53/1000 training solves" (first report) vs "80/1000 (v2) -> 95/1000 (v3)";
  tex also says "v2 filenames do not pin the receipt-generating run". Labelling of which run is 53 is
  ambiguous; kept all numbers with the version labels as given.
- **Founder loss**: 2.5113 (24 docs, 12,894 bytes) vs 2.042730829833363 (334 bytes) vs 4.0663511910499714
  (190 bytes) vs 2.4798989/2.480344 (387-byte probe) vs 2.4748-2.4826 (165-doc dev set, real-text MoE).
  Different scopes, not contradictions; clean adds Table `counts` and scope labels.
- **Scratch NCA "held-out success in one of three configurations"** (414 params, 3 steps, synthetic) vs
  later "0/40 train and test" on real ARC: different workloads, consistent.
- **"Native solves" range**: 3/200 for portfolios, 4/200 for the object child (exposed development,
  post-hoc attribution), 1/200 for several single genes; preserved per row.
- The notes say `ES 0/40 train` for 414-param run after CORRECTION; tex matches the corrected number.
- Original Limitations still say MNIST-1D crossover has "one independently tested child" while the table has
  two-parent and three-parent children (three-parent child was not independent of the two-parent parents);
  kept.

## Items dropped as process narration (no measured number lost except as noted by the check script)
"Compilation remains unverified" (x20), "manuscript source updated, PDF uncompiled" (x6), "owner now
requests / subsequently", pending-prototype lists without results, and duplicate policy statements.
Result numbers were kept; see `python paper/check_numbers.py` for the list of numeric tokens from the original
absent from the clean version.
