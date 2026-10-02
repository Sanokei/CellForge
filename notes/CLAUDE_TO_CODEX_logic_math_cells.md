# Claude -> Codex: logic/math weight cells (2026-09-30)

Lane: I added ONLY `runs/baseline-logic-math-cells/` (cells.json, test_cells.py). I touched nothing in `src/`, `tests/`, or any `runs/weights-*`. Your ARC-grid / spatial cells are untouched.

What it is: 18 literal-weight threshold-net cells, written by hand by the LLM (no generator): and, or, not, nand, nor, implies, xor, xnor, mux, exactly_one3,
half_adder, full_adder, add2, mul2, cmp2, inc3, mod3_of3, is_prime3. Format `threshold-net-v1`: y = [W@x+b > 0] per layer, bit inputs.
Verified exhaustively over every input pattern, plus half_adder+full_adder chained == flat add2: `python runs/baseline-logic-math-cells/test_cells.py` (4 tests OK).

Honest status: this is a FOUNDER INVENTORY / baseline, not a ONE_STANDARD run. No STANDARD.json (deliberately, so it does not
pose as a weights-* milestone run or break test_one_standard). No mutation, no random control, no competition yet.
Useful as: (a) niche seeds for a logic/math niche, (b) crossover parents (e.g. thermometer-layer + readout-layer swaps between add2/mul2/mod3),
(c) deduction/arithmetic niche tasks. Reusable trick: layer 1 = thermometer of a weighted sum, layer 2 = alternating-sign readout.

If you want these promoted into a weights-* run (founder hashes, mutation by counterexample vs random control), tell the owner; I will not do it unprompted.

## Addendum: tool-use cells (same dir, same baseline status)
`runs/baseline-logic-math-cells/tool_cells.json` + `test_tool_cells.py` (5 tests OK). Literal tensors only; harness code only runs the calls.
- Matchers (conv kernel over ASCII one-hots, pattern lives in the weights, regex-lite via char classes): grep_todo, grep_import, grep_def, grep_digits3, glob_py.
- Router: keyword bag -> {grep_*, glob_py, script_count_lines, read_file, none}; W is 7x18 with an abstain row.
- Emitters ([T,128] sparse byte tensors decoded by argmax): script_count_lines, script_sum_numbers; scripts are executed in a subprocess and output checked.
- Router held-out 7/9 (misses: "print the file notes.txt", "what does b.py say" -> abstain). Held-out written by the same author after the weights, so not blind.
- Limits: patterns/scripts are fixed per cell (no argument slots yet); write_file/edit not covered; that is where LLM mutation would matter.

## Addendum 2: argument slots + write/edit/run (tool_cells2.json, test_tool_cells2.py, 5 tests OK)
- arg_matcher / arg_matcher_ci: ONE shared [128,128] tensor A maps a pattern byte (input, <=8 bytes, NUL=wildcard pad) to a kernel row; any pattern, no per-pattern cell.
- slots: token-feature weights pick arg1/arg2/file from the request (quoted text, "for X", token with a dot); null_bias abstains.
- router v2 (9x25): grep_arg, glob_py, read_file, write_file, edit_replace, run_file, count_lines, sum_numbers, none. Held-out 11/13 (misses: "append 'x' to notes.txt" -> grep_arg [wrong tool], "what is in data.txt" -> none).
- Episode test: grep -> write calc.py -> run (4) -> edit 2+2->6*7 -> run (42) -> edit/read notes.txt; path traversal refused; failed edit reported, not faked.
- Harness (tokenizer, sandbox, subprocess) is code; weights decide. Still baseline: no STANDARD.json, no mutation/random control. Not promoted: that needs the owner/Codex.

## Addendum 3: promoted to a real run: `runs/weights-tool-router-v1/` (owner chose build-in-place, 2026-09-30)
Own directory only; nothing of yours touched. Files: PREDECLARED.json (written before founders), tr_eval.py (evaluator with make_tasks/predict/predict_routed/make_multi_tasks/uniform_router),
tr_driver.py (infra: evaluate, charge, ledger, apply maker edits, seeded random control, crossover), inbox/ + outbox/ (my literal responses and what I was shown), provenance/, weights/, ledger.jsonl, STANDARD.json, summary.json, state.json.
- 5 niches (route, match, slot, emit, joint=cells+router), 3 hand-authored founders each, seeds 7101-7103, budget 15 charged proposals/arm/seed (llm vs seeded random-edit, same coordinate count/tensor set/range), 36 uncharged crossovers (both lineages, arm "breeding"), 118 distinct genomes.
- All 12 checks in tests/standard_checks.py PASS on this run when called directly (hash, recomputed accuracy, non-trivial, seed separation via even/odd input-hash split, provenance, near-dups, fair random vocab, ledger chain, budget, llm>random, router matters [routed 0.90-1.00 vs uniform-router 0.25], router/joint LLM+random mutations).
- Shared suite still RED, but not from this run: test_one_standard/test_standard_semantic loop over runs in sorted order and stop at the first failing one, which is runs/weights-cell-population-g1 (3 vs 2 random charges, 1 seed, bbox_mask trivial, no router). Mine is never reached by those asserts.
- Honest weaknesses (also in summary.json): I wrote the generators so I know the hidden rules; 14 of my 45 proposals were explicit no-ops on already-perfect champions (charged, neutral); 1 invalid (-5 out of range); emit kinds needing 4 edits are unreachable under the 3-coordinate cap + strict-gain gate (LLM 0.62-0.68 on two seeds); tasks are synthetic abstractions of the tool cells, not ARC or real tool use.

## Addendum 4: 20 more component cells (components.json, test_components.py, 7 tests OK), baseline status, no STANDARD.json
Families beyond OCR/reasoning, all literal tensors verified exhaustively against independent references:
counting (popcount4, majority5), codecs (decoder2to4, encoder4to2, bin3_to_unary, gray2), display (seven_segment 0-9), ordering (sort2),
memory (lookup4x2 read, latch2 write), calendar (weekday_next mod 7), graph (reach03 on 4 nodes), text/char (is_digit_ascii, is_alpha_ascii, to_upper_ascii over all 128 ASCII),
symmetry (palindrome4), and STREAMING STATE MACHINES in a new recurrent format dfa-step-v1 (state one-hot ++ input bit -> next state): dfa_mod3_stream (divisible-by-3 of a binary stream), dfa_paren_balance (depth cap 3 + error sink), dfa_detect_101 (streaming substring detector). DFAs checked on all bit strings up to length 10.
Composition test: sort2 -> seven_segment shows the larger digit. Not yet in any weights-* run; promotion needs owner+Codex.

## Addendum 5: component cells through the loop: runs/weights-component-cells-v1/ (sibling of tool-router-v1)
Niches count (popcount thermometer), decode (2-of-5 bit decoder with distractor bits), stream (101-detector DFA transition table), mem (pointer read), joint (+router). Same protocol: PREDECLARED first, 3 hand-authored founders/niche, seeds 8101-8103, 15 charged proposals/arm/seed, paired seeded random control, even/odd input-hash held-out split, crossover on both lineages.
All 12 tests/standard_checks.py checks PASS on it when called directly; distinct genomes 96 (<100 milestone) because the LLM arm reached perfect champions by round 2 and the rest of its budget was explicit no-ops (29/45 neutral). See summary.json.
Also reviewed (read-only) your runs/weights-cell-sequential-g2: 24/24 proposals rejected (12/arm), no admitted children, 6 founders/22 genomes; fails 5 checks when called directly (bbox_mask+colormap trivial, 2 founders w/o provenance, llm=random=0.000, no router).

## Addendum 6 (2026-10-01, hourly loop tick 1): composition run started, runs/weights-composed-cells-v1/
- Added components2.json + test_components2.py to baseline-logic-math-cells: 38 more tiny cells (wide gates, permutes, gray3, 3to8 decoder, priority encoder, sub/abs/sat/square/x3, 3-bit compares, number properties, ASCII hex/space/to_lower, 5 more DFAs). 5 tests OK (one readout bug in to_lower_ascii found by the exhaustive test and fixed).
- weights-composed-cells-v1: genome = wiring tensors that connect FROZEN verified literal primitives (and/or/xor/nand/nor/xnor/half_adder/full_adder from cells.json) into 2/3/4-bit adders and a 3-bit equality cell; PREDECLARED.json written before founders; perfect wirings verified at 1.0 on all four niches; this tick: 12 hand-authored imperfect founder wirings + 3 routers ingested (founder niche-normalized-ish raw accuracies 0.0-0.44). No rounds run yet; next tick: r1a.
