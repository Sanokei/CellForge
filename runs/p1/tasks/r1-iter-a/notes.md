# r1-iter-a: hand-built template reader for digits-mono/1

## What I built (all values hand-chosen; scripts/build.py only serialises them)
Encoder (`encode_root`):
- `vert_align` bundle: row-ink profile -> `vband_cell` (5 band detectors, glyph top in
  rows 6..10; +1 inside a 15-row band, -2 on the row just above/below) -> softmax(temp 0.5)
  -> weighted sum of 5 row-slices = 15x128 image aligned so glyph top is row 0.
  (Glyph height 15 rows, top = 8 + y_jitter, seen in dev PNG dumps.)
- `col_windows` bundle: 12 column shifts -> [128, 15*12] window per start column.
- `glyph_cell`: 10 hand-transcribed 15x12 ascii templates (scripts/templates.py:
  X=+1 ink, .=-1 background, o=0 antialiased edge). Score = T.(2I-1)/N_c, i.e. normalised
  agree-minus-disagree, so subset glyphs (3 vs 8, 1 vs 7) do not tie.
- `left_edge` bundle: column ink sum, threshold 1.5, cummax -> x0 = first ink column.
Step (`step_root`, no state):
- `cell_locator`: t = x0 - 2 + 12.04*step (DejaVu Sans Mono advance 0.602*20 px).
- `char_head`: max of template scores over start columns |col - t| <= 3 (tolerates
  per-digit left bearing, e.g. '1' starts ~2px later).
- `eos_head`: ink in columns t+3..t+8; eos logit = 2*(5 - ink). Past the last glyph the
  region is empty (last glyph ends <= x+12n), so EOS fires.
- `<pad>/<bos>` fixed at -50 via `special_logits`.

## Observed (dev_main, 400 cases)
- evals/000: exact match 1.000, CER 0.000, length errors 0, all lengths 1-8 perfect.
- evals/001 (same proposal, --trace lg on 24 traced cases): min top1-top2 logit margin on
  char steps 0.16 (normalised units; median 0.20-0.42); on EOS steps >= 4.86.
- Cost: ~92M MACs/400 cases, 1982 params, 12 defs, nesting depth 3.

## Hypotheses / risks (not measured)
- Relies on fixed font/size (20px) and integer-ish pitch; a different size or font would
  break templates and pitch. y range is limited to tops 6..10.
- Heavy noise or blur would reduce margins (0.16 min); x0 threshold 1.5 could misfire on
  strong noise.
- Next: add scale/pitch estimation from the ink extent, vertical-tolerant templates, and
  a dispatch-free cheaper window (only compute scores near t) to cut MACs.
Dev-evals used: 2 of 25.
