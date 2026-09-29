# r1-flat notes (flat-graph arm)

## What I built (all in encode_root / step_root; 2 cells, no extra bundles, no dispatch)
- encode_root: slice image rows 6..24 (19-row band covering y_jitter -2..2), build 12 left-shifted
  copies -> per-column 19x12 windows ([128,228]); cell `tm_cell` (identity, W1 [50,228]) scores
  10 hand-typed digit templates at 5 vertical offsets; max over offsets -> S [128,10].
- Leftmost ink: relu(band-0.25) column sums > 0.05 -> step -> cummax; shifted left 3 and cumsum
  gives rel3 = c-(L-3). Slot k window: rel3 in [12k, 12k+4] (i.e. cell start c in [L-3+12k, L+1+12k]),
  12 slots at a 12-px pitch. Additive -1000 mask outside the window, max over columns -> [12,10].
- `head_cell` (identity, W1 [13,10], b1) maps slot scores to vocab logits: pad/bos -100, EOS constant 0,
  digits = template score. Empty/out-of-image slots stay ~-1000 or negative, so EOS wins.
- step_root: one-hot over 12 slots from relu(1-|coords-step|), weighted sum of slot rows -> logits;
  state passthrough (state_dim 0).

## Key numbers and why
- Templates: 15 rows x 12 cols (one 12-px advance cell), typed by hand after viewing cropped ASCII
  dumps of clean dev glyphs (DejaVuSansMono 20px: glyph top row = 8+y_jitter, height 15, ink in cell
  cols 1..10, pitch exactly 12). Weights '#'=1, '+'=0.6, '.'=0.2, blank-in-box=-0.6.
- Template bias = -0.5 x self-match (self-match uses assumed pixel values #1/+0.45/.0.15), so a
  superset glyph (8 vs 3/0) cannot win on raw ink count.
- Ink threshold 0.25 (noise sigma <= 0.03, so ~8 sigma); window slack covers first-glyph bearing
  1 (most digits) or 2 ('1').

## Observed (dev_main, n=400)
- evals/000: exact match 1.000, CER 0.000, 0 failures, all lengths 1-8 perfect.
- evals/001 (same proposal, --trace head, 24 traced cases): min top1-top2 slot margin 9.9,
  min EOS-vs-digit margin 20.8; winning digit scores 17.5..30.6.

## Hypotheses / not verified
- Should transfer to held-out sets with the same renderer spec (integer x, y jitter +-2, sigma <= .03);
  would break with other fonts/sizes, blur, or sub-pixel positioning (templates are exact bitmaps).
- Cost: ~584M MACs per 400 images because the template cell runs on all 128 columns; could be cut
  by striding, but was not needed.

## Next
- Robustness: blurrier/ softer templates, or evaluate on noisier data if the spec widens.
