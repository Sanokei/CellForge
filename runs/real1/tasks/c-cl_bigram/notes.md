# c-cl_bigram notes
Design: cluster cl_bigram slices ctx row 7, reshapes to [36], feeds 36 micro-cells (one per output token),
each identity cell with W1 [1,36] = column of the log-bigram table (ln p(next=c | prev=p) over p), b1 = 0.
Outputs concatenated to y [36]. Pad/bos cells: W1 zeros, b1 = -20 (constant -20, also for all-zero row 7).
Params: 34 tables x 36 + 36 zeros + 2 scalars = 1262 unique numbers (39 param tensors); cells: 36; every tensor <= 36 elements.
Component test: 15/15 cases passed.
Numbers: every relative weight was typed by me (scripts/build.py) from general knowledge of English children's
stories; I did not read the dev corpus for this circuit and ran no counting code. The script only floors unlisted
entries at 0.0002 (about -8.5 nats), normalises each row to sum 1 and takes ln (formula, no data).
Rows for pad/eos as prev (never queried in practice) are uniform ln(1/34); bos row is the story-start distribution.
Concerns: weights are rough guesses, not measured; row for prev='"' and '!'/'?' quote-vs-space split is uncertain.
Word-internal apostrophe-rows and rare letters are coarse. The '<eos>' evidence is nonzero after . ! ? and quote only.
