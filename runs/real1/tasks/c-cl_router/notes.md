# cl_router notes
Design: one cluster `cl_router`, 7 cells (all identity, W1 [1,36]): letter mask (applied to whole ctx -> [8,1] letter flags),
bos/space/punct(,.!?)/dquote/apostrophe class cells (weight 3 on row 7), and a "none" cell (1 - sum of known classes) for the default.
Word length: l5*l6*l7 (letter flags of rows 5,6,7) = long (>=3); short = 3*l7 - long. An apostrophe/boundary is a non-letter so it breaks the run.
y = concat[bos, space+none, short, long, punct, quote, apos] -> reshape [7]. Default (eos/pad/empty row 7) gives [0,1,0,...].
Params: 7 W1 (36 each) + 7 b1 = 14 unique params tensors (259 numbers). Cells defs: 7. Cell instances: 7.
Cases: see final message (all component-test cases).
Contract concerns: none. No corpus/data used; all values typed by me.
