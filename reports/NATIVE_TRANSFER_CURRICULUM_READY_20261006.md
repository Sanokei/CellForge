# Synthetic transfer curriculum frozen; two founders ready

Run: `runs/weights-native-transfer-curriculum01-20261006/`.
No authors have been dispatched or consumed. Current candidate/random charges
are zero; credit usage is unknown. The previous ARC spatial batch is unchanged.

Protocol hash:
`496773ea9e06b45fcb72a16bbb74c6f38ad74b6c244998a2a20b0770307f719d`.
Common founder payload: `AUTHORING/FOUNDER_INPUT.txt`, **4,663 UTF-8 bytes**,
including its final newline. SHA256:
`1fc5c5f9a6ca361ddc622956fe138a0ec7ed78be024238908a707c82cfd8925f`.
Both isolated tools-free Luna authors should receive that exact same text,
without other experimental context. Their external slots are `founder_a` and
`founder_b`; no slot-specific text needs transcription.

## Frozen question and panels

The independent synthetic oracle fills each occupied column with its single
marker's color and assembles a 2-by-2 identity/horizontal-flip/vertical-flip/
180-degree arrangement. Labels use direct column broadcast and list indexing;
the oracle does not call the candidate primitives or infer reference tensors.
The host deliberately supplies the spatial algorithms. This is parameter
inference and transfer within that family, not algorithm discovery or ARC
qualification. Full-column filling makes vertical flips observationally
redundant; successful behavior does not identify every orientation value.

- Six shown demonstrations: colors 2, 4 and 5; dimensions 3 through 5.
- Twelve separate search-fitness instances, with the same palette/size range.
- Twenty-four development inputs: eight unseen-color cases, eight cases forming
  four full color-bijection pairs with zero fixed, and eight larger rectangular
  grids with dimensions 6 through 9.
- Search seeds: 20261006201 and 20261006202; development seed: 20261006203.
- Exact input duplicates and recoloring/flip-equivalent layouts are excluded
  across panels, except the intentional development permutation pairs.

Only search labels were materialized. Development inputs are frozen under
`HOST_EVALUATION/DEVELOPMENT_INPUTS_SEALED.json`; their labels are **not yet
computed or scored**. Their deterministic labels are bound by the frozen oracle
source and input hash. This is a procedural seal, not encryption or an ACL.
Preflight, intake, search replay, repair preparation and winner selection never
read that file. It is opened only after the final search selection is saved.

## Finite authoring and evaluation

There are two founders and at most one replacement for each valid founder that
fails search fitness. Invalid and search-perfect founders receive no repair.
Both founders must be complete before repair preparation. Each eligible repair
gets the same six examples, its parent's exact tensors, and the first failed
search case in frozen order, including expected/actual output. No development
feedback enters authoring. No model dispatcher was added.

All attempted responses, including invalid ones, consume **167 positions** for
both the native and matched random arms: 103 free values plus64 prescribed
orientation entries. At most four completions therefore consume **668 positions
per arm**. Independent random seeds are 20261006211 through20261006214.
Partial/consumed slots refuse overwrite; stop files block work.

Both native and random winners are chosen by highest SEARCH exact count, then
ascending cell ID, before development opens. All random development scores are
reported; none can change either selected winner. The acceptance margin is
conservatively checked against every random development score, not just the
search-selected random winner.

Acceptance requires the search-selected native to score12/12 search and24/24
development, pass all four permutation checks, and exceed every matched random
development result by at least6/24. Every scored arm must have zero runtime
errors; the reference must score12/12 search and24/24 development. A failed
reference invalidates the experiment. The final24 reference check is deferred
to the one-time final opening. Nothing is automatically admitted to a population.

`SEARCH_SELECTION.json` is written before `FINAL_OPEN.json`, which is written
before reading sealed inputs or computing their labels. Either file closes
author intake. A partial final failure is preserved and cannot silently retry.
Completed final results support read-only independent replay.

## Focused verification and preserved failures

Ten focused tests passed in3.944 seconds; shell elapsed4.531 seconds. They cover
independent manual labels and separate reference fixtures; deterministic panel
structure and leakage checks without prospective labels; sealed-file exclusion;
one-time opening and exact replay; invalid charges and repeat refusal; search-only
repair eligibility; four-attempt limits; both search-only winners; the all-random
acceptance margin; tamper rejection; and stop controls.

Tests that exercise final evaluation use separate 2-row fixture panels. They
never evaluate the prospective development panel, whose dimensions are3-9.
Numerical libraries used one thread and child processes had30-second limits.

The initial structural generator test exhausted the available one-marker
layouts under duplicate exclusion. Before freezing, the unseen-color stratum
was changed to two or three markers, keeping all seeds unchanged. No development
labels or predictions informed that change. The original failing test log is
preserved alongside the passing rerun.

Actual preparation completed its freeze, then its external audit wrapper failed
an assertion that expected3 forward calls but counted6 additional recursive
leaf calls. Source and panels were left unchanged. A fresh process using saved
runtime independently confirmed3 chain calls plus6 leaf calls, all on the three
separate preparation fixtures, then passed preflight with all prediction/oracle
calls and sealed-file reads prohibited. This took0.422 seconds. Both the wrapper
failure and its explicit resolution are retained.

The required broad standard command was separately attempted with a30-second
limit. It timed out at30.016 seconds after seven contract/milestone failures
and semantic fixture errors; see its complete log below. Existing failures and
semantic temporary-directory access failures are not repaired or qualified by
this preparation. `STANDARD.json` explicitly declares a prospective, nonqualifying
study; no founder provenance or completed broad-study ledger was fabricated.

Logs:

- `reports/native-transfer-curriculum-tests-20261006.log` (initial structural failure)
- `reports/native-transfer-curriculum-tests-fix1-20261006.log` (10 tests pass)
- `reports/native-transfer-curriculum-prepare-20261006.log` (external counter assertion)
- `reports/native-transfer-curriculum-preflight-20261006.log` (independent verification)
- `reports/native-transfer-curriculum-standard-gates-20261006.log`

## Host commands after parent delivery

Use the existing bundled Python with one numerical thread and a30-second child
limit. The frozen script is:
`runs/weights-native-transfer-curriculum01-20261006/HOST_EVALUATION/runtime/native_transfer_curriculum.py`.

```text
python <frozen-script> import <run-root> --slot founder_a --response <raw-text> --source <source-json>
python <frozen-script> import <run-root> --slot founder_b --response <raw-text> --source <source-json>
python <frozen-script> verify <run-root> --slot founder_a
python <frozen-script> verify <run-root> --slot founder_b
```

Only if a valid founder fails search:

```text
python <frozen-script> prepare-repair <run-root> --slot repair_a
python <frozen-script> prepare-repair <run-root> --slot repair_b
```

Each applicable command emits a receipt and writes the exact corresponding
`AUTHORING/repair_a.txt` or `repair_b.txt` payload for the parent to dispatch.
An ineligible repair command refuses. After every required repair is consumed:

```text
python <frozen-script> finalize <run-root>
python <frozen-script> replay-final <run-root>
```

The supervisor source JSON has exactly eight fields: `author_task_name`,
`author_model`, `author_reasoning`, `route`, `context_sha256`,
`context_provenance`, `response_provenance`, `credit_usage`. Preserve actual
thread/turn identity in the source strings or a separate delivery receipt.
Use the expected packet hash only with `verbatim_frozen_payload_parent_attested`;
otherwise use `parent_transcription_exact_bytes_unknown` with a null context
hash. Credit usage remains null without a genuine meter receipt.

Changed source/tests: `scripts/native_transfer_curriculum.py` and
`tests/test_native_transfer_curriculum.py`. Earlier adapters, primitives,
completed runs, tests/floors and security settings were not changed. No server
work, model call, deployment, commit or push occurred.
