# Two founders pass the frozen synthetic transfer criterion

Both valid native founders scored12/12 search and24/24 development. Their
matched random controls scored0/12 and0/24. The hand-coded reference scored
12/12 and24/24; zero tensors scored0/12 and0/24. Every evaluated arm had zero
runtime errors. The selected native passed all four permutation-pair checks.
The frozen synthetic transfer criterion passed; no ARC qualification, algorithm
invention, chatbot capability, or population admission is claimed.

Run: `runs/weights-native-transfer-curriculum01-20261006/`.
Preparation, independent oracle, control vocabulary, seeds and criteria remain
exactly as frozen in `reports/NATIVE_TRANSFER_CURRICULUM_READY_20261006.md`.
No source change or additional author call occurred during intake/evaluation.

## Exact intake, search and stopping rule

The parent supplied both completed tools-free gpt-6-luna responses and attested
that each author received only the exact shared founder text, without inherited
context. Reasoning effort and credit consumption were not supplied and remain
unknown. The payload hash is parent-attested, not a direct capture of platform
prompt bytes. Exact supplied response strings are preserved as UTF-8, without
a trailing newline, under `SUPERVISED_TEXT_SUBMISSIONS/founder_a` and `founder_b`.
The consumed copies are byte-identical. No response was repaired or normalized.

| Slot | Thread | Turn | Raw bytes |
|---|---|---|---:|
| founder_a | `01a111c3-405a-708e-ad07-fc76dc622032` | `01a111c3-49ae-7606-af2e-530d81823cf2` | 810 |
| founder_b | `01a111c3-84c4-71c6-9faf-50d85757d022` | `01a111c3-90f4-7455-bfc5-eaac74d73847` | 848 |

Response SHA256s:

- A: `fb377c0b20ccd3580b368f1a779109500225809685a342046eaaf60f514e8c37`
- B: `ec2069f65c5aa9591abdd866bd3608f817f978f8b4697571a16763a57b5ede95`

Founder B's U+2019 apostrophe is retained. Thread/turn identity, source receipt,
parent attestation and raw hash are recorded alongside each submission.

Both founders passed strict validation and all12 search cases. Thus neither
was eligible for a replacement, and authoring ended after **two completions**.
No repair packet was produced. Search-only intake and independent search replay
ran with audit guards that prohibited sealed-file reads and all oracle calls.
Development remained unopened through completion of both search replays.

Each attempt and random control consumed all167 literal positions, including
inactive slots: **334 native and334 random positions** for this synthetic batch.
This is separate from the completed native ARC sequence's1206 per arm. Combining
only those two accounting categories gives1540 per arm; belief trials remain
separate. Position accounting is not a credit meter. Credits remain unknown.

## Selection, one-time opening and results

The frozen search-only tie rule selected native founder A, cell
`778d0678b19cd5d5723db1e988d22dca35182719e210c2b5a0afb4e45064aa34`, and random
founder B's control, cell
`18612d81401da38111552e7ca5df1f0b13ddcc3f9f1b8fedd97c2028e85560ab`.
Neither choice used development scores. `SEARCH_SELECTION.json` was saved
before `FINAL_OPEN.json`, both before the first sealed-file read or label
materialization. A runtime audit guard enforced this ordering.

| Arm | Search | Development | Runtime errors |
|---|---:|---:|---:|
| Founder A, selected native | 12/12 | 24/24 | 0 |
| Founder B | 12/12 | 24/24 | 0 |
| Matched random A | 0/12 | 0/24 | 0 |
| Matched random B, selected random | 0/12 | 0/24 | 0 |
| Independent hand-coded reference | 12/12 | 24/24 | 0 |
| Zero tensors | 0/12 | 0/24 | 0 |

The selected native scored8/8 in each development stratum: unseen colors,
permutation pairs, and larger rectangles. All four explicit permutation checks
passed. The native margin over every random control is24/24, exceeding the
prespecified6/24 threshold. The final24 reference check passed at the authorized
opening; no prospective development labels had been computed during preparation.

Finalization took0.672 seconds; a fresh saved-runtime process independently
replayed the full selection, scores, controls, acceptance decision and result
hash in0.563 seconds. Each process had a30-second limit and one numerical
thread. Replay was read-only; it did not reopen authoring or generate feedback.

Protocol hash:
`496773ea9e06b45fcb72a16bbb74c6f38ad74b6c244998a2a20b0770307f719d`.
Final result hash:
`01010af814c8e92462c61e399288b42d2e27c83749af1ec53d58f9f59719d8a8`.

## What this establishes and does not establish

Both authors chose all nonbackground source colors, vertical rays, source-color
painting, two tile rows, and identity/left-right-reflected tile columns. This
transferred to the prespecified new palettes, color permutations and larger
rectangles within the deliberately compatible synthetic rule family.

The two tensor genomes have different hashes but **one functional behavior**.
They differ in16 orientation maxima,9 versus1; source gates, directions,
stop/paint decisions, tile counts and every orientation argmax are identical.
The frozen primitives depend only on those decisions, so these are behaviorally
equivalent predictors, not two independently acquired skills. Source-derived
evidence is in `HOST_EVALUATION/BEHAVIORAL_COMPARISON.json`.

The host already implements ray propagation and tiling; the oracle and template
family were deliberately chosen to agree. No mutation was needed, no crossover
was attempted, and no advantage over the hand-coded reference was demonstrated.
Full-column filling makes vertical reflection redundant. These results do not
show general spatial reasoning, a successful evolution campaign, ARC accuracy,
or a usable chatbot. All prior nulls and failed broader milestones remain.

The adapter's10 focused integrity tests passed before dispatch. The most recent
required repository-wide command remains unsuccessful:

```text
python -m unittest tests.test_one_standard tests.test_standard_semantic -v
TIMEOUT_SECONDS: 30
STATUS: TIMEOUT
ELAPSED_SECONDS: 30.016
```

It recorded seven contract/milestone failures and semantic fixture errors.
Complete output: `reports/native-transfer-curriculum-standard-gates-20261006.log`.
The existing blocker/missing-evidence classification is unchanged. No tests,
floors or security settings were altered to obtain the synthetic result.

## Deliverables

- `HOST_EVALUATION/founder_a/` and `founder_b/`: raw consumed response, source,
  preparse charged attempt and search result.
- `HOST_EVALUATION/SEARCH_SELECTION.json`, `FINAL_OPEN.json`, `FINAL_RESULT.json`:
  selected winners, one-time opening and all development predictions/scores.
- `HOST_EVALUATION/BEHAVIORAL_COMPARISON.json` and `TERMINAL_STATUS.json`:
  functional-duplicate disclosure and terminal state.
- `reports/native-transfer-founder-a-search-20261006.log`
- `reports/native-transfer-founders-search-verified-20261006.log`
- `reports/native-transfer-final-replay-20261006.log`

No more author calls are eligible in this batch. No server work, deployment,
publication, commit, push, gradient or download occurred. The next research
milestone needs a separate frozen protocol; this batch's development panel is
now spent and must not serve as untouched evaluation in later tuning.
