# Fixed-weight post-selection stress

This is a declared stress pass after selection of the analytical reference
weights. It is not an untouched final test. No author, optimizer, parameter
repair, scenario selection from results or admission is performed. The exact
successful 293 coefficients and parent source are copied and hashed unchanged.

Before evaluating the model, `stress_check.py prepare` freezes nine original
deterministic 32-event scenarios, all with four actors and four objects. There
is no random seed search. Two categorical base schedules cover all 16 observed
actor/object pairs followed by repeated unseen moves, and a separate mixture
of distractors, unknown, no observers, partial observers and all observers.
Two consistent actor/location permutations per base produce six categorical
scenarios. Unknown remains index 4. Twelve index pairs fall outside the earlier
2x2 search domain; they are not claimed unseen in the prior 4x4 development
panel. The original panels remain unchanged.

Three distributed-state scenarios probe repeated unseen preservation, observed
distractors that should leave other objects' distributions unchanged, and near-
tie probability mixtures. These are deliberately separated from the categorical
cohort. The reference preserves the actual original probability mass; it does
not round a distribution to its largest component. The neural model always
receives its own previous output, with no reference repair.

Every one of 16 slots is scored after every event: 512 predictions/scenario,
4608 total. Whole-scenario top-1 and full-distribution-tolerance passes are
separate. Fixed max-absolute probability tolerance is 0.05; a 1e-12 identity
probe documents the nonidentity limit separately. Permutation equivariance
compares the complete predicted distributions at tolerance 1e-12. All slot
failures, per-event counts and first mixed-distribution failures are retained.

Evaluation uses one stdlib process, no numerical worker pool, and a 10 ms yield
after each event. An internal 105-second deadline and external 120-second
subprocess timeout cap the pass. Run the prepared hash once; repeated execution
cannot overwrite results. No broad suite or new scientific qualification is
claimed. Any distribution-preservation failure is evidence about these fixed
weights and must be reported, not optimized away in this pass.
