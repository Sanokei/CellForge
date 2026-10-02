# Oracle routing diagnostic and bounded weight-evolution direction

Owner intent: evolve actual numerical weights without gradient descent. Fixed-slot
shuffling describes the current comparison; it is not the intended future model.
This task implements only an offline oracle diagnostic. No live graph, router,
population experiment, server job, or co-researcher-owned file is changed.

## Primary literature checked

[Trinity, section 4.2](https://arxiv.org/html/2512.04695v3#S4.SS2)
compares its coordinator with a per-question union of correct answers across
seven models. That motivates our diagnostic bound, not permission to use report
labels in routing. Our bounds cover grid experts, not multi-turn LLM coordination.

[CycleQD, sections 3.1–3.4 and 4.1.2](https://arxiv.org/html/2410.14735v4#S3)
uses compatible experts descended from one base. Crossover combines their
weight-difference vectors; mutation rescales singular components of those
differences. Skill performance alternates between quality and archive descriptors.
Its initial Llama-3 experts were supervised fine-tuned; that initialization does
not satisfy CellForge's no-gradient primary constraint. Its reported final model
aggregates archive elites. Fixed-slot swapping is not this procedure.

## Proposed pilot for review, not launched

1. Use one small common graph topology, shared tensor shapes, activation functions,
   state ports, input encoding and output vocabulary. Freeze the numerical base,
   preprocessing and evaluator. Keep a maximum of 1,024 stored parameters.
   Reject incompatible tensor keys/shapes, nonfinite values, or undeclared graph
   changes. Current complementary slot parents have identical stored parameters;
   their weight deltas cannot seed a meaningful weight-merging experiment.
2. Create three specialist descendants through explicitly recorded numeric
   proposals: bounded rule composition, small relational reasoning, and simulated
   tool selection/argument construction. Author-written weight proposals and
   separately labeled derivative-free perturbation controls are possible;
   no backpropagation, gradient estimators, fitted adapters, imported pretrained
   weights, or runtime author calls in the primary experiment.
3. Preserve weight alignment through common ancestry and fixed neuron identities.
   Shape compatibility alone does not align hidden features. Validate port ranges,
   per-specialist behavior and activation correspondence on development probes.
   If alignment fails, stop numerical merging rather than automatically fitting a
   repair. Preserve expert-associated input contract, state update, output decoder,
   score calibration and provenance together; reset invocation state per task.
4. Compare mutation-only, weight-delta crossover-only, and crossover-plus-mutation.
   Start from identical diverse specialist descendants. Freeze three search seeds;
   allow 16 charged proposals per arm per seed. Every invalid or neutral proposal
   consumes budget. Use at most 24 development examples per evaluation, three
   skill-specific archives capped at four entries each, and identical physical
   evaluation counts. Log CPU/wall time as well as calls. Sequential single-thread
   execution, a 60-second candidate timeout, and a ten-minute total pilot cap are
   proposed; confirm feasibility before execution. No paid APIs or remote runs.
5. Declare weight-delta merge coefficients and SVD mutation ranges in advance.
   Begin with bounded convex coefficients and a delta-norm cap for safety; this is
   a deliberate constrained adaptation, not an exact reproduction of CycleQD,
   whose coefficients can be negative. Freeze treatment of bias/rank-one tensors.
   Do not substitute parameter shuffling for delta merging.
6. Freeze selection, validation and report generators with distinct seeds and exact
   overlap checks. Hold out compositions, longer sequences, renamed symbols/tool
   schemas and changed constraints, not merely random backgrounds. Use simulated
   tools with no external actions. Evaluate exact outputs, invalid actions, false
   positives and per-skill regressions. Report labels never enter selection,
   archive coordinates, calibration or candidate repair. Freeze winners before
   generating the report; oracle ceilings are scoring-only and cannot select them.
7. Promotion requires repeated gains over mutation under equal calls, retained
   specialist skills, resource compliance and transfer to frozen structural shifts.
   Harder real-domain evaluations come later under separately declared access and
   budgets. Synthetic improvement alone supports a narrow skill result, not AGI
   or real-domain transfer. Stop if specialists lack complementary capabilities,
   deltas are effectively zero, alignment fails, or gains disappear off-generator.

## Diagnostic boundaries

The implemented probe accepts only demonstration pairs and query input. It
executes every expert with its own demonstration-adapted association state,
then scores those frozen outputs against targets outside inference. Ordinary
top-two outputs are independently checked against GridPredictor on every query.
Both parents, four predeclared crossover children and four mutation children
receive exactly the same tasks; no best offspring is selected from report scores.
Each oracle has more candidate outputs than the real two-attempt router, so it is
a diagnostic ceiling, not an equal-compute deployable system. A parent-union
coverage gap can reflect absent inherited modules; it does not prove weight damage.
Dependency commitments reject results if relevant source changes during a run.

No fixed-slot routing patch is planned. Use the measured decomposition to decide
whether a future compatible weight model needs specialists, preservation of
capabilities, or a separately trained/calibrated router on development data only.
