# Paper anecdote: the blackboard (saved 2026-10-15, Sano)

> For the paper rewrite. Sano: "the blackboard architecture bit is such a great
> anecdote to add to the paper as it brings biology and neuroscience to computer
> science."

## The anecdote

In the 1970s, CMU's Hearsay-II speech-understanding system faced a problem:
no single module could understand speech, but many partial experts —
phonetics, syntax, semantics — each recognized something. The solution was the
**blackboard architecture**: independent knowledge sources all watching one
shared blackboard, each contributing *opportunistically* the moment it
recognized something relevant. No central controller told them when to speak.
Order emerged from bidding, not from instruction.

A decade later, cognitive science proposed the brain works the same way.
**Global Workspace Theory** (Baars, 1988; Dehaene) describes consciousness as
a workspace that specialist brain modules compete to write to — the winner's
content gets broadcast to all. The textbook demonstration is Sano's own:
you cannot look at written words without reading them. Vision intakes the
text *passively*; the reading just happens, unbidden, in another part of
the brain.

## Why it belongs in the CellForge paper

CellForge's router was rebuilt on exactly this principle (2026-10-15).
Capability cells no longer wait to be invoked as tools — a tiny model that
fumbles tool syntax 20–50% of the time cannot be trusted to command its own
perception. Instead, every cell *passively observes* each state and **bids**
for the router's attention ("hey, use me here"), and the winning
annotations arrive in context unasked. The language model learns *noticing*,
not tool use — the one thing transformers do natively.

The bids are the evolved glue of the original CellForge vision: cells start
with hand-written salience and then learn, under selection pressure on
action efficiency, how to win the router's attention. Independent cells,
evolving together, coordinating through competition — like cells that learned
to evolve together. The biology was the blueprint all along.
