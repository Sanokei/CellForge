# Prospective native CLI audit compatibility

This is an offline compatibility review before any scientific author attempt.
It does not qualify a live route, establish backend request counts, or authorize
dispatch. The original phase2 packet, contract, intake source and phase1 remain
unchanged.

## Raw validation precedes projection

The parent supplied the client-supported native event envelopes: thread start,
turn start, completed agent-message item, and turn completion with usage. A
completed reasoning item has the same exact id/type/text shape and was
prospectively allowed as a non-executing summary. Its body stays in the restricted
trace; publishable metadata contains classification and hashes only.

Validate exact top-level and nested key sets and types before extracting any
metadata. Unknown fields or events, command/tool items, malformed/blank/partial
lines, duplicate keys, gaps, truncation, errors, reconnection/fallback signals,
timeout or uncertain completion cannot become a passing audit by discarding
fields. No native sequence or channel is assumed: sequence is the wrapper's
physical line index. Exact original line terminators contribute to hashes.
Final bytes are the exact UTF-8 encoding of the completed message text, without
trimming, added newline, Markdown removal or repair.

## Five native counters, explicit normalized subset

Native usage has exactly five nonnegative non-boolean integer fields:

- input_tokens
- cached_input_tokens
- cache_write_input_tokens
- output_tokens
- reasoning_output_tokens

Preserve every field in a separate `NATIVE_USAGE` record bound to the exact
turn.completed line hash and whole raw-trace hash. Only after complete native
validation, project input_tokens, cached_input_tokens and output_tokens into
the immutable intake's existing host usage object, with credits=`unknown`.
This is an explicitly disclosed compatibility subset, not a complete native
usage record. Do not discard the other counters or invent additive relationships
between them. An account credit balance is not consumed cost.

The native schema specification is
`runs/weights-nca-author-tranche-20261008/NATIVE_EVENT_SCHEMA_SPEC.json`, SHA256
`f9027f1aedcb4aa8cd0cb0cc93e85a1d8403432bcb07c7e1ce971522120edf8b`.
It supplements the unchanged audit encoding specification
`45e069f7f5487c7320fa665dd3776eec9276b70ed1512ec58507880b714c36a1`.

## Invocation accounting and route limits

The prospectively selected unit is one CLI invocation producing at most one
accepted candidate. It does not prove one HTTP request. Request/stream retry
configuration and observed error handling do not reveal hidden backend requests
or establish equal monetary expenditure. The scientific cap remains two author
invocations and 2,976 literal positions per candidate/random arm, including
invalid/neutral/unknown attempts. No scientific invocation has occurred here.

The parent separately reported three prior server diagnostic invocations and
one later failed alias healthcheck. The healthcheck stopped on startup error
events with uncertain submission; it is counted as diagnostic overhead, zero
genome positions, not a founder or numerical result. Keep its actual receipt
and any later authorized diagnostic distinct from the two-author cap. The
prospective invocation specification preserves its earlier pending-at-freeze
wording; this paragraph records the subsequent parent-reported status without
rewriting that artifact. No raw healthcheck trace was inspected by this task.

## Verification and remaining gate

`python -m unittest tests.test_nca_native_event_envelopes -v`:
12 offline tests passed in 0.003 seconds. The fixtures cover full-object extra
field rejection, exact line/final-byte handling, all five counter preservation,
explicit normalized projection, private reasoning body exclusion, unsafe events,
and incomplete streams. These are synthetic envelope fixtures, not server traces.

The actual capture wrapper and complete raw stream must still be independently
checked against the frozen prospective specifications. Observed zero tool use
is a limited provenance claim, not proof that tool use was impossible. Generic
system/tool descriptions may remain; packet-only research input means no extra
research/repository/answer context, not absence of all system instructions.
Parent approval and reservation remain separate gates.
