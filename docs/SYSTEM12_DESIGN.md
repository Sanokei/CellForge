# System 1 / System 2 composition — design only

Source inspection: September 29–30, 2026. No weights were downloaded, no model
was run, and no paid endpoint was called. Keep the current corpus. This is a
separate pretrained-system proposal; it does not replace the active no-gradient
CellForge experiment or demonstrate AGI/frontier superiority.

## Verified candidates and unresolved deployment details

Google's official [Gemma 4 card](https://huggingface.co/google/gemma-4-31B)
lists E2B, E4B, 12B, 26B-A4B and 31B. Official repositories label the license
Apache-2.0. These are pretrained models; composing them imports learned capability.
Use instruction-tuned variants for eventual assistant tests, and pin exact
revisions, tokenizer, processor and quantization before execution.

| Family | Card parameter description | Theoretical 4-bit language-weight storage, decimal GB | Official inspected weight files |
|---|---|---:|---|
| E2B | 2.3B effective; 5.1B including embeddings | 2.55 | [Base tree](https://huggingface.co/google/gemma-4-E2B/tree/main): one 10.2 GB safetensors file, revision d29ff6b |
| E4B | 4.5B effective; 8B including embeddings | 4.00 | [Official file](https://huggingface.co/google/gemma-4-E4B/blob/main/model.safetensors): one 16 GB file; listed file revision 9f9f0f2 |
| 12B | 11.95B | 5.98 | Official file tree could not be fetched; actual shard sizes remain unverified |
| 26B-A4B | 25.2B total, 3.8B active | 12.60 | [Base tree](https://huggingface.co/google/gemma-4-26B-A4B/tree/main): 49.9 + 1.7 GB shards, revision 24548b6 |
| 31B | 30.7B | 15.35 | [Base tree](https://huggingface.co/google/gemma-4-31B/tree/main): 49.8 + 12.8 GB shards, revision 5bbc2fb |

The arithmetic column is N × 0.5 bytes, a storage lower bound, not a downloadable
artifact size or RAM requirement. Embeddings, quantization metadata, vision/audio
components, activations, KV cache and runtime allocations change the total.
Sparse active parameters do not remove inactive expert weights from storage.
E2B's effective label must not be used as its stored-weight count.
The official [E2B QAT GGUF tree](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/tree/main)
lists 3.35 GB text GGUF plus 987 MB multimodal projector (4.34 GB), revision
675cff4. These concrete files differ from the theoretical lower bound.

The canonical NanoJev namespace is **C-Tianyu/NanoJev**, release
**unified-games-v1**, verified through the author's [English README](https://github.com/TianyuCodings/NanoJev/blob/main/README.en.md).
The [official project](https://github.com/TianyuCodings/NanoJev) is MIT and
describes a Qwen3-0.6B backbone with Choice, Boolean and Score heads: batched
decisions without output-token decoding. Its checkpoint was trained with
cross-entropy on Maze, Snake and two ViZDoom tasks. This is not evidence of
general language/tool routing. The public instructions target CUDA; CPU
compatibility, checkpoint byte size, release hash and independent checkpoint
license metadata remain unverified because the linked HF tree was inaccessible.
Do not substitute similarly named third-party repositories. A nominal 0.6B
backbone alone suggests 1.2 GB at 16 bits or 0.3 GB at four bits; neither number
is a verified NanoJev package or supported quantization.

Reported resources are local 10 GB RAM, no NVIDIA GPU and 3.5 GB free disk;
server two CPUs and 4 GB RAM. Multimodal E2B already exceeds local free disk.
Even text-only E2B nearly fills it, before NanoJev and working space. Concurrent
server fit and useful CPU latency are unproven. No download/run follows from
this document. First prerequisites are sufficient approved storage, supported
runtime, pinned artifacts, and measured peak RAM and latency.

## One assistant interface with two internal roles

System 1 receives the question, typed state, available tools and explicit
candidate actions. Proposed outputs are route, tool eligibility, calibrated
confidence and escalation. NanoJev is a candidate decision engine; its game
heads require transfer testing. Start with a deterministic fixed router as a
transparent control. Low confidence, unsupported formats, contradictory
observations or exhausted budgets escalate to System 2.

System 2 generates reasoning and the final language answer using Gemma.
Tools return typed observations with status, provenance and errors. System 2
must demonstrably use these observations, tested with withheld and shuffled
controls. The public interface can be cohesive while the implementation is a
modular system. Separate models connected through observations are not one
internal MoE, and compatibility of NanoJev/Gemma hidden states is not assumed.

| Existing tool | Permitted claim and boundary |
|---|---|
| OCR | Authored glyph/digit recognition with narrow font evidence; no general page QA. Alphabet probe 3/6; long document language phases timed out. |
| Arithmetic | Authored exact bounded carry kernel on 63 cases; no general mathematical reasoning. |
| Deduction | Unary controlled Horn grammar; 120/120 component cases and three actual composed harness cases. No general English semantics. |
| Grid | Development results 0/200 and associative 3/200; no ARC-AGI capability or frontier result. |

## Recipe evolution and archive

[Sakana's paper](https://arxiv.org/abs/2403.13187) and
[official repository](https://github.com/SakanaAI/evolutionary-model-merge)
support investigating recipes combining pretrained components. CMA-ES for
continuous recipe coefficients and NSGA-II for performance/cost tradeoffs are
design candidates, not implemented results here; the inspected abstract/repo
does not establish every proposed algorithm detail. Do not assume that merging
different architectures or decision heads produces a valid model. Prefer
bounded prompt, routing, threshold and tool-wiring recipes before weight merging.
CMA-ES/ES is excluded from the active no-gradient from-scratch arm under the
owner's current rule. The latest owner decision prohibits gradients of any kind; no gradient optimizer or test-time gradient adaptation is authorized on the pretrained track either.

[Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) motivates an archive
of tested agent code. Here a child contains immutable code, prompts, wiring,
model/quantization hashes, parent lineage and budget. Validate syntax, resource
limits and two disjoint development batches before eligibility. Keep failures
and valid novel children; archive selection may use development performance
plus a predeclared behavioral novelty descriptor. Novelty is not capability.
Freeze selected children before sealed final evaluation; final results cannot
choose parents, prompts, tools, thresholds or archive members.

## Comparison required before a capability claim

Compare (A) Gemma alone, (B) fixed NanoJev plus the same Gemma, and (C) evolved
composition. Use identical tasks, inputs, model access, specialist availability,
answer limits and total inference budgets, including System 1 overhead. Also
report a tool-free Gemma ablation separately. Account for search compute as
well as per-question compute; report quality, calls, escalation, latency, RAM,
invalid actions and observation-control results. Predeclare splits and scoring;
document family isolation remains unresolved and the document benchmark hold
remains active. Start with plain-text QA and typed tool tasks, then independently
evaluate broader tasks. A routing speedup alone does not establish understanding.

## Current CellForge work remains separate

The current order is exact count-reference parity, four matched arms, a bounded
mixer recipe archive, then word/cache extensions. The root's WB6 IR export has
four passing tiny probability fixtures; full-corpus parity is not established.
Count-fitted primary seeds are authorized with explicit training provenance,
excluding prior report families; preserve the pure f0b43dbe evolution line.
No gradients, gradient estimators or backpropagation are authorized there.
The first GA pilot had no random-search advantage or report improvement.
