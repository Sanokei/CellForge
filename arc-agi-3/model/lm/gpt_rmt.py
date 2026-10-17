"""RMT memory tokens on the tiny nanoGPT router (Bulatov et al. RMT lineage).

Architecture: 8 learnable memory tokens persist across decisions within an
episode, read AND rewritten each step by the SAME transformer weights
(decoder-only, no separate encoder).

POSITION SCHEME (documented choice): memory tokens occupy absolute positions
0..7; text occupies 8..8+T. The position embedding table is EXTENDED from
block_size (192) to block_size + n_mem (200). Warm-start copies the
pretrained wpe rows 0..191 into rows 8..199 (so text keeps the exact
positional semantics it learned during pretraining; relative text-text
distances are unchanged), and rows 0..7 (memory positions) are fresh
small-random init. Memory tokens therefore get their own positional
signature while the text path is warm-start compatible.

Forward:
    forward(idx, mem=None, targets=None)
    - mem is (B, n_mem, C) from the previous decision, or None -> mem_init.
    - embeddings: x = [mem + wpe[0:8], wte(idx) + wpe[8:8+T]]
    - run blocks, ln_f.
    - mem_out = hidden states at the 8 memory positions (after ln_f).
    - logits = lm_head(hidden at the text positions only).
    - loss = CE on text positions only.
    Returns (logits, loss, mem_out).

BPTT note: mem_out carries a live autograd graph back through mem, so
training loops can accumulate gradients across steps of an episode WITHOUT
detaching mem -- the gradient path through the memory WRITE is exactly what
teaches the model to write useful memory. The memory ZERO-load-bearing test
(zero mem at a mid-episode decision) is the anti-cheat standard for whether
the model actually depends on memory vs. ignoring it.

Determinism: no dropout in the fixed config; all randomness (token sampling)
uses the caller's torch RNG state. RMTGPT itself is deterministic given
(idx, mem).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt import Block


class RMTGPT(nn.Module):
    def __init__(self, vocab_size, n_layer=4, n_head=4, n_embd=128,
                 block_size=192, dropout=0.0, n_mem=8):
        super().__init__()
        self.block_size = block_size          # max TEXT tokens per step
        self.n_mem = n_mem
        self.wte = nn.Embedding(vocab_size, n_embd)
        # extended position table: mem positions 0..n_mem-1, text 8..8+T
        self.wpe = nn.Embedding(block_size + n_mem, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout)
                                      for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # learnable initial memory state (B, n_mem, C) template
        self.mem_init = nn.Parameter(torch.zeros(n_mem, n_embd))
        self.apply(self._init)
        # mem_init gets its own smaller init (overwritten by _init's normal)
        nn.init.normal_(self.mem_init, std=0.01)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def warm_start(self, gpt_state, verbose=True):
        """Load transformer weights from a plain GPT checkpoint (router_v3
        lineage). The extended wpe rows [n_mem : n_mem+block_size] are filled
        from the pretrained rows [0 : block_size]; the memory rows
        [0 : n_mem] and mem_init stay as freshly initialized. Extra keys in
        the model (none) are ignored; missing keys (mem_init if someone
        saved only GPT keys) are kept as init.

        Returns the list of keys skipped because of shape mismatch.
        """
        sd = self.state_dict()
        skipped = []
        for k, v in gpt_state.items():
            if k == "wpe.weight":
                want = sd[k]
                got = v
                if got.shape[0] == self.block_size and \
                        want.shape[0] == self.block_size + self.n_mem:
                    want[self.n_mem:self.n_mem + self.block_size] = \
                        got[:self.block_size]
                    if verbose:
                        print(f"warm_start: shifted pretrained wpe rows "
                              f"0..{self.block_size} -> "
                              f"{self.n_mem}..{self.n_mem + self.block_size}; "
                              f"rows 0..{self.n_mem} stay fresh init")
                    continue
                skipped.append(k)
                continue
            if k in sd and sd[k].shape == v.shape:
                sd[k].copy_(v)
            else:
                skipped.append(k)
        if verbose and skipped:
            print(f"warm_start: skipped {skipped} (kept fresh init)")
        return skipped

    def forward(self, idx, mem=None, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, \
            f"text len {T} > block_size {self.block_size}"
        device = idx.device
        if mem is None:
            mem = self.mem_init.unsqueeze(0).expand(B, -1, -1)
        assert mem.shape == (B, self.n_mem, self.wte.embedding_dim), \
            f"mem shape {tuple(mem.shape)} != (B, {self.n_mem}, C)"

        pos = torch.arange(self.n_mem + T, device=device)
        mpos = self.wpe(pos[:self.n_mem]).unsqueeze(0)      # (1, 8, C)
        tpos = self.wpe(pos[self.n_mem:]).unsqueeze(0)      # (1, T, C)
        x = torch.cat([mem + mpos, self.wte(idx) + tpos], dim=1)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        mem_out = x[:, :self.n_mem, :]                      # (B, 8, C)
        h_text = x[:, self.n_mem:, :]                       # (B, T, C)
        logits = self.lm_head(h_text)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss, mem_out


def rmt_generate(model, mem, idx, max_new_tokens, temperature=0.7, top_k=40):
    """Constrained-decoder-friendly generation: memory is held FIXED for the
    whole decision (it is read, not rewritten, while generating); the caller
    runs one final forward on the full segment to obtain mem_out for the next
    decision. Returns (full_idx, mem_out_of_final_segment).

    idx: (1, T) prompt token ids, T <= block_size. Deterministic iff the
    caller's torch RNG state is deterministic (temperature 0 -> greedy).
    """
    assert idx.shape[0] == 1, "eval path is batch-1"
    assert idx.shape[1] <= model.block_size
    block = model.block_size
    cur = idx
    with torch.no_grad():
        for _ in range(max_new_tokens):
            seg = cur[:, -block:] if cur.shape[1] > block else cur
            logits, _, _ = model(seg, mem)
            lg = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(lg, min(top_k, lg.shape[-1]))
                lg[lg < v[:, [-1]]] = -float("inf")
            if temperature <= 0:
                nxt = lg.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(lg, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            cur = torch.cat([cur, nxt], dim=1)
        final_seg = cur[:, -block:] if cur.shape[1] > block else cur
        _, _, mem_out = model(final_seg, mem)
    return cur, mem_out
