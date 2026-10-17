"""Word-level tokenizer built from the trace corpus."""
import json
import os
import re

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]


def render_trace(d):
    return (f"STATE {d['state']}\nTHINK {d['think']}\n"
            f"CALL {json.dumps(d['call'], separators=(',', ':'))}\n"
            f"OBS {d['obs']}\n")


def tokenize(text):
    return TOKEN_RE.findall(text)


def build_vocab(trace_path, out_path, cap=4000, min_count=2):
    from collections import Counter
    cnt = Counter()
    n_docs = 0
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cnt.update(tokenize(render_trace(d)))
            n_docs += 1
    vocab = list(SPECIAL)
    for tok, c in cnt.most_common():
        if c < min_count or len(vocab) >= cap:
            break
        if tok not in vocab:
            vocab.append(tok)
    stoi = {t: i for i, t in enumerate(vocab)}
    with open(out_path, "w") as f:
        json.dump({"vocab": vocab, "n_docs": n_docs}, f)
    print(f"vocab {len(vocab)} from {n_docs} docs")
    return vocab, stoi


def load_vocab(path):
    with open(path) as f:
        d = json.load(f)
    vocab = d["vocab"]
    return vocab, {t: i for i, t in enumerate(vocab)}


def encode(text, stoi):
    unk = stoi["<unk>"]
    return [stoi.get(t, unk) for t in tokenize(text)]


def decode(ids, vocab):
    return " ".join(vocab[i] for i in ids)
