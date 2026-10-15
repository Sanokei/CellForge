"""Lossless UTF-8 byte tokenization for a common language-model interface."""
from .decoder import BOS, EOS, PAD

BYTE_CODEC = "utf8-bytes/1"
BYTE_VOCAB = [PAD, BOS, EOS] + [f"<byte:{i:02x}>" for i in range(256)]


def encode_content(vocab, text, codec=None):
    if codec == BYTE_CODEC:
        if vocab != BYTE_VOCAB:
            raise ValueError("utf8-bytes/1 requires the canonical 259-token vocabulary")
        return [byte + 3 for byte in text.encode("utf-8")]
    if codec is not None:
        raise ValueError(f"unsupported text codec: {codec}")
    idx = {c: i for i, c in enumerate(vocab)}
    return [idx[c] for c in text]


def decode_content(vocab, tokens, codec=None):
    if codec == BYTE_CODEC:
        if vocab != BYTE_VOCAB or any(not 3 <= token < 259 for token in tokens):
            raise ValueError("invalid UTF-8 byte token sequence")
        return bytes(token - 3 for token in tokens).decode("utf-8")
    if codec is not None:
        raise ValueError(f"unsupported text codec: {codec}")
    return "".join(vocab[token] for token in tokens)
