"""Reference task protocols (protected surface).

Changing a protocol requires a new version id and re-evaluation of
comparison candidates; results from different protocol versions are separate
result epochs.  Final-split seeds are never stored here: the operator
supplies them on the evaluation server after an artifact is frozen.
"""
from __future__ import annotations

from ..data.render import RenderSpec

DIGITS = "0123456789"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

PROTOCOLS = {
    "digits-mono/1": {
        "charset": DIGITS,
        "max_len": 12,
        "anchor": {"spec": RenderSpec(charset=DIGITS, min_len=1, max_len=8, fonts=["DejaVuSansMono.ttf"],
                                      size_range=(20, 20), x_range=(2, 14), y_jitter=2,
                                      noise_sigma=(0.0, 0.03)),
                   "n": 1000, "seed": 1001},
        # Final splits are rendered on the evaluation server by the operator.
        "final": {
            "iid": RenderSpec(charset=DIGITS, min_len=1, max_len=8, fonts=["DejaVuSansMono.ttf"],
                              size_range=(20, 20), x_range=(2, 14), y_jitter=2, noise_sigma=(0.0, 0.03)),
            "unseen_font": RenderSpec(charset=DIGITS, min_len=1, max_len=8, fonts=["LiberationMono-Regular.ttf"],
                                      size_range=(20, 20), x_range=(2, 14), y_jitter=2, noise_sigma=(0.0, 0.03)),
            "size_shift": RenderSpec(charset=DIGITS, min_len=1, max_len=8, fonts=["DejaVuSansMono.ttf"],
                                     size_range=(18, 22), x_range=(2, 10), y_jitter=2, noise_sigma=(0.0, 0.03)),
            "length_shift": RenderSpec(charset=DIGITS, min_len=9, max_len=9, fonts=["DejaVuSansMono.ttf"],
                                       size_range=(20, 20), x_range=(2, 8), y_jitter=2, noise_sigma=(0.0, 0.03)),
        },
        "final_n": {"iid": 2000, "unseen_font": 1000, "size_shift": 1000, "length_shift": 500},
    },
    "alnum-mono/1": {
        "charset": DIGITS + UPPER,
        "max_len": 12,
        "anchor": {"spec": RenderSpec(charset=DIGITS + UPPER, min_len=1, max_len=8, fonts=["DejaVuSansMono.ttf"],
                                      size_range=(20, 20), x_range=(2, 14), y_jitter=2,
                                      noise_sigma=(0.0, 0.03)),
                   "n": 1500, "seed": 2001},
        "final": {
            "iid": RenderSpec(charset=DIGITS + UPPER, min_len=1, max_len=8, fonts=["DejaVuSansMono.ttf"],
                              size_range=(20, 20), x_range=(2, 14), y_jitter=2, noise_sigma=(0.0, 0.03)),
            "unseen_font": RenderSpec(charset=DIGITS + UPPER, min_len=1, max_len=8,
                                      fonts=["LiberationMono-Regular.ttf"], size_range=(20, 20),
                                      x_range=(2, 14), y_jitter=2, noise_sigma=(0.0, 0.03)),
        },
        "final_n": {"iid": 2000, "unseen_font": 1000},
    },
}

LM_CHARSET = " ,.abcdefghijklmnopqrstuvwxyz"

PROTOCOLS["tinylm/1"] = {
    "kind": "lm",
    "charset": LM_CHARSET,
    "ctx_len": 8,
    "gen_max_len": 200,
    # author-visible text generated on the evaluation server from the private grammar
    "anchor_file": "data/tinylm/anchor.json",
    "dev_files": {"dev_train": "data/tinylm/dev_train.json", "dev_eval": "data/tinylm/dev_eval.json"},
    "final_n_docs": 1000,
    "final_gen": {"n": 200, "temperature": 0.8},
}

REAL_CHARSET = " ,.!?'\"abcdefghijklmnopqrstuvwxyz"

PROTOCOLS["tinyreal/1"] = {
    "kind": "lm",
    "charset": REAL_CHARSET,
    "ctx_len": 8,
    "gen_max_len": 400,
    "anchor_file": "data/tinyreal/anchor.json",
    "dev_files": {"dev_train": "data/tinyreal/dev_train.json", "dev_eval": "data/tinyreal/dev_eval.json"},
    "final_n_docs": 500,
    "final_gen": {"n": 100, "temperature": 0.8},
    "max_unique_params": 20_000_000,
}


PROTOCOLS["mnist1d/1"] = {
    "kind": "clf",
    "n_classes": 10,
    "input_length": 40,
    "generator": "mnist1d (Greydanus 2020) default args; num_samples/seed/train_split overridden per split",
    "dev_train": {"seed": 1101, "num_samples": 5000, "train_split": 0.8, "part": "x"},      # 4000 signals
    "dev_eval": {"seed": 1102, "num_samples": 2500, "train_split": 0.6, "part": "x_test"},   # 1000 signals
    "anchor": {"seed": 1103, "num_samples": 2500, "train_split": 0.6, "part": "x_test"},     # 1000 signals
    "final": {"n": 2000},  # fresh secret seed per evaluation, generated on the evaluation server
    "literature_reference": {"logistic": 0.32, "mlp": 0.68, "cnn": 0.94, "gru": 0.91, "human": 0.96,
                             "source": "Greydanus & Kobak, Scaling down deep learning with MNIST-1D (ICML 2024), Table 1; "
                                       "gradient-trained, ~4000 examples; not reproduced here"},
    "max_unique_params": 2_000_000,
}
