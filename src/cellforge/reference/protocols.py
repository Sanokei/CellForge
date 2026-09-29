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
