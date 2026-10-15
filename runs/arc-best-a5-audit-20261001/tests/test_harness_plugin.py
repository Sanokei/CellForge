"""G3: harness plugin sandbox and negative tests."""
import numpy as np

from cellforge.harness.sandbox import PluginError, declared_trace_nodes, run_plugin, static_check


def test_static_check_blocks_dangerous_code():
    assert static_check("import numpy as np\ndef report(c):\n    return {}") == []
    assert static_check("import os\n")
    assert static_check("def report(c):\n    return open('x').read()")
    assert static_check("def report(c):\n    return c.__class__")
    assert static_check("import numpy as np\nnp.load('x')")


def test_run_plugin_roundtrip_and_timeout(tmp_path):
    p = tmp_path / "p.py"
    p.write_text("import numpy as np\nTRACE_NODES=['a']\ndef report(c):\n    return {'n': int(c['X'].shape[0])}\n")
    ctx = {"X": np.zeros((3, 4)), "y": np.zeros(3), "logits": np.zeros((3, 2)), "pred": np.zeros(3),
           "train_X": np.zeros((2, 4)), "train_y": np.zeros(2), "ref_acc": 1.0, "traces": {}}
    assert run_plugin(p, ctx) == {"n": 3}
    assert declared_trace_nodes(p.read_text()) == ["a"]
    q = tmp_path / "q.py"
    q.write_text("def report(c):\n    while True:\n        pass\n")
    try:
        run_plugin(q, ctx, timeout=2)
        raise AssertionError("expected timeout")
    except PluginError as e:
        assert "exceeded" in str(e)
