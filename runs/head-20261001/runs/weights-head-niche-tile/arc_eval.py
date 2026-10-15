from pathlib import Path
import importlib.util
_h = Path(__file__).parent
_s = importlib.util.spec_from_file_location('head_core', _h / 'head_core.py')
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)
CORE = _m.make_head_core(_h / 'pop1_core.py')
from cellforge.construction.arc_weight_moe import predict as _p

def predict(weights, task):
    return _p(weights, task, CORE)
