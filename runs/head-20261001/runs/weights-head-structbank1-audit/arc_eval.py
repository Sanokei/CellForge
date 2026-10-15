import importlib.util
import sys
from pathlib import Path
_h = Path(__file__).parent
sys.path.insert(0, str(_h))
_s = importlib.util.spec_from_file_location('skillcore', _h / 'skillcore.py')
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)
_m.load_legacy(_h / 'pop1_core.py')

def predict(weights, task):
    return _m.predict(weights, task)
