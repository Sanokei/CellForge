from pathlib import Path
from cellforge.construction.arc_weight_moe import make_core,predict as predict_weights
CORE=make_core(Path(__file__).with_name("pop1_core.py"))
def predict(weights,task):return predict_weights(weights,task,CORE)
