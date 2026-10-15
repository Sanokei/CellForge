"""Explicit interactive-task boundary for ONE_STANDARD semantic discovery.
No synthetic accuracy data or recomputed outcome is substituted for actual games.
"""
class InteractiveContractUnsupported(RuntimeError): pass
def make_tasks(seed,n):
    raise InteractiveContractUnsupported("Unqualified ARC3 interactive pilot cannot represent make_tasks/predict supervised accuracy; use RESULTS + paired completed levels, not a synthetic proxy")
def predict(tensors,x):
    raise InteractiveContractUnsupported("No classification proxy is accepted for ARC3")

