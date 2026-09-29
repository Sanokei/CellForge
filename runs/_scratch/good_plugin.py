import numpy as np
TRACE_NODES = []
def report(ctx):
    pred, y, X = ctx["pred"], ctx["y"], ctx["X"]
    acc = float((pred == y).mean())
    # class-mean signals (dev_train) as a heatmap: what does each class look like on average?
    means = np.stack([ctx["train_X"][ctx["train_y"] == c].mean(0) for c in range(10)])
    return {"acc": acc, "mean_abs_signal": float(np.abs(X).mean()), "heatmaps": {"class_means": means}}
