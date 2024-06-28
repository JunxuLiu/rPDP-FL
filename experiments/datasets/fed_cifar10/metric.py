import numpy as np

def metric(y_true, y_pred):
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        pred = y_pred.argmax(axis=1)  # get the index of the max log-probability
        # return 1. * (pred == y_true).sum() / len(y_true)
        return (pred == y_true).sum()

    except ValueError:
        return np.nan
