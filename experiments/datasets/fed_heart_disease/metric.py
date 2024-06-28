import numpy as np

def metric(y_true, y_pred, per_class=False):
    y_true = y_true.astype("uint8").flatten()
    y_pred = y_pred.flatten()
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        # proposed modification in order to get a metric that calcs on center 2
        # (y=1 only on that center)
        correct_vec = ((y_pred > 0.5) == y_true)
        if per_class:
            keys = np.unique(y_true)
            total_dict = dict(zip(keys, [0]*len(keys)))
            correct_dict = dict(zip(keys, [0]*len(keys)))
            for idx, flag in enumerate(correct_vec):
                if flag:
                    correct_dict[y_true[idx]] += 1
                total_dict[y_true[idx]] += 1
            return correct_vec.sum(), correct_dict, total_dict
        
        return correct_vec.sum()

    except ValueError:
        return np.nan
    