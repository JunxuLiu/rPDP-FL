import numpy as np

def metric(y_true, y_pred, per_class=False):
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        correct_vec = (y_pred.argmax(axis=1) == y_true)
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
