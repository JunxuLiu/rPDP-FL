import numpy as np

def MultiLevels(n_levels, ratios, values, size):
    assert abs(sum(ratios) - 1.0) < 1e-6, "the sum of `ratios` must equal to one."
    assert len(ratios) == len(values) and len(ratios) == n_levels, "both the size of `ratios` and the size of `values` must equal to the value of `n_levels`."

    tgt_epsilons = [0]*size
    pre = 0
    for i in range(n_levels-1):
        l = int(size * ratios[i])
        tgt_epsilons[pre: pre+l] = [values[i]]*l
        pre = pre+l
    l = size-pre
    tgt_epsilons[pre:] = [values[-1]]*l
    return np.array(tgt_epsilons)

def MixGauss(ratios, means_and_stds, size):
    assert abs(sum(ratios) - 1.0) < 1e-6, "the sum of `ratios` must equal to one."
    assert len(ratios) == len(means_and_stds), "the size of `ratios` and `means_and_stds` must be equal."

    tgt_epsilons = []
    for i in range(size):
        dist_idx = np.argmax(np.random.multinomial(1, ratios))
        value = np.random.normal(loc=means_and_stds[dist_idx][0], scale=means_and_stds[dist_idx][1])
        tgt_epsilons.append(value)
    
    return np.array(tgt_epsilons)

def Gauss(mean_and_std, size):
    return np.random.normal(loc=mean_and_std[0], scale=mean_and_std[1], size=size)


def Pareto(shape, mean, scale, size):
    return (np.random.pareto(shape, size) + mean) * scale

BOUNDED_BUDGET_FUNC = lambda x, minimum, maximum: min(max(x, minimum), maximum)

EPSILONS_GEN_FUNC = {
    "TwoLevels": lambda n: MultiLevels(2, [0.8,0.2], [0.5, 5.0], n),
    "ThreeLevels": lambda n: MultiLevels(3, [0.7,0.2,0.1], [0.5, 2.0, 5.0], n),
    "BoundedPareto": lambda n: Pareto(2.5, 0.01, 5.0, n), 
    "BoundedMixGauss": lambda n: MixGauss([0.7,0.2,0.1], [(0.5, 0.1), (2.0, 0.5), (5.0, 1.0)], n),
    # "BoundedGauss": lambda n: Gauss((1.0, 0.2), n),
}