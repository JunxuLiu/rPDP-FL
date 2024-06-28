import matplotlib.pyplot as plt
import numpy as np
from pynverse import inversefunc
from typing import List, Tuple, Callable

import scipy
from sklearn.metrics import r2_score

from myopacus.accountants.analysis import rdp as privacy_analysis

class PrivCostEstimator:
    r"""Compute the privacy cost curves for private SGD / FedAvg mechanisms.

    Attr:
        noise_multiplier: The ratio of the standard deviation of the
            additive Gaussian noise to the L2-sensitivity of the function
            to which it is added. Note that this is same as the standard
            deviation of the additive Gaussian noise when the L2-sensitivity
            of the function is 1.
        inner_steps: The number of iterations of the mechanism.
        outer_steps: The number of rounds (applied to the FL mechanisms).
        outer_rate: Sampling rate of clients (applied to the FL mechanisms).
        delta: The target delta.
    """
    def __init__(self, 
                 noise_multiplier: float, 
                 steps: int, 
                 delta: float = 1e-5,
                 rounds: int = None,
                 client_rate: float = None):
        
        self.noise_multiplier = noise_multiplier
        self.inner_steps = steps
        self.outer_steps = rounds
        self.outer_rate = client_rate
        self.delta = delta
        
    def _estimate(self, qs: List[float] = None, orders: List[float] = None):
        if orders is None:
            orders = privacy_analysis.generate_rdp_orders()

        if qs is None:
            qs = np.concatenate([np.arange(0, 1.0, 0.01), np.arange(0, 0.1, 0.001), [1.0]])
        qs = np.array(sorted(np.unique(qs)), dtype=np.float32)
        self.q_candidates = qs[qs > 1e-5] # remove all candidates that less than 1e-5

        # generate examples
        examples = []
        for q in self.q_candidates:
            # compute rdp
            if self.outer_steps is None:
                total_rdp = privacy_analysis.compute_rdp(q=q, noise_multiplier=self.noise_multiplier, steps=self.inner_steps, orders=orders)
            else:
                total_rdp = privacy_analysis.compute_rdp_fed(q=q, client_q=self.outer_rate, noise_multiplier=self.noise_multiplier, steps=self.inner_steps, rounds=self.outer_steps, orders=orders)
            # access the best alpha
            eps, _ = privacy_analysis.get_privacy_spent(orders=orders, rdp=total_rdp, delta=self.delta)
            examples.append(eps)
        self.examples = np.array(examples, dtype=np.float32)

        if self.outer_steps is None:
            upper_rdp = privacy_analysis.compute_rdp(q=1.0, noise_multiplier=self.noise_multiplier, steps=self.inner_steps, orders=orders)
            lower_rdp = privacy_analysis.compute_rdp(q=1e-5, noise_multiplier=self.noise_multiplier, steps=self.inner_steps, orders=orders)

        else:
            upper_rdp = privacy_analysis.compute_rdp_fed(q=1.0, client_q=self.outer_rate, noise_multiplier=self.noise_multiplier, steps=self.inner_steps, rounds=self.outer_steps, orders=orders)
            lower_rdp = privacy_analysis.compute_rdp_fed(q=1e-5, client_q=self.outer_rate, noise_multiplier=self.noise_multiplier, steps=self.inner_steps, rounds=self.outer_steps, orders=orders)

        self.upper_priv_cost, _ = privacy_analysis.get_privacy_spent(orders=orders, rdp=upper_rdp, delta=self.delta)
        self.lower_priv_cost, _ = privacy_analysis.get_privacy_spent(orders=orders, rdp=lower_rdp, delta=self.delta)

    
    def _curve_fit(self):
        func = lambda x, a, b, c: np.exp(a*x + b) + c
        popt, pcov = scipy.optimize.curve_fit(func, self.q_candidates, self.examples, maxfev=5000)
        r2 = r2_score(func(self.q_candidates, *popt), self.examples)
        print('The R-Squared value of the best-fit curve :', r2)
        self.q_eps_fit_fn = lambda x: func(x, *popt)
        self.q_eps_fit_param = popt
    
    def get_sample_rate_estimator(self):
        self._estimate()
        self._curve_fit()
        def inv_fit_fn(x):
            if x >= self.upper_priv_cost:
                return 1.0
            elif x <= self.lower_priv_cost:
                return 0.0
            else:
                return float(inversefunc(self.q_eps_fit_fn)(x))
        return inv_fit_fn

def MultiLevels(n_levels, ratios, values, size):
    assert abs(sum(ratios) - 1.0) < 1e-6, "the sum of `ratios` must equal to one."
    assert len(ratios) == len(values) and len(ratios) == n_levels, "both the size of `ratios` and the size of `values` must equal to the value of `n_levels`."

    target_epsilons = [0]*size
    pre = 0
    for i in range(n_levels-1):
        l = int(size * ratios[i])
        target_epsilons[pre: pre+l] = [values[i]]*l
        pre = pre+l
    l = size-pre
    target_epsilons[pre:] = [values[-1]]*l
    return np.array(target_epsilons)

def MixGauss(ratios, means_and_stds, size):
    assert abs(sum(ratios) - 1.0) < 1e-6, "the sum of `ratios` must equal to one."
    assert len(ratios) == len(means_and_stds), "the size of `ratios` and `means_and_stds` must be equal."

    target_epsilons = []
    pre = 0
    for i in range(size):
        # random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]) # 掷骰子（多项式分布）
        dist_idx = np.argmax(np.random.multinomial(1, ratios))
        value = np.random.normal(loc=means_and_stds[dist_idx][0], scale=means_and_stds[dist_idx][1])
        target_epsilons.append(value)
    
    return np.array(target_epsilons)

def Gauss(mean_and_std, size):
    return np.random.normal(loc=mean_and_std[0], scale=mean_and_std[1], size=size)

def Pareto(shape, lower, size):
    return np.random.pareto(shape, size) + lower

GENERATE_EPSILONS_FUNC = {
    "ThreeLevels": lambda n, params: MultiLevels(3, *params, n),
    "BoundedPareto": lambda n, params: Pareto(*params, n), 
    "BoundedMixGauss": lambda n, params: MixGauss(*params, n),
}