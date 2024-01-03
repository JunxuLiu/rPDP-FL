#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from pynverse import inversefunc
import math
import numpy as np
from typing import List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

from .accountants.rdp import RDPAccountant
from .accountants.analysis import rdp as privacy_analysis

def plot_priv_cost_curve(self, x, y, fit_curve, popt):
        font = {'weight': 'bold', 'size': 13}

        # matplotlib.rc('font', **font)
        # my_colors = list(mcolors.TABLEAU_COLORS.keys())

        plt.figure(num=1, figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(x, y, linewidth=1)
        plt.plot(x, [fit_curve(elem) for elem in x], 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

        #Labels
        plt.title(r'Subsampled gaussian with $\sigma=10.0, \delta=1e-5$\n(Quadratic Function Fitting)')
        plt.xlabel(r'sampling ratio $\alpha$')
        plt.ylabel(rf'$\epsilon({self.delta})$')
        plt.legend()
        leg = plt.legend()  # remove the frame of Legend, personal choice
        leg.get_frame().set_linewidth(0.0) # remove the frame of Legend, personal choice
        #leg.get_frame().set_edgecolor('b') # change the color of Legend frame
        #plt.show()")
        plt.grid(True)
        plt.savefig("q_opteps_scatter.pdf", bbox_inches='tight')
        plt.show()

def generate_rdp_orders():
    dense = 1.07
    alpha_list = [int(dense ** i + 1) for i in range(int(math.floor(math.log(1000, dense))) + 1)]
    alpha_list = np.unique(alpha_list)
    return alpha_list

def _compute_rdp_4fed(
    noise_multiplier: float, 
    alpha: float, 
    inner_rate: float, 
    outer_rate: float, 
    inner_steps: int) -> float:
    r"""
    noise_multiplier: The ratio of the standard deviation of the
            additive Gaussian noise to the L2-sensitivity of the function
            to which it is added.
    """
    assert inner_rate >= 0.0 and inner_rate <= 1.0 and outer_rate >= 0.0 and outer_rate <= 1.0, "both the input `inner_rate` and `outer_rate` must be a positive real number in [0,1]."
    assert inner_steps >= 1, "The input `inner_steps` must be a positive integer larger than 1."

    inner_rdp = privacy_analysis.compute_rdp(
                    q=inner_rate,
                    noise_multiplier=noise_multiplier,
                    steps=inner_steps,
                    orders=alpha)
    
    if outer_rate == 1.: # no outer-level privacy amplification
        return inner_rdp
    else:
        log_term_1 = np.log(1. - outer_rate)
        log_term_2 = np.log(outer_rate) + (alpha-1) * inner_rdp
        outer_rdp = privacy_analysis._log_add(log_term_1, log_term_2)/(alpha-1)
        return outer_rdp

def compute_pers_rdp(
    *,
    noise_multiplier: float, 
    alphas: Union[List[float], float], 
    inner_rate: float, 
    outer_rate: float, 
    inner_steps: int, 
    outer_rounds: int
) -> Union[List[float], float]:

    if isinstance(alphas, float):
        rdp = _compute_rdp_4fed(noise_multiplier, alphas, inner_rate=inner_rate, inner_steps=inner_steps, outer_rate=outer_rate)
    else:
        rdp = np.array([_compute_rdp_4fed(noise_multiplier, float(alpha), inner_rate=inner_rate, inner_steps=inner_steps, outer_rate=outer_rate) for alpha in alphas])

    return rdp * outer_rounds

# def get_privacy_spent(rdp_func, k, T, noise_multiplier, sample_rate, alphas, delta, orders=None):
#     total_rdp = compute_pers_rdp(
#                     noise_multiplier=noise_multiplier,
#                     inner_rate=inner_rate, 
#                     outer_rate=outer_rate, 
#                     inner_steps=inner_steps, 
#                     outer_rounds=outer_rounds,
#                     orders=orders,
#                 )
#     eps, best_alpha = privacy_analysis.get_privacy_spent(
#             orders=alphas, rdp=total_rdp, delta=delta
#         )
#     # if orders is not None:
#     #     optimal_order, epsilon = _ternary_search(lambda order: _apply_pdp_sgd_analysis(rdp_func, order, k, T, p, delta), options=orders, iterations=72)
#     # else:
#     #     ## TODO: have not been updated
#     #     optimal_order, epsilon = _ternary_search(lambda order: _apply_pdp_sgd_analysis(rdp_func, order, k, T, p, delta), left=1, right=512, iterations=100)
#     return  eps, best_alpha

class PrivCostEstimator:
    r"""Compute the privacy cost curves for private SGD / FedAvg mechanisms.

    Attr:
        noise_multiplier: The ratio of the standard deviation of the
            additive Gaussian noise to the L2-sensitivity of the function
            to which it is added. Note that this is same as the standard
            deviation of the additive Gaussian noise when the L2-sensitivity
            of the function is 1.
        inner_steps: The number of iterations of the mechanism.
        outer_rounds: The number of rounds (applied to the FL mechanisms).
        outer_rate: Sampling rate of clients (applied to the FL mechanisms).
        delta: The target delta.
    """
    def __init__(self, noise_multiplier: float, inner_steps: int, outer_rounds: int=0, outer_rate: float=1.0, delta: float=1e-5, mech:str='fedlearn'):
        self.noise_multiplier = noise_multiplier
        self.inner_steps = inner_steps
        self.outer_rounds = outer_rounds
        self.outer_rate = outer_rate
        self.delta = delta
        
    def estimate(self, alphas:list=None, sample_rates:list=None, plot=False, ret_fn='inv_fit_fn'):
        if alphas is None:
            # alphas = generate_rdp_orders()
            alphas = RDPAccountant().DEFAULT_ALPHAS

        if sample_rates is None:
            sample_rates = np.concatenate([np.arange(0, 1.0, 0.01), np.arange(0, 0.1, 0.001), [1.0]])
            self.sample_rates = sorted(np.unique(sample_rates))[1:] # delete q = 0.0
        minq = min(self.sample_rates)

        # examples, upperbound, lowerbound = self._generate_examples(sample_rates, alphas)
        # generate examples
        examples = []
        lower_bound, upper_bound = None, None
        for inner_rate in self.sample_rates:
            # compute rdp
            total_rdp = compute_pers_rdp(
                    noise_multiplier=self.noise_multiplier,
                    inner_rate=inner_rate, 
                    outer_rate=self.outer_rate, 
                    inner_steps=self.inner_steps, 
                    outer_rounds=self.outer_rounds,
                    alphas=alphas
                )
            # access the best alpha
            eps, _ = privacy_analysis.get_privacy_spent(
                orders=alphas, rdp=total_rdp, delta=self.delta)
            examples.append(eps)

            if np.abs(inner_rate - 1.0) < 1e-5: # if inner_rate == 1.
                upper_bound = eps
            
            if np.abs(inner_rate - minq) < 1e-5: # if inner_rate is the minimum
                lower_bound = eps

        if upper_bound is None: # which means 1.0 is not included in the sample_rates.
            total_rdp = compute_pers_rdp(
                    noise_multiplier=self.noise_multiplier,
                    inner_rate=1.0, 
                    outer_rate=self.outer_rate, 
                    inner_steps=self.inner_steps, 
                    outer_rounds=self.outer_rounds,
                    alphas=alphas
                )
            upper_bound, _ = privacy_analysis.get_privacy_spent(
                orders=alphas, rdp=total_rdp, delta=self.delta)

        return examples, lower_bound, upper_bound
    
    def curve_fit(self, examples):
        sample_rates = np.array(self.sample_rates, dtype=np.float32)
        examples = np.array(examples, dtype=np.float32)
        func = lambda x, a, b, c: a * np.exp(b*x) + c
        popt, pcov = curve_fit(func, sample_rates, examples)
        r2 = r2_score(func(sample_rates, *popt), examples)
        print('r2 score of the curve fitting.', r2)
        return lambda x: func(x, *popt), popt

def per_sample_rate_estimation(
        actual_epsilons: Union[List[float], float], 
        delta: float=1e-5,
        noise_multiplier: float=1.1, 
        inner_steps: int=1, 
        outer_rounds: int=0, 
        outer_rate: float=1.0, 
        plot: bool=False
    ):

    pce = PrivCostEstimator(
        noise_multiplier=noise_multiplier, 
        inner_steps=inner_steps, 
        outer_rounds=outer_rounds, 
        outer_rate=outer_rate, 
        delta=delta,
        mech='fedlearn')
    examples, lower_bound, upper_bound = pce.estimate()

    fit_fn, opt_params = pce.curve_fit(examples)

    def inv_fit_fn(x):
        if x >= upper_bound:
            return 1.0
        elif x <= lower_bound:
            return 0.0
        else:
            return inversefunc(fit_fn)(x)
        
    per_sample_rates = [float(inv_fit_fn(eps)) for eps in actual_epsilons]

    if plot:
        plot_priv_cost_curve()

    return per_sample_rates