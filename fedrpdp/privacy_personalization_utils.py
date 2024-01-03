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
import numpy as np
from typing import List, Union, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

from .accountants.rdp import RDPAccountant, compute_rdp_4fed, compute_rdp_4sgd
from .accountants.analysis import rdp as privacy_analysis


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
                 delta: float=1e-5,
                 outer_steps: Optional[int] = None,
                 client_rate: Optional[float] = None):
        
        self.noise_multiplier = noise_multiplier
        self.inner_steps = steps
        self.outer_steps = outer_steps
        self.outer_rate = client_rate
        self.delta = delta
        
    def estimate(self, alpha:list=None, sample_rates:list=None, plot=False, ret_fn='inv_fit_fn'):
        if alpha is None:
            alpha = RDPAccountant.generate_rdp_orders()

        if sample_rates is None:
            sample_rates = np.concatenate([np.arange(0, 1.0, 0.01), np.arange(0, 0.1, 0.001), [1.0]])
            self.sample_rates = sorted(np.unique(sample_rates))[1:] # delete q = 0.0
        minq = min(self.sample_rates)

        # examples, upperbound, lowerbound = self._generate_examples(sample_rates, alphas)
        # generate examples
        self.examples = []
        lower_bound, upper_bound = None, None
        for inner_rate in self.sample_rates:
            # compute rdp
            if self.outer_steps is None:
                total_rdp = compute_rdp_4sgd(
                        noise_multiplier=self.noise_multiplier,
                        sample_rate=inner_rate, 
                        steps=self.inner_steps,
                        alphas=alpha
                    )
            else:
                total_rdp = compute_rdp_4fed(
                        noise_multiplier=self.noise_multiplier,
                        inner_rate=inner_rate, 
                        outer_rate=self.outer_rate, 
                        inner_steps=self.inner_steps, 
                        outer_steps=self.outer_steps,
                        alphas=alpha
                    )
            # access the best alpha
            eps, _ = privacy_analysis.get_privacy_spent(
                orders=alpha, rdp=total_rdp, delta=self.delta)
            self.examples.append(eps)

            if np.abs(inner_rate - 1.0) < 1e-5: # if inner_rate == 1.
                upper_bound = eps
            
            if np.abs(inner_rate - minq) < 1e-5: # if inner_rate is the minimum
                lower_bound = eps

        if upper_bound is None: # which means 1.0 is not included in the sample_rates.
            total_rdp = compute_rdp_4fed(
                    noise_multiplier=self.noise_multiplier,
                    inner_rate=1.0, 
                    outer_rate=self.outer_rate, 
                    inner_steps=self.inner_steps, 
                    outer_steps=self.outer_steps,
                    alpha=alpha
                )
            upper_bound, _ = privacy_analysis.get_privacy_spent(
                orders=alpha, rdp=total_rdp, delta=self.delta)

        return self.examples, lower_bound, upper_bound
    
    def curve_fit(self, examples):
        sample_rates = np.array(self.sample_rates, dtype=np.float32)
        examples = np.array(examples, dtype=np.float32)
        # func = lambda x, a, b, c: a * np.exp(b*x) + c
        func = lambda x, a, b, c: np.exp(a*x + b) + c
        popt, pcov = curve_fit(func, sample_rates, examples)
        r2 = r2_score(func(sample_rates, *popt), examples)
        print('r2 score of the curve fitting.', r2)
        return lambda x: func(x, *popt), popt

    def plot_priv_cost_curve(self, fit_curve, popt):
        font = {'weight': 'bold', 'size': 13}

        # matplotlib.rc('font', **font)
        # my_colors = list(mcolors.TABLEAU_COLORS.keys())

        plt.figure(num=1, figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(self.sample_rates, self.examples, linewidth=1)
        plt.plot(self.sample_rates, [fit_curve(elem) for elem in self.sample_rates], 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

        #Labels
        plt.title('Best-fit exp func: $\varepsilon(q)=ae^(bx)+c$')
        plt.xlabel(r'Inclusion Probability')
        plt.ylabel(rf'$\epsilon({self.delta})$')
        plt.legend()
        leg = plt.legend()  # remove the frame of Legend, personal choice
        leg.get_frame().set_linewidth(0.0) # remove the frame of Legend, personal choice
        #leg.get_frame().set_edgecolor('b') # change the color of Legend frame
        #plt.show()")
        plt.grid(True)
        plt.savefig("q_opteps_scatter.pdf", bbox_inches='tight')
        plt.show()
