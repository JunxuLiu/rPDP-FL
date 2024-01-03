"""
Core classes for dpTools:

Mechanism --- A `mechanism' describes a randomized algorithm and its privacy properties.
                All `mechanism's (e.g., those in the `mechanism_zoo' module) inherit this class.

Transformer ---  A transformer takes one or a list of mechanism and outputs another mechanism
                All `transformer's (e.g., those in the `transformer_zoo' module, e.g., amplificaiton
                 by sampling, shuffling, and composition) inherit this class.

Calibrator --- A `calibrator' takes a mechanism with parameters (e.g. noise level) and automatically
                choose those parameters to achieve a prespecified privacy budget.
                All `calibrator's (e.g., the Analytical Gaussian Mechanism calibration, and others
                in `calibrator_zoo'inherit this class)

"""

import numpy as np

class Calibrator():
    """
    A calibrator calibrates noise (or other parameters) meet a pre-scribed privacy budget
    """

    def __init__(self):
        self.name = 'generic_calibrator'

        self.eps_budget = np.inf
        self.delta_budget = 1.0

        self.obj_func = lambda x: 0

        self.calibrate = lambda x: x
        # Input privacy budget, a mechanism with params,  output a set of params that works
        # while minimizing the obj_func as much as possible

    def __call__(self, *args, **kwargs):
        return self.calibrate(*args, **kwargs)