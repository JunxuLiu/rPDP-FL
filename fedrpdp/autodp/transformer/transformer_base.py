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

class Transformer():
    """
    A transformer is a callable object that takes one or more mechanism as input and
    **transform** them into a new mechanism
    """

    def __init__(self):
        self.name = 'generic_transformer'
        self.unary_operator = False  # If true it takes one mechanism as an input,
        # otherwise it could take many, e.g., composition
        self.preprocessing = False  # Relevant if unary is true, it specifies whether the operation
        # is before or after the mechanism, e.g., amplification by sampling is before applying the
        # mechanism, "amplification by shuffling" is after applying the LDP mechanisms
        self.transform = lambda x: x

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


