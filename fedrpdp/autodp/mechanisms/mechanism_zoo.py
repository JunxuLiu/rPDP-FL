"""
'mechanism_zoo' module implements popular DP mechanisms with their privacy guarantees

ref: https://github.com/imcjp/Opacus-DPCR/blob/65e0709179adb9b4825c957f9248cf3af3196b91/dpcrpy/utils/dpTools/mechanism_zoo.py#L147
"""
import math

from .mechanism_base import Mechanism
from ..analysis import rdp_bank, utils # dp_bank, fdp_bank
from ..transformer import transformer_zoo

from scipy.optimize import minimize_scalar


# Example of a specific mechanism that inherits the Mechanism class
class GaussianMechanism(Mechanism):
    def __init__(self, sigma, name='Gaussian',
                 RDP_off=False, approxDP_off=False, fdp_off=True,
                 use_basic_RDP_to_approxDP_conversion=False,
                 use_fDP_based_RDP_to_approxDP_conversion=False):
        # the sigma parameter is the std of the noise divide by the l2 sensitivity
        Mechanism.__init__(self)
        print(f"Gaussian Mechanism rdp budget: {self.RenyiDP(1.0)}")
        self.name = name # When composing
        self.params = {'sigma': sigma} # This will be useful for the Calibrator
        # TODO: should a generic unspecified mechanism have a name and a param dictionary?

        self.delta0 = 0
        if not RDP_off:
            new_rdp = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
            if use_fDP_based_RDP_to_approxDP_conversion:
                # This setting is slightly more complex, which involves converting RDP to fDP,
                # then to eps-delta-DP via the duality
                self.propagate_updates(new_rdp, 'RDP', fDP_based_conversion=True)
            elif use_basic_RDP_to_approxDP_conversion:
                self.propagate_updates(new_rdp, 'RDP', BBGHS_conversion=False)
            else:
                # This is the default setting with fast computation of RDP to approx-DP
                self.propagate_updates(new_rdp, 'RDP')
                print(f"Gaussian Mechanism rdp budget (updated): {self.RenyiDP(1.0)}")

        # if not approxDP_off: # Direct implementation of approxDP
        #     new_approxdp = lambda x: dp_bank.get_eps_ana_gaussian(sigma, x)
        #     self.propagate_updates(new_approxdp,'approxDP_func')

        # if not fdp_off: # Direct implementation of fDP
        #     fun1 = lambda x: fdp_bank.log_one_minus_fdp_gaussian({'sigma': sigma}, x)
        #     fun2 = lambda x: fdp_bank.log_neg_fdp_grad_gaussian({'sigma': sigma}, x)
        #     self.propagate_updates([fun1,fun2],'fDP_and_grad_log')
        #     # overwrite the fdp computation with the direct computation
        #     self.fdp = lambda x: fdp_bank.fDP_gaussian({'sigma': sigma}, x)

        # the fDP of gaussian mechanism is equivalent to analytical calibration of approxdp,
        # so it should have been automatically handled numerically above

        # Discussion:  Sometimes delta as a function of eps has a closed-form solution
        # while eps as a function of delta does not
        # Shall we represent delta as a function of eps instead?


class ExactGaussianMechanism(Mechanism):
    """
    The Gaussian mechanism to use in practice with tight direct computation of everything
    """
    def __init__(self, sigma=None, name='Gaussian'):
        # the sigma parameter is the std of the noise divide by the l2 sensitivity
        Mechanism.__init__(self)

        self.name = name # When composing
        self.params = {'sigma': sigma} # This will be useful for the Calibrator
        self.delta0 = 0
        if sigma is not None:
            new_rdp = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
            self.propagate_updates(new_rdp, 'RDP')
            # Overwrite the approxDP and fDP with their direct computation
            self.approxDP = lambda x: dp_bank.get_eps_ana_gaussian(sigma, x)
            self.fDP = lambda x: fdp_bank.fDP_gaussian({'sigma': sigma}, x)


class LaplaceMechanism(Mechanism):
    """
    param params:
    'b' --- is the is the ratio of the scale parameter and L1 sensitivity
    """
    def __init__(self, b=None, name='Laplace'):

        Mechanism.__init__(self)

        self.name = name
        self.params = {'b': b} # This will be useful for the Calibrator
        self.delta0 = 0
        if b is not None:
            new_rdp = lambda x: rdp_bank.RDP_laplace({'b': b}, x)
            self.propagate_updates(new_rdp, 'RDP')

class RandresponseMechanism(Mechanism):

    """
        param params:
        'p' --- is the Bernoulli probability p of outputting the truth.
        """

    def __init__(self, p=None, name='Randresponse'):
        Mechanism.__init__(self)

        self.name = name
        self.params = {'p': p}  # This will be useful for the Calibrator
        self.delta0 = 0
        if p is not None:
            new_rdp = lambda x: rdp_bank.RDP_randresponse({'p': p}, x)
            self.propagate_updates(new_rdp, 'RDP')


class PureDP_Mechanism(Mechanism):
    def __init__(self, eps, name='PureDP'):
        # the eps parameter is the pure DP parameter of this mechanism
        Mechanism.__init__(self)

        self.name = name # Used for generating new names when composing
        self.params = {'eps': eps} #

        self.propagate_updates(eps, 'pureDP')

        # ------- I verified that the following options give the same results ----
        # def new_rdp(x):
        #     return rdp_bank.RDP_pureDP({'eps': eps}, x)
        #
        # if use_basic_RDP_to_approxDP_conversion:
        #     self.propagate_updates(new_rdp, 'RDP', BBGHS_conversion=False)
        # else:


        #     self.propagate_updates(new_rdp, 'RDP')


class SubsampleGaussianMechanism(Mechanism):
    """
    This one is used as an example for calibrator with subsampled Gaussian mechanism
    """
    def __init__(self,params,name='SubsampleGaussian'):
        Mechanism.__init__(self)
        self.name=name
        self.params={'prob':params['prob'],'sigma':params['sigma'],'coeff':params['coeff']}
        # create such a mechanism as in previously
        subsample = transformer_zoo.AmplificationBySampling()  # by default this is using poisson sampling
        mech = GaussianMechanism(sigma=params['sigma'])

        # Create subsampled Gaussian mechanism
        SubsampledGaussian_mech = subsample(mech, params['prob'], improved_bound_flag=True)

        # Now run this for niter iterations
        compose = transformer_zoo.Composition()
        mech = compose([SubsampledGaussian_mech], [params['coeff']])
        
        # Now we've get the composed mechanism, but haven't implement the rdp-relative methods
        # 1) extract the RDP function
        # 2) assign it to the current mech being constructed
        rdp_total = mech.RenyiDP # the function system.analysis.converter.pointwise_minimum(self.RenyiDP, func)
        self.propagate_updates(rdp_total, type_of_update='RDP')


class ComposedGaussianMechanism(Mechanism):
    """
    This one is used as an example for calibrator with composed Gaussian mechanism
    """
    def __init__(self,params,name='SubsampleGaussian'):
        Mechanism.__init__(self)
        self.name=name
        self.params={'sigma':params['sigma'],'coeff':params['coeff']}
        # create such a mechanism as in previously

        mech = GaussianMechanism(sigma=params['sigma'])
        # Now run this for coeff iterations
        compose = transformer_zoo.Composition()
        mech = compose([mech], [params['coeff']])

        # Now we get it and let's extract the RDP function and assign it to the current mech being constructed
        rdp_total = mech.RenyiDP
        self.propagate_updates(rdp_total, type_of_update='RDP')


# # Example 1: Short implementation of noisy gradient descent mechanism as a composition of GMs
# class NoisyGD_mech(GaussianMechanism):
#     def __init__(self,sigma_list,name='NoisyGD'):
#         GaussianMechanism.__init__(self, sigma=np.sqrt(np.sum(sigma_list)),name=name)
#         self.params = {'sigma_list':sigma_list}
#
# # The user could log sigma_list and then just declare a NoisyGD_mech object.
# mech = NoisyGD_mech(sigma_list)
# mech.get_approxDP(delta=1e-6)
#
#
# # Example 2: Implementing NoisySGD from basic building blocks
# subsample = Transformers.Subsample(prob=0.01)
# mech = Mechanisms.GaussianMechanism(sigma=5.0)
# # Create subsampled Gaussian mechanism
# SubsampledGaussian_mech = subsample(mech)
#
# # Now run this for 100 iterations
# compose = Transformers.Composition()
# NoisySGD_mech = compose(mechanism_list = [SubsampledGaussian_mech],coeffs_list=[100])
#
#
# # Example 3: You could also package this together by defining a NoisySGD mechanism
# class NoisySGD_mech(Mechanism):
#     def __init__(self,prob,sigma,niter,name='NoisySGD'):
#         Mechanism.__init__()
#         self.name=name
#         self.params={'prob':prob,'sigma':sigma,'niter':niter}
#
#         rdp = rdp_bank.subsampled_gaussian({'prob':params['prob'],'sigma':params['sigma']})
#         self.propagate_updates(rdp,type_of_update='RDP')
#
#
# # Example 4: Online decision. Hetereogenous sigma decided online
# # (maybe as a function of computed eps)
# # Alternatively if we want to do it via composition, so we can make online decision about
# # the sigma in the sigma_list
#
# delta = 1e-6
# online_sgd = Mechanisms.SubsampledGaussian_mech(prob=prob,sigma=sigma)
# compose = Transformers.Composition()
# for i in range(niter):
#     eps = online_ngd.get_approxDP(delta)
#     #determine the next prob, sigma
#     prob, sigma = func(eps)
#     mech_cur = Mechanisms.SubsampledGaussian_mech(prob=prob, sigma=sigma)
#     online_ngd = compose([online_ngd, mech_cur])
#
# # The above is quite general and can be viewed as a privacy accountant
