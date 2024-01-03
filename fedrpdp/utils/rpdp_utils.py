import numpy as np
import math
from typing import IO, Any, BinaryIO, Dict, List, Optional, Tuple, Union

from torch import nn, optim
from torch.utils.data import DataLoader

from fedrpdp.accountants.utils import get_noise_multiplier as get_nm
from fedrpdp.privacy_personalization_utils import PrivCostEstimator
from pynverse import inversefunc

class FilterManager:
    def __init__(self):
        pass 

    def get_privacy_engine(self,
                           norm_sq_budgets: List[float],
                           module: nn.Module, 
                           data_loader: DataLoader,
                           noise_multiplier: float,
                           max_grad_norm: Union[float, List[float]]):
    
        assert norm_sq_budgets is not None, "the inputs `budgets` must not be None."
        from torchdp import PrivacyEngine
        return PrivacyEngine(
            module=module,
            batch_size=len(data_loader.dataset),
            sample_size=len(data_loader.dataset),
            alphas = generate_rdp_orders(),
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            norm_sq_budget = norm_sq_budgets,
            should_clip = True,
        )
    
    def get_epsilon(self, privacy_engine, 
                    delta: float = 1e-5, 
                    id: int = None):
        return privacy_engine.get_epsilon(privacy_engine.norm_sq_budget[id], delta)[0]

class rPDPManager:
    def __init__(self, accountant: str):
        self.accountant = accountant

    def get_privacy_engine(self,
                       module: nn.Module, 
                       optimizer: optim.Optimizer,
                       data_loader: DataLoader,
                       sample_rate: Union[float, List[float]],
                       noise_multiplier: float,
                       max_grad_norm: Union[float, List[float]]):
        
        from fedrpdp import PrivacyEngine
        privacy_engine = PrivacyEngine(accountant=self.accountant, sample_rate=sample_rate, noise_multiplier=noise_multiplier)
        privacy_engine.sample_rate = sample_rate

        if self.accountant == 'pers_rdp':
            model, optimizer, train_loader = privacy_engine.make_private_with_personalization(
                module=module,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm
            )

        elif self.accountant == 'rdp':
            model, optimizer, train_loader = privacy_engine.make_private(
                module=module,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm
            )

        print(f"Using sigma={optimizer.noise_multiplier} and C={max_grad_norm}")

        return privacy_engine, model, optimizer, train_loader
        
    def get_epsilon(self, privacy_engine, 
                    delta: float = 1e-5, 
                    id: int = None):
        if self.accountant == 'rdp':
            return privacy_engine.get_epsilon(target_delta=delta)
        elif self.accountant == 'pers_rdp':
            return privacy_engine.get_epsilon(sample_id=id, target_delta=delta)
    
def get_sample_rate_curve(target_delta: float,
                      noise_multiplier: float,
                      num_updates: int,
                      num_rounds: int = None,
                      client_rate: float = None):
    
    pce = PrivCostEstimator(
        noise_multiplier = noise_multiplier, 
        steps = num_updates, 
        outer_steps = num_rounds, 
        client_rate = client_rate,
        delta = target_delta)
    
    examples, lower_bound, upper_bound = pce.estimate()
    fit_fn, opt_params = pce.curve_fit(examples)
    # pce.plot_priv_cost_curve(fit_fn, opt_params)

    def inv_fit_fn(x):
        if x >= upper_bound:
            return 1.0
        elif x <= lower_bound:
            return 0.0
        else:
            return float(inversefunc(fit_fn)(x))
    return inv_fit_fn

def get_per_sample_rates(target_epsilon: List[float],
                         noise_multiplier: float,
                         num_updates: int,
                         num_rounds: int = None,
                         client_rate: float = None,
                         target_delta: float = 1e-5):
    pce = PrivCostEstimator(
        noise_multiplier = noise_multiplier, 
        steps = num_updates, 
        outer_steps = num_rounds, 
        client_rate = client_rate,
        delta = target_delta)
    
    examples, lower_bound, upper_bound = pce.estimate()
    fit_fn, opt_params = pce.curve_fit(examples)
    pce.plot_priv_cost_curve(fit_fn, opt_params)

    def inv_fit_fn(x):
        if x >= upper_bound:
            return 1.0
        elif x <= lower_bound:
            return 0.0
        else:
            return inversefunc(fit_fn)(x)
    
    return [float(inv_fit_fn(eps)) for eps in target_epsilon]
     

def get_noise_multiplier(target_epsilon: float,
                         num_updates: int,
                         num_rounds: int = 1,
                         target_delta: float = 1e-5):
    
    return get_nm(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=1.0,
        epochs=num_updates * num_rounds,
        accountant='rdp'
    )
    
def generate_rdp_orders():
    dense = 1.07
    alpha_list = [int(dense ** i + 1) for i in range(int(math.floor(math.log(1000, dense))) + 1)]
    alpha_list = np.unique(alpha_list)
    return alpha_list

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
