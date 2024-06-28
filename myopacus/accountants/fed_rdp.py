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

from typing import List, Optional, Tuple, Union
import numpy as np
import math

from .accountant import IAccountant
from .analysis import rdp as privacy_analysis 

class FedRDPAccountant(IAccountant):
    
    # def __init__(self, federated: bool = True):
    def __init__(self):
        super().__init__()
        # self.federated = federated

    def init(self, 
             noise_multiplier: float, 
             sample_rate: Union[List[float], float],
             client_rate: Optional[float] = None,
             steps: Optional[int] = None, 
             rounds: Optional[int] = None):
        
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.client_rate = client_rate
        self.steps = steps
        self.rounds = rounds 
    
    def step(self):
        """
        In this funciton, we consider a simplified case where the other hyper-parameters 
        **expect for** the `num_steps` will not change during the training process.

        Besides, for the federated learning scenarios, the current number of communication  
        rounds wiil also be adaptively updated based on the current value of `num_steps`.
        """

        if len(self.history) >= 1:
            num_rounds, num_steps = self.history.pop()[-2:]
            if self.steps is None:
                self.history = [(self.noise_multiplier, self.sample_rate, None, None, num_steps+1)] 
            else:
                if self.steps == 1:
                    self.history = [(self.noise_multiplier, self.sample_rate, self.client_rate, num_rounds+1, self.steps)]
                elif num_steps == self.steps:
                    self.history = [(self.noise_multiplier, self.sample_rate, self.client_rate, num_rounds, 1)]
                elif (num_steps+1) == self.steps:
                    self.history = [(self.noise_multiplier, self.sample_rate, self.client_rate, num_rounds+1, self.steps)]
                else:
                    self.history = [(self.noise_multiplier, self.sample_rate, self.client_rate, num_rounds, num_steps+1)]
        else:
            if self.steps is None:
                self.history = [(self.noise_multiplier, self.sample_rate, None, None, 1)]
            else:
                if self.steps == 1:
                    self.history = [(self.noise_multiplier, self.sample_rate, self.client_rate, 1, self.steps)] 
                else:
                    self.history = [(self.noise_multiplier, self.sample_rate, self.client_rate, 0, 1)]

        # if self.federated:
        #     if len(self.history) >= 1:
        #         num_rounds, num_steps = self.history.pop()[-2:]
        #         if num_steps == self.steps:
        #             self.history = [(self.noise_multiplier, self.sample_rate, self.client_rate, num_rounds, 1)]
        #         elif (num_steps+1) == self.steps:
        #             self.history = [(self.noise_multiplier, self.sample_rate, self.client_rate, num_rounds+1, self.steps)]
        #         else:
        #             self.history = [(self.noise_multiplier, self.sample_rate, self.client_rate, num_rounds, num_steps+1)]
        #     else:
        #         if self.steps == 1:
        #             self.history = [(self.noise_multiplier, self.sample_rate, self.client_rate, 1, 0)] 
        #         else:
        #             self.history = [(self.noise_multiplier, self.sample_rate, self.client_rate, 0, 1)]
        # else:
        #     if len(self.history) >= 1:
        #         num_steps = self.history.pop()[-1]
        #         self.history = [(self.noise_multiplier, self.sample_rate, None, None, num_steps+1)] 
        #     else:
        #         self.history = [(self.noise_multiplier, self.sample_rate, None, None, 1)]
    
    def get_privacy_spent(
        self, *, 
        delta: float, 
        alphas: Optional[List[Union[float, int]]] = None,
        mode: Union[str, int] = "max"
    ) -> Tuple[float, float]:
        if not self.history:
            return 0, 0
        
        if alphas is None:
            alphas = privacy_analysis.generate_rdp_orders()
        
        # if not self.federated:
        #     noise_multiplier, sample_rate, steps = self.history[-1]
        # else:
        noise_multiplier, sample_rate, client_rate, rounds, steps = self.history[-1]
        if not isinstance(sample_rate, float):
            if mode == "max":
                q = max(sample_rate)
            elif mode == "min":
                q = max(sample_rate)
            elif mode == "mean":
                q = np.mean(sample_rate)
            elif mode == "median":
                q = np.median(sample_rate)
            elif isinstance(mode, int):
                q = sample_rate[mode] # the input `mode` specifies the index of a specific sample.
            else:
                raise RuntimeError("The users must specify the expected computation mode when " \
                                "the `get_epsilon` is called within the `FedRDPAccountant` accountant.")
        else:
            q = sample_rate

        COMPUTE_RDP_FUNC = privacy_analysis.compute_rdp if (rounds is None) else privacy_analysis.compute_rdp_fed
        rdp = COMPUTE_RDP_FUNC(
            q=q, 
            client_q=client_rate,
            noise_multiplier=noise_multiplier, 
            steps=steps, 
            rounds=rounds,
            orders=alphas
        )
        
        eps, best_alpha = privacy_analysis.get_privacy_spent(
            orders=alphas, rdp=rdp, delta=delta
        )
        return float(eps), float(best_alpha)

    def get_epsilon(
        self, delta: float, alphas: Optional[List[Union[float, int]]] = None, **kwargs
    ):
        eps, _ = self.get_privacy_spent(delta=delta, alphas=alphas, **kwargs)
        return eps
    
    def get_epsilon_by_id(
        self, id: int, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ):
        eps, _ = self.get_privacy_spent(delta=delta, alphas=alphas, mode=id)
        return eps

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        return "fed_rdp"