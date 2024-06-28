from collections import OrderedDict
from typing import Any, List, Mapping, TypeVar, Optional, Union

from numpy import sort

from .accountant import IAccountant


T_state_dict = TypeVar("T_state_dict", bound=Mapping[str, Any])

"""Author : Boenisch et al. (NeurIPS, 2023), modified by Junxu."""
class IndividualAccountant(IAccountant):

    def __init__(self, accountants: List[IAccountant], n_groups: int):
        """
        This is a wrapper around multiple accountants which are supposed to
        correspond to a privacy group (data points of training data who share
        the same privacy budget). The groups are supposed to be in ascending
        order in terms of their budgets.
        """
        self.n_groups = n_groups
        self.nm_scalars = [1.0] * n_groups
        self.sr_scalars = [1.0] * n_groups
        self.accountants = accountants
        self.history = []   # used to check if privacy histories were updated

    def step(self):
        pass

    def get_epsilon(self, delta: float, optimal: Optional[bool] = None,
                    min_alpha: Optional[Union[List[float], float]] = None,
                    max_alpha: Optional[Union[List[float], float]] = None,
                    **kwargs) -> List[float]:
        """
        Returns the expended privacy costs epsilon of all privacy groups.
        """
        if optimal:
            if not isinstance(max_alpha, List):
                max_alpha = [max_alpha for _ in range(self.n_groups)]
            if not isinstance(min_alpha, List):
                min_alpha = [min_alpha for _ in range(self.n_groups)]
            return [accountant.get_epsilon(
                delta=delta, optimal=optimal, min_alpha=min_alpha[group],
                max_alpha=max_alpha[group], **kwargs)
                for group, accountant in enumerate(self.accountants)]
        return [accountant.get_epsilon(delta=delta, **kwargs)
                for group, accountant in enumerate(self.accountants)]

    def __len__(self) -> int:
        return len(self.accountants[0].history)

    @classmethod
    def mechanism(cls) -> str:
        return "idp"

    # def state_dict(self, destination: T_state_dict = None) -> T_state_dict:
    #     if destination is None:
    #         destination = OrderedDict()
    #     destination["accountants"] = self.accountants
    #     destination["mechanism"] = self.__class__.mechanism
    #     destination["n_groups"] = self.n_groups
    #     destination["nm_scalars"] = self.nm_scalars
    #     destination["sr_scalars"] = self.sr_scalars
    #     return destination

    # def load_state_dict(self, state_dict: T_state_dict):
    #     if state_dict is None or len(state_dict) == 0:
    #         raise ValueError(
    #             "state dict is either None or empty and hence cannot be loaded"
    #             " into Privacy Accountant."
    #         )
    #     if "history" not in state_dict.keys():
    #         raise ValueError(
    #             "state_dict does not have the key `history`."
    #             " Cannot be loaded into Privacy Accountant."
    #         )
    #     if "mechanism" not in state_dict.keys():
    #         raise ValueError(
    #             "state_dict does not have the key `mechanism`."
    #             " Cannot be loaded into Privacy Accountant."
    #         )
    #     if self.__class__.mechanism != state_dict["mechanism"]:
    #         raise ValueError(
    #             f"state_dict of {state_dict['mechanism']} cannot be loaded into"
    #             f" Privacy Accountant with mechanism {self.__class__.mechanism}"
    #         )
    #     self.accountants = state_dict["accountants"]
    #     self.n_groups = state_dict["n_groups"]
    #     self.nm_scalars = state_dict["nm_scalars"]
    #     self.sr_scalars = state_dict["sr_scalars"]
    #     self.history = [self.accountants[group].history
    #                     for group in range(self.n_groups)]

    # def update_nm_scalars(self, scalars: List[float]):
    #     assert all(scalars == sort(scalars)[::-1])
    #     self.nm_scalars = scalars

    # def update_sr_scalars(self, scalars: List[float]):
    #     assert all(scalars == sort(scalars))
    #     self.sr_scalars = scalars
