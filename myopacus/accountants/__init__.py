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

from .accountant import IAccountant
from .gdp import GaussianAccountant
from .prv import PRVAccountant
from .rdp import RDPAccountant
from .fed_rdp import FedRDPAccountant
# from .pers_rdp import PersRDPAccountant
# from .fed_pers_rdp import FedPersRDPAccountant
from .idp import IndividualAccountant

__all__ = [
    "IAccountant",
    "GaussianAccountant",
    "RDPAccountant",
    "PersRDPAccountant",
    "FedRDPAccountant"
    "FedPersRDPAccountant"
]

def create_accountant(mechanism: str) -> IAccountant:
    if mechanism == "rdp":
        return RDPAccountant()
    elif mechanism == "gdp":
        return GaussianAccountant()
    elif mechanism == "prv":
        return PRVAccountant()
    elif mechanism == "fed_rdp":
        return FedRDPAccountant()
    
    raise ValueError(f"Unexpected accounting mechanism: {mechanism}")

def create_accountant_fedlean(mechanism: str, n_clients: int, **kwargs) -> IAccountant:
    assert n_clients>1, "The input `n_clients` must be larger than 1 when the federated accountants is enabled."
    accountants = [create_accountant(mechanism=mechanism, **kwargs) for _ in range(n_clients)]
    return IndividualAccountant(accountants=accountants, n_groups=n_clients)
