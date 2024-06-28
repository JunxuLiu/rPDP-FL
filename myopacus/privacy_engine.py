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
from itertools import chain
import numpy as np
import os
from typing import IO, Any, BinaryIO, Dict, List, Optional, Tuple, Union
import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import warnings

from myopacus.accountants import create_accountant, create_accountant_fedlean, IAccountant
from myopacus.accountants.utils import get_noise_multiplier
from myopacus.data_loader import DPDataLoader, PersonalizedDPDataLoader, switch_generator
from myopacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from myopacus.grad_sample import (
    AbstractGradSampleModule,
    GradSampleModule,
    get_gsm_class,
    wrap_model,
)
from myopacus.optimizers import DPOptimizer, get_optimizer_class
from myopacus.schedulers import _GradClipScheduler, _NoiseScheduler
from myopacus.validators.module_validator import ModuleValidator
from myopacus.utils.module_utils import trainable_parameters
from myopacus.utils.batch_memory_manager import BatchMemoryManager
from myopacus.accountants.rpdp_utils import PrivCostEstimator

def forbid_accumulation_hook(
    module: AbstractGradSampleModule,
    _grad_input: torch.Tensor,
    _grad_output: torch.Tensor,
):
    """
    Model hook that detects repetitive forward/backward passes between optimizer steps.

    This is a backward hook that will be wrapped around the whole model using
    `register_backward_hook`. We wish to detect a case where:
        -  `optimizer.zero_grad()` is not called before the backward pass; and
        -  `p.grad_sample` was updated in a *previous* iteration.

    To do so, we attach a backward hook to the model that runs *after* the computation
    of `grad_sample` for the current step. We compute the number of accumulated iterations
    like on `optimizers/optimizer.py` and check whether it's strictly larger than one.

    Args:
        module: input module
        _grad_input: module input gradient (not used here)
        _grad_output: module output gradient (not used here)

    Raises:
        ValueError
            If the hook detected multiple forward/backward passes between optimizer steps

    """
    if not module.training:
        return

    for _, p in trainable_parameters(module):
        if p.grad_sample is not None:
            if isinstance(p.grad_sample, torch.Tensor):
                accumulated_iterations = 1
            elif isinstance(p.grad_sample, list):
                accumulated_iterations = len(p.grad_sample)

            if accumulated_iterations > 1:
                raise ValueError(
                    "Poisson sampling is not compatible with grad accumulation. "
                    "You need to call optimizer.step() after every forward/backward pass "
                    "or consider using BatchMemoryManager"
                )


class PrivacyEngine:
    """
    Example:
        >>> dataloader = demo_dataloader
        >>> model = MyCustomModel()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        >>> privacy_engine = PrivacyEngine()
        >>>
        >>> model, optimizer, dataloader = privacy_engine.make_private(
        ...    module=model,
        ...    optimizer=optimizer,
        ...    data_loader=dataloader,
        ...    noise_multiplier=1.0,
        ...    max_grad_norm=1.0,
        ... )
        >>> # continue training as normal
    """

    def __init__(
        self, *, 
        accountant: str = "rdp",
        n_clients: int = 1,
        secure_mode: bool = False,
        **kwargs
    ):
        if n_clients == 1:
            self.accountant = create_accountant(mechanism=accountant)
        else:
            self.accountant = create_accountant_fedlean(mechanism=accountant, n_clients=n_clients)

        self.secure_mode = secure_mode
        self.secure_rng = None
        # self.dataset = None  # only used to detect switching to a different dataset
        # if self.secure_mode:
        #     try:
        #         import torchcsprng as csprng
        #     except ImportError as e:
        #         msg = (
        #             "To use secure RNG, you must install the torchcsprng package! "
        #             "Check out the instructions here: https://github.com/pytorch/csprng#installation"
        #         )
        #         raise ImportError(msg) from e

        #     self.secure_rng = csprng.create_random_device_generator("/dev/urandom")
        # else:
        #     warnings.warn(
        #         "Secure RNG turned off. This is perfectly fine for experimentation as it allows "
        #         "for much faster training performance, but remember to turn it on and retrain "
        #         "one last time before production with ``secure_mode`` turned on."
        #     )

    def _prepare_optimizer(
        self,
        optimizer: optim.Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode="hooks",
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optim_class = get_optimizer_class(
            clipping=clipping,
            distributed=distributed,
            grad_sample_mode=grad_sample_mode,
        )

        return optim_class(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
        )

    def _prepare_data_loader(
        self,
        data_loader: DataLoader,
        *,
        poisson_sampling: bool,
        distributed: bool,
        accountant: IAccountant = None
    ) -> DataLoader:

        accountant = self.accountant if accountant is None else accountant
        if poisson_sampling:
            DataLoader_cls = DPDataLoader
            inputs = {
                "data_loader": data_loader,
                "generator": self.secure_rng,
                "distributed": distributed
            }
            if (accountant.mechanism() == "fed_rdp"):
                inputs["steps"] = 1
                inputs["sample_rate"] = accountant.sample_rate
                if (not isinstance(accountant.sample_rate, float)):
                    DataLoader_cls = PersonalizedDPDataLoader

            return DataLoader_cls.from_data_loader(**inputs)
            
            # # fed_rpdp
            # if (accountant.mechanism() == "fed_rdp") and (not isinstance(accountant.sample_rate, float)):
            #     return PersonalizedDPDataLoader.from_data_loader(
            #         data_loader,
            #         per_record_sample_rates=accountant.sample_rate,
            #         generator=self.secure_rng, 
            #         distributed=distributed)
            # # feddp
            # elif accountant.mechanism() == "fed_rdp":
            #     return DPDataLoader.from_data_loader(
            #         data_loader, 
            #         sample_rate=accountant.sample_rate,
            #         generator=self.secure_rng,
            #         steps=1,
            #         distributed=distributed)
            
            # return DPDataLoader.from_data_loader(
            #     data_loader, generator=self.secure_rng,
            #     distributed=distributed)
        
        elif self.secure_mode:
            return switch_generator(data_loader=data_loader, generator=self.secure_rng)
        else:
            return data_loader

    def _prepare_model(
        self,
        module: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        grad_sample_mode: str = "hooks",
    ) -> AbstractGradSampleModule:
        # Ideally, validation should have been taken care of by calling
        # `get_compatible_module()`
        self.validate(module=module, optimizer=None, data_loader=None)

        # wrap
        if isinstance(module, AbstractGradSampleModule):
            if (
                module.batch_first != batch_first
                or module.loss_reduction != loss_reduction
                or type(module) != get_gsm_class(grad_sample_mode)
            ):
                raise ValueError(
                    f"Pre-existing GradSampleModule doesn't match new arguments."
                    f"Got: module.batch_first: {module.batch_first}, module.loss_reduction: {module.loss_reduction}, type(module): {type(module)}"
                    f"Requested: batch_first:{batch_first}, loss_reduction: {loss_reduction}, grad_sample_mode: {grad_sample_mode} "
                    f"Please pass vanilla nn.Module instead"
                )
            return module
        else:
            return wrap_model(
                module,
                grad_sample_mode=grad_sample_mode,
                batch_first=batch_first,
                loss_reduction=loss_reduction,
            )

    def is_compatible(
        self,
        *,
        module: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: Optional[DataLoader],
    ) -> bool:
        """
        Check if task components are compatible with DP.

        Args:
            module: module to be checked
            optimizer: optimizer to be checked
            data_loader: data_loader to be checked

        Returns:
            ``True`` if compatible, ``False`` otherwise
        """
        return ModuleValidator.is_valid(module)

    def validate(
        self,
        *,
        module: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: Optional[DataLoader],
    ):
        """
        Validate that task components are compatible with DP.
        Same as ``is_compatible()``, but raises error instead of returning bool.

        Args:
            module: module to be checked
            optimizer: optimizer to be checked
            data_loader: data_loader to be checked

        Raises:
            UnsupportedModuleError
                If one or more modules found to be incompatible
        """
        ModuleValidator.validate(module, strict=True)

    @classmethod
    def get_compatible_module(cls, module: nn.Module) -> nn.Module:
        """
        Return a privacy engine compatible module. Also validates the module after
        running registered fixes.

        Args:
            module: module to be modified

        Returns:
            Module with some submodules replaced for their deep copies or
            close equivalents.
            See :class:`~opacus.validators.module_validator.ModuleValidator` for
            more details
        """
        module = ModuleValidator.fix(module)
        ModuleValidator.validate(module, strict=True)
        return module

    def make_private(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        noise_multiplier: float = None,
        max_grad_norm: Union[float, List[float]] = None,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
    ) -> Tuple[GradSampleModule, DPOptimizer, DataLoader]:
        """
        Add privacy-related responsibilities to the main PyTorch training objects:
        model, optimizer, and the data loader.

        All of the returned objects act just like their non-private counterparts
        passed as arguments, but with added DP tasks.

        - Model is wrapped to also compute per sample gradients.
        - Optimizer is now responsible for gradient clipping and adding noise to the gradients.
        - DataLoader is updated to perform Poisson sampling.

        Notes:
            Using any other models, optimizers, or data sources during training
            will invalidate stated privacy guarantees.

        Args:
            module: PyTorch module to be used for training
            optimizer: Optimizer to be used for training
            data_loader: DataLoader to be used for training
            noise_multiplier: The ratio of the standard deviation of the Gaussian noise to
                the L2-sensitivity of the function to which the noise is added
                (How much noise to add)
            max_grad_norm: The maximum norm of the per-sample gradients. Any gradient with norm
                higher than this will be clipped to this value.
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            poisson_sampling: ``True`` if you want to use standard sampling required
                for DP guarantees. Setting ``False`` will leave provided data_loader
                unchanged. Technically this doesn't fit the assumptions made by
                privacy accounting mechanism, but it can be a good approximation when
                using Poisson sampling is unfeasible.
            clipping: Per sample gradient clipping mechanism ("flat" or "per_layer" or "adaptive").
                Flat clipping calculates the norm of the entire gradient over
                all parameters, per layer clipping sets individual norms for
                every parameter tensor, and adaptive clipping updates clipping bound per iteration.
                Flat clipping is usually preferred, but using per layer clipping in combination
                with distributed training can provide notable performance gains.
            noise_generator: torch.Generator() object used as a source of randomness for
                the noise
            grad_sample_mode: mode for computing per sample gradients. Determines the
                implementation class for the wrapped ``module``. See
                :class:`~opacus.grad_sample.gsm_base.AbstractGradSampleModule` for more
                details

        Returns:
            Tuple of (model, optimizer, data_loader).

            Model is a wrapper around the original model that also computes per sample
                gradients
            Optimizer is a wrapper around the original optimizer that also does
             gradient clipping and noise addition to the gradients
            DataLoader is a brand new DataLoader object, constructed to behave as
                equivalent to the original data loader, possibly with updated
                sampling mechanism. Points to the same dataset object.
        """
        if noise_generator and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        # compare module parameter with optimizer parameters
        model_parameters = set(module.parameters())
        for p in chain.from_iterable(
            [param_group["params"] for param_group in optimizer.param_groups]
        ):
            if p not in model_parameters:
                raise ValueError(
                    "Module parameters are different than optimizer Parameters"
                )

        distributed = isinstance(module, (DPDDP, DDP))

        module = self._prepare_model(
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            grad_sample_mode=grad_sample_mode,
        )
        if poisson_sampling:
            module.forbid_grad_accumulation()

        data_loader = self._prepare_data_loader(
            data_loader, distributed=distributed, poisson_sampling=poisson_sampling
        )
        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)
        print("---> expected_batch_size: ", expected_batch_size)
        
        # expected_batch_size is the *per worker* batch size
        if distributed:
            world_size = torch.distributed.get_world_size()
            expected_batch_size /= world_size

        optimizer = self._prepare_optimizer(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
        )
        optimizer.attach_step_hook(
            self.accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
        )

        return module, optimizer, data_loader

    def make_private_with_epsilon(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        **kwargs,
    ):
        """
        Version of :meth:`~opacus.privacy_engine.PrivacyEngine.make_private`,
        that calculates privacy parameters based on a given privacy budget.

        For the full documentation see
        :meth:`~opacus.privacy_engine.PrivacyEngine.make_private`

        Args:
            module: PyTorch module to be used for training
            optimizer: Optimizer to be used for training
            data_loader: DataLoader to be used for training
            max_grad_norm: The maximum norm of the per-sample gradients. Any gradient with norm
                higher than this will be clipped to this value.
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            poisson_sampling: ``True`` if you want to use standard sampling required
                for DP guarantees. Setting ``False`` will leave provided data_loader
                unchanged. Technically this doesn't fit the assumptions made by
                privacy accounting mechanism, but it can be a good approximation when
                using Poisson sampling is unfeasible.
            clipping: Per sample gradient clipping mechanism ("flat" or "per_layer" or "adaptive").
                Flat clipping calculates the norm of the entire gradient over
                all parameters, per layer clipping sets individual norms for
                every parameter tensor, and adaptive clipping updates clipping bound per iteration.
                Flat clipping is usually preferred, but using per layer clipping in combination
                with distributed training can provide notable performance gains.
            noise_generator: torch.Generator() object used as a source of randomness for
                the noise
            grad_sample_mode: mode for computing per sample gradients. Determines the
                implementation class for the wrapped ``module``. See
                :class:`~opacus.grad_sample.gsm_base.AbstractGradSampleModule` for more
                details

        Returns:
            Tuple of (model, optimizer, data_loader).

            Model is a wrapper around the original model that also computes per sample
                gradients
            Optimizer is a wrapper around the original optimizer that also does
                gradient clipping and noise addition to the gradients
            DataLoader is a brand new DataLoader object, constructed to behave as
                equivalent to the original data loader, possibly with updated
                sampling mechanism. Points to the same dataset object.
        """
        sample_rate = 1 / len(data_loader)
        if len(self.accountant) > 0:
            warnings.warn(
                "You're calling make_private_with_epsilon with non-zero privacy budget "
                "already spent. Returned noise_multiplier assumes zero starting point, "
                "so your overall privacy budget will be higher."
            )

        return self.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                epochs=epochs,
                accountant=self.accountant.mechanism(),
                **kwargs,
            ),
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            grad_sample_mode=grad_sample_mode,
            poisson_sampling=poisson_sampling,
            clipping=clipping,
        )

    def get_epsilon(self,
            target_delta: float = 1e-5):
        """
        Computes the (epsilon, delta) privacy budget spent so far.

        Args:
            delta: The target delta.

        Returns:
            Privacy budget (epsilon) expended so far.
        """
        return self.accountant.get_epsilon(delta=target_delta)
    
    def get_epsilon_by_id(self,
            id: Optional[int] = None, 
            target_delta: float = 1e-5):
        """
        Computes the (epsilon, delta) privacy budget of a specific sample spent so far.

        Args:
            id: The sample index.
        """
        return self.accountant.get_epsilon_by_id(id=id, delta=target_delta)

    def save_checkpoint(
        self,
        *,
        path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        module: GradSampleModule,
        optimizer: Optional[DPOptimizer] = None,
        noise_scheduler: Optional[_NoiseScheduler] = None,
        grad_clip_scheduler: Optional[_GradClipScheduler] = None,
        checkpoint_dict: Optional[Dict[str, Any]] = None,
        module_state_dict_kwargs: Optional[Dict[str, Any]] = None,
        torch_save_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Saves the state_dict of module, optimizer, and accountant at path.
        Args:
            path: Path to save the state dict objects.
            module: GradSampleModule to save; wrapped module's state_dict is saved.
            optimizer: DPOptimizer to save; wrapped optimizer's state_dict is saved.
            noise_scheduler: _NoiseScheduler whose state we should save.
            grad_clip_scheduler: _GradClipScheduler whose state we should save.
            checkpoint_dict: Dict[str, Any]; an already-filled checkpoint dict.
            module_state_dict_kwargs: dict of kwargs to pass to ``module.state_dict()``
            torch_save_kwargs: dict of kwargs to pass to ``torch.save()``

        """
        checkpoint_dict = checkpoint_dict or {}
        checkpoint_dict["module_state_dict"] = module.state_dict(
            **(module_state_dict_kwargs or {})
        )
        checkpoint_dict["privacy_accountant_state_dict"] = self.accountant.state_dict()
        if optimizer is not None:
            checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()
        if noise_scheduler is not None:
            checkpoint_dict["noise_scheduler_state_dict"] = noise_scheduler.state_dict()
        if grad_clip_scheduler is not None:
            checkpoint_dict[
                "grad_clip_scheduler_state_dict"
            ] = grad_clip_scheduler.state_dict()

        torch.save(checkpoint_dict, path, **(torch_save_kwargs or {}))

    def load_checkpoint(
        self,
        *,
        path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        module: GradSampleModule,
        optimizer: Optional[DPOptimizer] = None,
        noise_scheduler: Optional[_NoiseScheduler] = None,
        grad_clip_scheduler: Optional[_GradClipScheduler] = None,
        module_load_dict_kwargs: Optional[Dict[str, Any]] = None,
        torch_load_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        checkpoint = torch.load(path, **(torch_load_kwargs or {}))
        module.load_state_dict(
            checkpoint["module_state_dict"], **(module_load_dict_kwargs or {})
        )
        self.accountant.load_state_dict(checkpoint["privacy_accountant_state_dict"])

        optimizer_state_dict = checkpoint.pop("optimizer_state_dict", {})
        if optimizer is not None and len(optimizer_state_dict) > 0:
            optimizer.load_state_dict(optimizer_state_dict)
        elif (optimizer is not None) ^ (len(optimizer_state_dict) > 0):
            # warn if only one of them is available
            warnings.warn(
                f"optimizer_state_dict has {len(optimizer_state_dict)} items"
                f" but optimizer is {'' if optimizer else 'not'} provided."
            )

        noise_scheduler_state_dict = checkpoint.pop("noise_scheduler_state_dict", {})
        if noise_scheduler is not None and len(noise_scheduler_state_dict) > 0:
            noise_scheduler.load_state_dict(noise_scheduler_state_dict)

        grad_clip_scheduler_state_dict = checkpoint.pop(
            "grad_clip_scheduler_state_dict", {}
        )
        if grad_clip_scheduler is not None and len(grad_clip_scheduler_state_dict) > 0:
            grad_clip_scheduler.load_state_dict(grad_clip_scheduler_state_dict)

        return checkpoint

# ===================
    def make_pers_private(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks"
    ) -> Tuple[GradSampleModule, DPOptimizer, DataLoader]:
        
        if noise_generator and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        # compare module parameter with optimizer parameters
        model_parameters = set(module.parameters())
        for p in chain.from_iterable(
            [param_group["params"] for param_group in optimizer.param_groups]
        ):
            if p not in model_parameters:
                raise ValueError(
                    "Module parameters are different than optimizer Parameters"
                )

        distributed = isinstance(module, (DPDDP, DDP))
        module = self._prepare_model(
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            grad_sample_mode=grad_sample_mode,
        )
        if poisson_sampling:
            module.register_full_backward_hook(forbid_accumulation_hook)

        data_loader = self._prepare_data_loader(
            data_loader, distributed=distributed, poisson_sampling=poisson_sampling
        )

        expected_batch_size = max(1, int(sum(self.accountant.sample_rate)))
        print("expected_batch_size: ", expected_batch_size)

        # expected_batch_size is the *per worker* batch size
        if distributed:
            world_size = torch.distributed.get_world_size()
            expected_batch_size /= world_size
        
        optimizer = self._prepare_optimizer(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
        )
        
        optimizer.attach_step_hook(
            self.accountant.get_optimizer_hook_fn()
        )

        if self.max_physical_batch_size > 0:
            with BatchMemoryManager(data_loader=data_loader,
                    max_physical_batch_size=self.max_physical_batch_size,
                    optimizer=optimizer) as memory_safe_dl:
                data_loader = memory_safe_dl

        return module, optimizer, data_loader
    
    def make_private_with_personalization(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        target_epsilons: List[float],
        target_delta: float,
        num_steps: int,
        max_grad_norm: Union[float, List[float]],
        noise_multiplier: float = None,
        max_epsilon: float = None,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        max_physical_batch_size: int = 0,
        **kwargs
    ):  
        assert self.accountant.mechanism() == "fed_rdp", f"ERROR: the type of privacy accountant is {self.accountant.mechanism()}."
        if len(self.accountant) > 0:
            warnings.warn(
                "You're calling make_private_with_personalization with non-zero privacy budget "
                "already spent. Returned noise_multiplier assumes zero starting point, "
                "so your overall privacy budget will be higher."
            )

        assert (noise_multiplier is None) != (max_epsilon is None), f"ERROR: noise_multiplier is {noise_multiplier} and max_epsilon is {max_epsilon}."
        if noise_multiplier is None:
            # estimate sample_rates of all points
            self.noise_multiplier = get_noise_multiplier(
                target_epsilon=max_epsilon,
                target_delta=target_delta,
                sample_rate=1.0,
                steps=num_steps,
                accountant="fed_rdp"
            )
        else:
            self.noise_multiplier = noise_multiplier
        print("noise_multiplier : ", self.noise_multiplier)
        
        pce = PrivCostEstimator(
            noise_multiplier = self.noise_multiplier, 
            steps = num_steps,
            delta = target_delta
        )
        curve_fn = pce.get_sample_rate_estimator()
        sample_rate = [curve_fn(budget) for budget in target_epsilons]
        self.accountant.init(sample_rate = sample_rate, 
                             noise_multiplier = self.noise_multiplier)
        self.max_physical_batch_size = max_physical_batch_size
        
        return self.make_pers_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            grad_sample_mode=grad_sample_mode,
            poisson_sampling=poisson_sampling,
            clipping=clipping,
        )
    
    def make_private_with_fedrpdp(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        accountant: IAccountant,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        **kwargs
    ):  
        if noise_generator and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        # compare module parameter with optimizer parameters
        model_parameters = set(module.parameters())
        for p in chain.from_iterable(
            [param_group["params"] for param_group in optimizer.param_groups]
        ):
            if p not in model_parameters:
                raise ValueError(
                    "Module parameters are different than optimizer Parameters"
                )

        distributed = isinstance(module, (DPDDP, DDP))
        module = self._prepare_model(
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            grad_sample_mode=grad_sample_mode,
        )
        if poisson_sampling:
            module.register_full_backward_hook(forbid_accumulation_hook)

        data_loader = self._prepare_data_loader(
            data_loader, 
            distributed=distributed, 
            poisson_sampling=poisson_sampling, 
            accountant=accountant
        )
        
        assert (not isinstance(accountant.sample_rate, float))
        expected_batch_size = max(1, int(sum(accountant.sample_rate)))
        print(f"sample_rate: (min) {round(min(accountant.sample_rate), 2)}, (max) {round(max(accountant.sample_rate), 2)}, expected_batch_size: {expected_batch_size}")

        optimizer = self._prepare_optimizer(
            optimizer,
            noise_multiplier=self.default_noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
        )
        optimizer.attach_step_hook(
            accountant.get_optimizer_hook_fn()
        )

        if self.max_physical_batch_size > 0:
            with BatchMemoryManager(data_loader=data_loader,
                    max_physical_batch_size=self.max_physical_batch_size,
                    optimizer=optimizer) as memory_safe_dl:
                data_loader = memory_safe_dl

        return module, optimizer, data_loader, accountant

    def prepare_fedrpdp(
        self,
        num_steps: int,
        num_rounds: int,
        client_rate: float,
        target_epsilons: List[float],
        target_delta: float,
        max_epsilon: float,
        max_grad_norm: float,
        max_physical_batch_size: int,
        **kwargs
    ):  
        
        assert (self.accountant.mechanism() == "idp") and (len(self.accountant.accountants)==len(target_epsilons)), \
            f"Type of accountant must be `idp` (but here is {self.accountant.mechanism()}) and the sizes of both self.accountant.accountants and target_epsilons "\
            f"must be equal to the total number of clients (but here are {len(self.accountant.accountants)} and {len(target_epsilons)})."
        
        # estimate sample_rates of all points
        self.default_noise_multiplier = get_noise_multiplier(
            target_epsilon=max_epsilon,
            target_delta=target_delta,
            sample_rate=1.0,
            client_rate=client_rate,
            steps=num_steps,
            rounds=num_rounds,
            accountant="fed_rdp"
        )
        print("max_epsilon : ", max_epsilon, " noise_multiplier : ", self.default_noise_multiplier)
        pce = PrivCostEstimator(
            noise_multiplier = self.default_noise_multiplier, 
            steps = num_steps, 
            rounds = num_rounds, 
            client_rate = client_rate,
            delta = target_delta
        )
        curve_fn = pce.get_sample_rate_estimator()
        self.target_epsilons = target_epsilons
        self.target_delta = target_delta

        for _, (budgets, acct) in enumerate(list(zip(self.target_epsilons, self.accountant.accountants))):
            per_client_sample_rates = [curve_fn(budget) for budget in budgets]
            acct.init(noise_multiplier = self.default_noise_multiplier,
                      sample_rate = per_client_sample_rates, 
                      client_rate = client_rate,
                      steps = num_steps,
                      rounds = num_rounds
            )
        self.max_grad_norm = max_grad_norm
        self.max_physical_batch_size = max_physical_batch_size

    def make_private_with_feddp(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        accountant: IAccountant,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator = None,
        grad_sample_mode: str = "hooks",
        **kwargs
    ):  
        # compare module parameter with optimizer parameters
        model_parameters = set(module.parameters())
        for p in chain.from_iterable(
            [param_group["params"] for param_group in optimizer.param_groups]
        ):
            if p not in model_parameters:
                raise ValueError(
                    "Module parameters are different than optimizer Parameters"
                )

        distributed = isinstance(module, (DPDDP, DDP))
        module = self._prepare_model(
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            grad_sample_mode=grad_sample_mode,
        )
        if poisson_sampling:
            module.register_full_backward_hook(forbid_accumulation_hook)

        data_loader = self._prepare_data_loader(
            data_loader, distributed=distributed, poisson_sampling=poisson_sampling, accountant=accountant
        )
        
        assert accountant.mechanism() == "fed_rdp", "The type of the accountant mechanism must be `FedRDPAccountant` when the `feddp` mode is enabled."

        expected_batch_size = int(len(data_loader.dataset) * accountant.sample_rate)

        print(f"sample_rate: {round(accountant.sample_rate, 2)}, expected_batch_size: {expected_batch_size}")

        optimizer = self._prepare_optimizer(
            optimizer,
            noise_multiplier=self.default_noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
        )
        optimizer.attach_step_hook(
            accountant.get_optimizer_hook_fn()
        )

        if self.max_physical_batch_size > 0:
            with BatchMemoryManager(data_loader=data_loader,
                    max_physical_batch_size=self.max_physical_batch_size,
                    optimizer=optimizer) as memory_safe_dl:
                data_loader = memory_safe_dl

        return module, optimizer, data_loader, accountant


    def prepare_feddp(
        self,
        num_steps: int,
        num_rounds: int,
        sample_rate: float,
        client_rate: float,
        target_epsilon: float,
        target_delta: float,
        max_grad_norm: float,
        max_physical_batch_size: int,
        **kwargs
    ):  
        assert (self.accountant.mechanism() == "idp"), \
            "Type of accountant must be `IndividualAccountant`."
        
        print("default_sample_rate:", sample_rate)
        self.default_noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            client_rate=client_rate,
            steps=num_steps,
            rounds=num_rounds,
            accountant="fed_rdp"
        )
        print("noise_multiplier : ", self.default_noise_multiplier)
        for acct in self.accountant.accountants:
            if acct.mechanism() == "fed_rdp":
                acct.init(noise_multiplier = self.default_noise_multiplier,
                          sample_rate = sample_rate, 
                          client_rate = client_rate,
                          steps = num_steps,
                          rounds = num_rounds)
            else:
                raise RuntimeError("the accountant must be `fed_rdp`.")

        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.max_physical_batch_size = max_physical_batch_size
