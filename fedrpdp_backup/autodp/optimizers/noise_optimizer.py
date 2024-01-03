
from typing import OrderedDict
import torch
from torch.optim import SGD, Adagrad, Adam, RMSprop

def make_optimizer_class(cls):
    class NoiseOptimizerClass(cls):
        def __init__(self, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):
            super(NoiseOptimizerClass, self).__init__(*args, **kwargs)

            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size

            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def zero_microbatch_grad(self):
            super(NoiseOptimizerClass, self).zero_grad()

        def microbatch_accum(self):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul_(1.0))

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, minibatch_size=None, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.add_(self.noise_multiplier * torch.randn_like(param.grad.data))

            super(NoiseOptimizerClass, self).step(*args, **kwargs)

        def grads_perturbation(self, minibatch_size=None, *args, **kwargs):
            if minibatch_size is None and self.minibatch_size is None:
                raise ValueError('you must set a minibatch_size value when either initializing the class or invoking this method.')
            elif minibatch_size is None:
                minibatch_size = self.minibatch_size
            
            grads = []
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.add_(self.noise_multiplier * torch.randn_like(param.grad.data))
                        param.grad.data.mul_(self.microbatch_size / minibatch_size)
                        grads.append(param.grad.data.clone())
            return grads

        def switch_lr(self, new_lr):
            for group in self.param_groups:
                group['lr'] = new_lr

    return NoiseOptimizerClass

# DPAdam = make_optimizer_class(Adam)
# DPAdagrad = make_optimizer_class(Adagrad)
# DPSGD = make_optimizer_class(SGD)
# DPRMSprop = make_optimizer_class(RMSprop)

