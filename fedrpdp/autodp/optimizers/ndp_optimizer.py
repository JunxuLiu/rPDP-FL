
import torch
from torch.optim import SGD, Adagrad, Adam, RMSprop

def make_optimizer_class(cls):
    class NDPOptimizerClass(cls):
        def __init__(self, minibatch_size, microbatch_size, *args, **kwargs):
            super(NDPOptimizerClass, self).__init__(*args, **kwargs)

            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size

            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def zero_microbatch_grad(self):
            super(NDPOptimizerClass, self).zero_grad()

        def microbatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data)

            return total_norm

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, minibatch_size=None, *args, **kwargs):
            
            if self.minibatch_size is not None:
                minibatch_size = self.minibatch_size
            elif minibatch_size is None:
                raise RuntimeError(r'the param `minibatch_size` is invalid!')

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.mul_(self.microbatch_size / minibatch_size)

            super(NDPOptimizerClass, self).step(*args, **kwargs)

    return NDPOptimizerClass

# DPAdam = make_optimizer_class(Adam)
# DPAdagrad = make_optimizer_class(Adagrad)
# DPSGD = make_optimizer_class(SGD)
# DPRMSprop = make_optimizer_class(RMSprop)

