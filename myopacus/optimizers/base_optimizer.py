
from typing import OrderedDict
import torch
# from torch.optim import SGD, Adagrad, Adam, RMSprop

def make_optimizer_class(cls):
    class BaseOptimizerClass(cls):
        def __init__(self, l2_norm_clip, minibatch_size, microbatch_size, *args, **kwargs):
            super(BaseOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size

            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def zero_microbatch_grad(self):
            super(BaseOptimizerClass, self).zero_grad()

        def microbatch_norm(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5
            
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(1.0))

            return total_norm

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()
            
        def step(self, minibatch_size=None, *args, **kwargs):
            assert minibatch_size is not None, "the type of `minibatch_size` should be int, not NoneType."
            
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.mul_(1 / minibatch_size)
            
            super(BaseOptimizerClass, self).step(*args, **kwargs)

        def switch_lr(self, new_lr):
            for group in self.param_groups:
                group['lr'] = new_lr

    return BaseOptimizerClass

# DPAdam = make_optimizer_class(Adam)
# DPAdagrad = make_optimizer_class(Adagrad)
# DPSGD = make_optimizer_class(SGD)
# DPRMSprop = make_optimizer_class(RMSprop)

