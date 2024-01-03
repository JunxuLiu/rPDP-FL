
from typing import OrderedDict
import torch
from torch.optim import SGD, Adagrad, Adam, RMSprop

def make_optimizer_class(cls):
    class ClipOptimizerClass(cls):
        def __init__(self, l2_norm_clip, minibatch_size, microbatch_size, *args, **kwargs):
            super(ClipOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size

            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def zero_microbatch_grad(self):
            super(ClipOptimizerClass, self).zero_grad()

        def microbatch_clipping(self):
            total_norm_pre = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm_pre += param.grad.data.norm(2).item() ** 2.
            total_norm_pre = total_norm_pre ** .5
            # print(f"l2_norm_clip: {self.l2_norm_clip}, norm: {total_norm_pre:5.4f}")

            clip_coef = min(self.l2_norm_clip / (total_norm_pre + 1e-6), 1.)
            # clip_coef = 1.0
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul_(clip_coef))
            # print(f"pre: {total_norm_pre:5.4f}, clip_coef: {clip_coef:5.4f}")

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, *args, **kwargs):
            # assert minibatch_size is not None, "the type of `minibatch_size` should be int, not NoneType."
            total_norm = 0.
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        # param.grad.data.add_(torch.zeros_like(param.grad.data)) # add noise (all zeros)
                        # param.grad.data.mul_(self.microbatch_size / minibatch_size) # scale
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5
            super(ClipOptimizerClass, self).step(*args, **kwargs)
            return total_norm

    return ClipOptimizerClass

# DPAdam = make_optimizer_class(Adam)
# DPAdagrad = make_optimizer_class(Adagrad)
# DPSGD = make_optimizer_class(SGD)
# DPRMSprop = make_optimizer_class(RMSprop)

