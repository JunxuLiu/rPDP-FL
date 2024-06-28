import torch
from .dataset import SNLIRaw, FedSNLI

Optimizer = torch.optim.AdamW
FedClass = FedSNLI
RawClass = SNLIRaw

