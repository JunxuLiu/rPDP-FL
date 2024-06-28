import torch
from .dataset import FedMnist, MnistRaw
Optimizer = torch.optim.Adam
FedClass = FedMnist
RawClass = MnistRaw