from .common import (
    Optimizer,
    FedClass,
    RawClass,
)
# from .dataset import FedCifar10, Cifar10Raw
from .loss import BaselineLoss
from .metric import metric
from .model import ResNet18

BaselineModel = ResNet18()