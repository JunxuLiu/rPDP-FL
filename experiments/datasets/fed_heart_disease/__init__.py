from .common import (
    Optimizer,
    FedClass,
    RawClass
)
from .dataset import FedHeartDisease, HeartDiseaseRaw
from .loss import BaselineLoss
from .metric import metric
from .model import LinearRegression, DNNModel

BaselineModel = LinearRegression()
# BaselineModel = DNNModel()
