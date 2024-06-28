import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from myopacus.validators import ModuleValidator

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = resnet18(num_classes=10) # ResNet-18 from scratch
        self.model = ModuleValidator.fix(self.model)
        ModuleValidator.validate(self.model, strict=False)

    def forward(self, x):
        return self.model(x)

    def name(self):
        return "ResNet18"
