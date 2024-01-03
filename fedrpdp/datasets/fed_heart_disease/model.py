import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super(BaselineModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
