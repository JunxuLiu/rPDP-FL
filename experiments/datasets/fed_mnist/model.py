import torch
import torch.nn as nn
import torch.nn.functional as F

class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))   # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)   # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))   # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)   # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))     # -> [B, 32]
        x = self.fc2(x)             # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"
    
# def CNN_MNIST(device):
#     model = nn.Sequential(
#                 nn.Conv2d(1, 16, 8, 2),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, 1),
#                 nn.Conv2d(16, 32, 4, 2),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, 1),
#                 Flatten(),
#                 nn.Linear(288, 10),
#                 nn.LogSoftmax(dim=1)
#             ).to(device)
#     return model