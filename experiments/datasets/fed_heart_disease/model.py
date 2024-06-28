import torch
import torch.nn.functional as F

class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def name(self):
        return "LinearRegression"

class DNNModel(torch.nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super(DNNModel, self).__init__()
        """ 2-layer DNN, 151 params """
        self.fc1 = torch.nn.Linear(input_dim, 10)
        self.fc2 = torch.nn.Linear(10, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))