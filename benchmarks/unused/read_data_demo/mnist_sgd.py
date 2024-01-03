# generate 6000 samples
import numpy as np
import os
import sys
sys.path.append('../..')

from torchdp import PrivacyEngine

project_abspath = os.path.dirname(os.getcwd())
print(project_abspath)

DATASET_NAME = 'mnist' # or 'cifar10'
DATA_ROOT = '/data/privacyGroup/liujunxu/datasets/{}'.format(DATASET_NAME)
RES_ROOT = project_abspath + '/results/sgd/{}'.format(DATASET_NAME)
if not os.path.exists(RES_ROOT):
    os.makedirs(RES_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

IMAGENET_MEAN = {'mnist':[0.5], 'cifar10':[0.485, 0.456, 0.406]}
IMAGENET_STD = {'mnist':[0.5], 'cifar10':[0.229, 0.224, 0.225]}
if DATASET_NAME == 'mnist':
    train_data = MNIST(DATA_ROOT,
                    train=True,
                    download=True,
                    transform=Compose([ToTensor(), Normalize(IMAGENET_MEAN[DATASET_NAME], IMAGENET_STD[DATASET_NAME])]))
    test_data = MNIST(DATA_ROOT, 
                  train=False, 
                  download=True, 
                  transform=Compose([ToTensor(), Normalize(IMAGENET_MEAN[DATASET_NAME], IMAGENET_STD[DATASET_NAME])]))

elif DATASET_NAME == 'cifar10':
    train_data = CIFAR10(DATA_ROOT, 
                    train=True, 
                    download=True, 
                    transform=Compose([ToTensor(), Normalize(IMAGENET_MEAN[DATASET_NAME], IMAGENET_STD[DATASET_NAME])]))
    test_data = CIFAR10(DATA_ROOT, 
                  train=False, 
                  download=True, 
                  transform=Compose([ToTensor(), Normalize(IMAGENET_MEAN[DATASET_NAME], IMAGENET_STD[DATASET_NAME])]))
    
kwargs = {"num_workers": 1, "pin_memory": True}
device = 'cuda:2'

# choose 10000 points randomly
# indices = torch.randperm(len(train_data))[:1000] 
# train_data = Subset(train_data, indices) 
train_loader = DataLoader(
    train_data,
    batch_size=60000, # use all data points
    shuffle=False,
    **kwargs,
)
test_loader = DataLoader(
    test_data,
    batch_size=1024,
    shuffle=True,
    **kwargs,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    
from tqdm import tqdm

def train(model, device, train_loader, optimizer, epoch, running_norms, disable_dp=False, delta=1e-5):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    data, target = next(iter(train_loader))
    correct = 0
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)

    # compute train acc
    pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item() 
    
    loss = criterion(output, target)
    loss.backward()
    # gradient_norms is a Tensor with size being equal to `dataset_size`
    gradient_norms = optimizer.step(running_norms)
    gradient_norms_sq = gradient_norms * gradient_norms
    losses.append(loss.item())

    # for data, target in train_loader:
    #     correct = 0
    #     data, target = data.to(device), target.to(device)
    #     optimizer.zero_grad()
    #     output = model(data)

    #     # compute train acc
    #     pred = output.argmax(
    #             dim=1, keepdim=True
    #         )  # get the index of the max log-probability
    #     correct = pred.eq(target.view_as(pred)).sum().item() 

    #     loss = criterion(output, target)
    #     loss.backward()
    #     gradient_norms = optimizer.step(running_norms)
    #     gradient_norms_sq = gradient_norms * gradient_norms
    #     losses.append(loss.item())

    if not disable_dp:
        # Note that we only show the 1st point's cumulative privacy cost.
        epsilon_1, best_alpha_1 = optimizer.privacy_engine.get_privacy_spent(0, delta)
        epsilon_2, best_alpha_2 = optimizer.privacy_engine.get_privacy_spent(40000, delta)
        epsilon_3, best_alpha_3 = optimizer.privacy_engine.get_privacy_spent(50000, delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"Acc: {correct/60000.0:.6f} "
            f"δ: {delta} "
            f"ε1 = {epsilon_1:.2f} for α1 = {best_alpha_1}, "
            f"ε2 = {epsilon_2:.2f} for α2 = {best_alpha_2}, "
            f"ε3 = {epsilon_3:.2f} for α3 = {best_alpha_3}. "
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

    return gradient_norms_sq


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


epochs = 150
budgets = np.array([11200]*40000 + [14400]*10000 + [19800]*10000)
lr = .2
sigma = 170
max_per_sample_grad_norm = 10
delta = 1e-5
disable_dp = False

run_results = []
active_points = []
running_gradient_sq_norms = [0]

model = SampleConvNet().to(device)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
if not disable_dp:
    privacy_engine = PrivacyEngine(
        model,
        batch_size=60000,
        sample_size=len(train_loader.dataset),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 80)),
        noise_multiplier=sigma,
        max_grad_norm=max_per_sample_grad_norm,
        norm_sq_budget = budgets,
        should_clip = True,
    )
    privacy_engine.attach(optimizer)

for epoch in range(1, epochs + 1):
    gradient_norms = train(model, device, train_loader, optimizer, epoch, running_gradient_sq_norms[-1])

    # update running squared grad norms
    running_gradient_sq_norms.append(running_gradient_sq_norms[-1] + gradient_norms)

    # add new test accuracy 
    run_results.append(test(model, device, test_loader))
    num_active_points = np.sum(running_gradient_sq_norms[-1].cpu().numpy() < np.array(budgets))
    active_points.append(num_active_points)
    if num_active_points == 0:
        break
	   
alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
eps_val = min([alpha/2*budgets/(sigma**2 * max_per_sample_grad_norm**2) + np.log(1/delta)/(alpha-1) for alpha in alphas])

