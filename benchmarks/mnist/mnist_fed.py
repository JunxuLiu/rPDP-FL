import argparse
import numpy as np
import math
import os
import sys
sys.path.append('..')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from fedrpdp.datasets.fed_mnist import (
    BaselineModel,
    BaselineLoss,
    metric
)
from utils_sgd import train, test
from fedrpdp.utils.rpdp_utils import get_per_sample_rates, FilterManager, rPDPManager

parser = argparse.ArgumentParser()
parser.add_argument("--log", action="store_true",
    help="Whether to activate tensorboard logging or not default to no logging")
parser.add_argument("--log-period", type=int, default=10,
    help="The period in batches for the logging of metric and loss")
parser.add_argument("--gpu-id", type=int, default=0,
    help="Index of the GPU device.")
parser.add_argument("--num-workers-torch", type=int, default=20,
    help="How many workers to use for the batching.")
parser.add_argument("--dataset", type=str, default='mnist')
parser.add_argument("--method", type=str, default='ours')
parser.add_argument("--max-epochs", type=int, default=100)
parser.add_argument("--noise-multiplier", type=float, default=100.0)
parser.add_argument("--max-grad-norm", type=float, default=10.0)
parser.add_argument("--delta", type=float, default=1e-5)
parser.add_argument("--lr", type=float, default=0.05)

args = parser.parse_args()
device = f'cuda:{args.gpu_id}'
disable_dp = False
if args.method == 'privacy_free':
    disable_dp = True
elif args.method == 'filter':
    budgets = np.array([10000]*40000 + [20000]*10000 + [3000]*10000)
else:
    budgets = np.array([0.15]*40000 + [0.3]*10000 + [0.45]*10000)

if args.log:
    if not disable_dp:
        writer = SummaryWriter(log_dir=f"./runs/{args.dataset}/{args.method}")
    else:
        writer = SummaryWriter(log_dir=f"./runs/{args.dataset}/privacy_free")

project_abspath = os.path.dirname(os.getcwd())
DATA_ROOT = '/data/privacyGroup/liujunxu/datasets/{}'.format(args.dataset)

train_data = MNIST(DATA_ROOT, train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
test_data = MNIST(DATA_ROOT, train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))

kwargs = {"num_workers": args.num_workers_torch, "pin_memory": True}
train_loader = DataLoader(train_data,
    batch_size=len(train_data), # use all data points
    shuffle=False,
    **kwargs,
)
test_loader = DataLoader(test_data,
    batch_size=1024,
    shuffle=True,
    **kwargs,
)

model = BaselineModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
criterion = BaselineLoss()

kwargs = {'module': model, 
          'optimizer': optimizer, 
          'data_loader': train_loader,
          'noise_multiplier': args.noise_multiplier,
          'max_grad_norm': args.max_grad_norm,
          'target_delta': args.delta,
          'epochs':args.max_epochs}
if args.method == 'privacy_free':
    pass
elif args.method == 'filter':
    manager = FilterManager()
    privacy_engine = manager.get_privacy_engine(method=args.method, budgets=budgets, **kwargs)
    privacy_engine.attach(optimizer)
else:
    per_sample_rates = get_per_sample_rates(budgets=budgets, **kwargs)
    manager = rPDPManager()
    privacy_engine, model, optimizer, train_loader = manager.get_privacy_engine(method=args.method, account_type='pers_rdp', per_sample_rates=per_sample_rates, **kwargs)

# ======== Start Training ==========
run_results = []
active_points = []
running_gradient_sq_norms = [torch.Tensor([0]*len(train_data)).to(device)]

for epoch in range(1, args.max_epochs + 1):

    if args.method == 'filter':
        num_active_points = np.sum(np.round(running_gradient_sq_norms[-1].cpu().numpy(), decimals=3) < np.array(budgets))
        print('num_active_points: ', num_active_points.item())
        active_points.append(num_active_points.item())
        if num_active_points == 0:
            break

        train_loss, train_acc, gradient_norms = train(model, device, train_loader, 
        optimizer, criterion, metric, running_norms=running_gradient_sq_norms[-1])
        # update running squared grad norms
        running_gradient_sq_norms.append(running_gradient_sq_norms[-1] + gradient_norms)

    else:
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, metric)

    test_loss, test_acc = test(model, device, test_loader, criterion, metric)
    run_results.append(test_acc)
    
    if args.log:
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)

    if not disable_dp:
        epsilon_1, best_alpha_1 = manager.get_epsilon(privacy_engine, 0, args.delta)
        epsilon_2, best_alpha_2 = manager.get_epsilon(privacy_engine, 40000, args.delta)
        epsilon_3, best_alpha_3 = manager.get_epsilon(privacy_engine, 50000, args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {train_loss:.6f} "
            f"Acc: {train_acc:.6f} "
            f"δ: {args.delta} "
            f"ε1 = {epsilon_1:.2f} for α1 = {best_alpha_1}, "
            f"ε2 = {epsilon_2:.2f} for α2 = {best_alpha_2}, "
            f"ε3 = {epsilon_3:.2f} for α3 = {best_alpha_3}. "
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {train_loss:.4f} \t Acc: {100.0 * test_acc:.2f}")
    
    print("\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(test_loss, 100.0 * test_acc))

    