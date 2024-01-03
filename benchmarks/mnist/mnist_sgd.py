import argparse
import numpy as np
import math
import os
import sys
sys.path.append('../..')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from fedrpdp.datasets.fed_mnist import (
    BaselineModel,
    BaselineLoss,
    metric
)
from utils_sgd import train, test
from fedrpdp.utils.rpdp_utils import (
    get_noise_multiplier,
    get_per_sample_rates, 
    FilterManager, 
    rPDPManager, 
    MultiLevels,
)

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
parser.add_argument("--noise-multiplier", type=float, default=5.37)
parser.add_argument("--max-grad-norm", type=float, default=10.0)
parser.add_argument("--delta", type=float, default=1e-5)
parser.add_argument("--lr", type=float, default=0.1)

args = parser.parse_args()
device = f'cuda:{args.gpu_id}'
disable_dp = False
if args.method == 'privacy_free':
    disable_dp = True
# elif args.method == 'filter':
#     budgets = np.array([10000]*40000 + [20000]*10000 + [3000]*10000)
# else:
#     budgets = np.array([0.15]*40000 + [0.3]*10000 + [0.45]*10000)

if args.log:
    if not disable_dp:
        writer = SummaryWriter(log_dir=f"./runs/{args.method}")
    else:
        writer = SummaryWriter(log_dir=f"./runs/privacy_free")

project_abspath = os.path.dirname(os.getcwd())
DATA_ROOT = '/data/privacyGroup/liujunxu/datasets/{}'.format(args.dataset)

train_data = MNIST(DATA_ROOT, train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
test_data = MNIST(DATA_ROOT, train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))

train_loader = DataLoader(train_data,
    batch_size=len(train_data), # use all data points
    shuffle=False,
    num_workers=args.num_workers_torch,
)
test_loader = DataLoader(test_data,
    batch_size=len(test_data),
    shuffle=False,
    num_workers=args.num_workers_torch,
)

MIN_EPSILON, MAX_EPSILON, VALID_EPSILON = 0.1, 10.0, 2.0

model = BaselineModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
criterion = BaselineLoss()

if disable_dp:
    pass
elif args.method == 'filter':
    # compute noise_multiplier
    args.noise_multiplier = get_noise_multiplier(
        target_epsilon=MAX_EPSILON, 
        num_updates=args.max_epochs
    )
    print('noise_multiplier: ', args.noise_multiplier)
    # compute per-sample num_updates
    per_sample_num_updates = MultiLevels(3, [0.7,0.2,0.1], [int(args.max_epochs*0.2), int(args.max_epochs*0.5), int(args.max_epochs*0.8)], len(train_data))
    # compute per-sample norm_sq_budgets
    norm_sq_budgets = [ int(num_update * args.max_grad_norm**2) 
                       for num_update in per_sample_num_updates]

    kwargs = {'module': model, 
        'data_loader': train_loader,
        'noise_multiplier': args.noise_multiplier,
        'max_grad_norm': args.max_grad_norm,
    }
    manager = FilterManager()
    privacy_engine = manager.get_privacy_engine(
        norm_sq_budgets=norm_sq_budgets, **kwargs)
    privacy_engine.attach(optimizer)
else:
    BOUND = lambda budgets: np.array([min(max(x, MIN_EPSILON), MAX_EPSILON) for x in budgets])
    per_sample_epsilons = MultiLevels(3, [0.7,0.2,0.1], [4.34, 7.20, 9.37], len(train_data))
    target_epsilons = BOUND(per_sample_epsilons)
    per_sample_rates = get_per_sample_rates(
        target_epsilon=target_epsilons, 
        noise_multiplier=args.noise_multiplier,
        num_updates=args.max_epochs
    )
    kwargs = {
        'module': model, 
        'data_loader': train_loader,
        'optimizer': optimizer,
        'noise_multiplier': args.noise_multiplier,
        'max_grad_norm': args.max_grad_norm,
    }
    
    if args.method == 'ours':
        manager = rPDPManager(accountant='pers_rdp')
        (
            privacy_engine, 
            model, 
            optimizer, 
            train_loader
        ) = manager.get_privacy_engine( 
            sample_rate=per_sample_rates,
            **kwargs
        )
    else:
        # 每一个client有一个privacy_engine
        # 记录client被选中的轮次、参与训练的steps以及累积的budgets
        if args.method == 'StrongForAll':
            sample_rate = min(per_sample_rates)
        elif args.method == 'WeakForAll':
            sample_rate = max(per_sample_rates)
        elif args.method == 'Dropout':
            valid_indices = np.where(target_epsilons > VALID_EPSILON)[0]
            valid_target_epsilons = target_epsilons[valid_indices]
            train_data = Subset(train_data, valid_indices)
            train_loader = DataLoader(
                train_data,
                num_workers=args.num_workers_torch,
                batch_size=len(train_data),
                shuffle=False,
            )
        
        manager = rPDPManager(accountant='rdp')
        (
            privacy_engine, 
            model, 
            optimizer, 
            train_loader
        ) = manager.get_privacy_engine(
            sample_rate=sample_rate,
            **kwargs
        )
        
# kwargs = {'module': model, 
#           'optimizer': optimizer, 
#           'data_loader': train_loader,
#           'noise_multiplier': args.noise_multiplier,
#           'max_grad_norm': args.max_grad_norm,
#           'target_delta': args.delta,
#           'epochs':args.max_epochs}

# if args.method == 'privacy_free':
#     pass
# elif args.method == 'filter':
#     manager = FilterManager()
#     privacy_engine = manager.get_privacy_engine(method=args.method, budgets=budgets, **kwargs)
#     privacy_engine.attach(optimizer)
# else:
#     per_sample_rates = get_per_sample_rates(budgets=budgets, **kwargs)
#     manager = rPDPManager()
#     privacy_engine, model, optimizer, train_loader = manager.get_privacy_engine(method=args.method, account_type='pers_rdp', per_sample_rates=per_sample_rates, **kwargs)

# ======== Start Training ==========
run_results = []

if args.method == 'filter':
    active_points = []
    running_grad_sq_norms = [torch.Tensor([0]*len(train_data)).to(device)]

import datetime
for epoch in range(1, args.max_epochs + 1):
    if args.method == 'filter':
        num_active_points = np.sum(np.round(running_grad_sq_norms[-1].cpu().numpy(), decimals=3) < np.array(norm_sq_budgets))
        print('num_active_points: ', num_active_points.item())
        active_points.append(num_active_points.item())
        if num_active_points == 0:
            break
        
        start_time = datetime.datetime.now()
        train_loss, train_correct, gradient_norms = train(model, device, train_loader, 
        optimizer, criterion, metric, running_norms=running_grad_sq_norms[-1])
        train_acc = float(train_correct / len(train_data))

        print('duration: ', (datetime.datetime.now() - start_time).total_seconds())
        # update running squared grad norms
        running_grad_sq_norms.append(running_grad_sq_norms[-1] + gradient_norms)

    else:
        train_loss, train_correct = train(model, device, train_loader, optimizer, criterion, metric)
        train_acc = float(train_correct / len(train_data))

    test_loss, test_correct = test(model, device, test_loader, criterion, metric)
    test_acc = float(test_correct / len(test_data))
    run_results.append(test_acc)
    
    if args.log:
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)

    print(f"Epoch: {epoch}")
    if disable_dp:
        print(f"Train Loss: {train_loss:.4f} \t Acc: {100*train_acc:.2f}%")
    elif args.method == 'ours' or args.method == 'filter':
        epsilon_1 = manager.get_epsilon(
            privacy_engine=privacy_engine, 
            delta=args.delta, id=0)
        epsilon_2 = manager.get_epsilon(
            privacy_engine=privacy_engine, 
            delta=args.delta, id=-1)
        print(
            f"Train Loss: {train_loss:.6f} \t Acc: {100*train_acc:.2f}% "
            f"| δ: {args.delta} "
            f"ε1 = {epsilon_1:.2f}, "
            f"ε2 = {epsilon_2:.2f}. "
        )
    else:
        epsilon = manager.get_epsilon(privacy_engine, args.delta)
        print(
            f"Train Loss: {train_loss:.6f} \t Acc: {100*train_acc:.2f}% "
            f"| δ: {args.delta} "
            f"ε = {epsilon:.2f}."
        )
        
    print("Test  Loss: {:.4f} \t Acc: {:.2f}%\n".format(test_loss, 100*test_acc))

if args.log:
    writer.close()