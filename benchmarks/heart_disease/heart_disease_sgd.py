import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append('../..')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from fedrpdp.datasets.fed_heart_disease import (
    BaselineModel,
    BaselineLoss,
    FedHeartDisease,
    NUM_CLIENTS,
    metric
)

from fedrpdp.strategies import FedAvg
from fedrpdp.utils import evaluate_model_on_tests

parser = argparse.ArgumentParser()
parser.add_argument("--log", action="store_true",
    help="Whether to activate tensorboard logging or not default to no logging")
parser.add_argument("--log-period", type=int, default=10,
    help="The period in batches for the logging of metric and loss")
parser.add_argument("--gpu-id", type=int, default=0,
    help="Index of the GPU device.")
parser.add_argument("--num-workers", type=int, default=20,
    help="How many workers to use for the batching.")
parser.add_argument("--dataset", type=str, default='heart_disease')
parser.add_argument("--method", type=str, default='privacy_free')
parser.add_argument("--noise-multiplier", type=float, default=10.0)
parser.add_argument("--max-grad-norm", type=float, default=1.1)
parser.add_argument("--num-steps", type=int, default=1)
parser.add_argument("--num-rounds", type=int, default=50)
parser.add_argument("--delta", type=float, default=1e-5)
parser.add_argument("--lr", type=float, default=0.05)

args = parser.parse_args()
device = f'cuda:{args.gpu_id}'

BATCH_SIZE = 4
LR = 0.01
device = "cuda:0"
train_pooled = DataLoader(
    FedHeartDisease(train=True, pooled=True),
    batch_size=4,
    shuffle=True,
)
test_pooled = DataLoader(
    FedHeartDisease(train=False, pooled=True),
    batch_size=4,
    shuffle=False,
)

model = BaselineModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
bloss = BaselineLoss()

# ======== Start Training ==========

for epoch in range(50):

    for i, (data, target) in enumerate(iter(train_pooled)):
        model.train()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # compute train acc
        train_metric = metric(target.detach().cpu().numpy(), output.detach().cpu().numpy())

        # compute train loss
        loss = bloss(output, target)
        train_loss = loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        total_correct, total_points = 0, 0
        for i, (data, target) in enumerate(iter(train_pooled)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            y_true = target.detach().cpu().numpy().astype("uint8")
            correct = sum((output.detach().cpu().numpy() > 0.5) == y_true)[0]
            total_correct += correct
            total_points += len(target)
            
            test_loss = bloss(output, target).item()

    print(f"Epoch={epoch}, perf={total_correct} / {total_points} ({total_correct/total_points:.4f}%)")
