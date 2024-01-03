import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append('../..')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fedrpdp.datasets.fed_heart_disease import (
    BaselineModel,
    BaselineLoss,
    FedHeartDisease,
    # BATCH_SIZE,
    # LR,
    NUM_CLIENTS,
    NUM_EPOCHS_POOLED,
    get_nb_max_rounds,
    metric
)
from conf import (
    check_config,
    get_dataset_args,
    get_results_file,
    get_strategies,
)
from benchmark_utils import (
    ensemble_perf_from_predictions,
    evaluate_model_on_local_and_pooled_tests,
    fill_df_with_xp_results,
    find_xps_in_df,
    get_logfile_name_from_strategy,
    init_data_loaders,
    init_xp_plan,
    set_dataset_specific_config,
    set_seed,
    train_single_centric,
)
from fedrpdp.strategies import FedAvg
from fedrpdp.utils import evaluate_model_on_tests
from fedrpdp.utils.rpdp_utils import MultiLevels, MixGauss, Pareto, BOUNDED_BUDGET_FUNC
from utils_sgd import train, test

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
disable_dp = False
if args.method == 'privacy_free':
    disable_dp = True

GENERATE_EPSILONS = {
    "TwoLevels": lambda n: MultiLevels(2, [0.8,0.2], [0.5, 5.0], n),
    "ThreeLevels": lambda n: MultiLevels(3, [0.7,0.2,0.1], [0.5, 2.0, 5.0], n),
    "BoundedPareto": lambda n: Pareto(2.5, 0.01, 5.0, n), 
    "BoundedMixGauss": lambda n: MixGauss([0.7,0.2,0.1], [(0.5, 0.1), (2.0, 0.5), (5.0, 1.0)], n),
}
MIN_BUDGET, MAX_BUDGET = 0.1, 1.0
BOUND = lambda budgets: np.array([min(max(x, MIN_BUDGET), MAX_BUDGET) for x in budgets])

# ======== Generate Individual Privacy Budgets for All Records ==========
training_dls, test_dls = [], []
target_epsilons = []
for i in range(NUM_CLIENTS):
    train_ds = FedHeartDisease(center=i, train=True, pooled=False)
    if not disable_dp:
        epsilons = BOUND(GENERATE_EPSILONS["ThreeLevels"](len(train_ds)))
        target_epsilons.append(epsilons)
    else:
        target_epsilons.append([None])

    training_dls.append(
        DataLoader(
            train_ds,
            batch_size=len(train_ds),
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=None,
        )
    )
    test_ds = FedHeartDisease(center=i, train=False, pooled=False)
    test_dls.append(
        DataLoader(
            test_ds,
            batch_size=len(test_ds),
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=None,
        )
    )

train_ds = FedHeartDisease(train=True, pooled=True)
test_ds = FedHeartDisease(train=False, pooled=True)
train_pooled = DataLoader(
    train_ds,
    batch_size=len(train_ds),
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=None
)
test_pooled = DataLoader(
    test_ds,
    batch_size=len(test_ds),
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=None
)

model = BaselineModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
criterion = BaselineLoss()

kwargs = {
    "training_dataloaders": training_dls,
    "test_dataloaders": test_dls,
    "model": model,
    "loss": criterion,
    "optimizer_class": torch.optim.SGD,
    "learning_rate": args.lr,
    "num_updates": args.num_steps,
    "nrounds": args.num_rounds,
    "client_sample_rate": 1.0,
}
kwargs_dp = {
    'dp_noise_multiplier': args.noise_multiplier,
    'dp_max_grad_norm': args.max_grad_norm,
    'dp_target_delta': args.delta,
    'dp_target_epsilon': target_epsilons, #
}

# ======== Start Training ==========
s = FedAvg(**kwargs, **kwargs_dp, 
           log=args.log, 
           logdir=f'./runs/fed/{args.method}')

print("FL strategy: FedAvg \t num_updates ", kwargs['num_updates'])

eval_fn = evaluate_model_on_local_and_pooled_tests
model = s.run(eval_fn=eval_fn, device=device, metric=metric, test_pooled=test_pooled)[0]
