import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader as dl
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
sys.path.append('../..')

from fedrpdp import PrivacyEngine
from fedrpdp.datasets.fed_heart_disease import (
    NUM_CLIENTS,
    CLIENT_SAMPLE_RATE,
    BaselineModel,
    BaselineLoss,
    FedHeartDisease,
    metric
)
from fedrpdp.strategies import FedAvg
from fedrpdp.utils.rpdp_utils import (
    get_sample_rate_curve,
    MultiLevels, 
    MixGauss, 
    Pareto,
)

torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, default='./results', help="stored dir of the results")
parser.add_argument("--seed", type=int, default=41, help="random seed")
hargs = parser.parse_args()
if not os.path.exists(hargs.outdir):
    os.mkdir(hargs.outdir)

# FL params
BATCH_SIZE = 16
LR = 0.01
LR_DP = 0.01
num_updates = 50
num_rounds = 15
device = "cuda:2"
bloss = BaselineLoss()

# dp params
grad_norm = 1.0
target_delta = 1e-3

# personalized epsilons: different distributions
GENERATE_EPSILONS = {
    "ThreeLevels": lambda n, params: MultiLevels(3, *params, n),
    "BoundedPareto": lambda n, params: Pareto(*params, n), 
    "BoundedMixGauss": lambda n, params: MixGauss(*params, n),
}
# 改 noise multiplier
SETTINGS = {
    "ThreeLevels": [[[0.7,0.2,0.1], [0.1, 1.0, 5.0]]],
    "BoundedPareto": [[4, 0.1]], 
    "BoundedMixGauss": [[[0.7,0.2,0.1], [(0.1, 0.05), (1.0, 0.1), (5.0, 0.5)]]],
}
MIN_EPSILON, MAX_EPSILON = 0.1, 10.0
BoundedFunc = lambda values: np.array([min(max(x, MIN_EPSILON), MAX_EPSILON) for x in values])
epsilons = []
for name in GENERATE_EPSILONS.keys():
    epsilons.extend([f"{name}-{_}" for _ in range(len(SETTINGS[name]))])

from fedrpdp.accountants.utils import get_noise_multiplier_fed
nm = get_noise_multiplier_fed(
    target_epsilon=10.0,
    target_delta=target_delta,
    inner_sample_rate=1.0,
    outer_sample_rate=CLIENT_SAMPLE_RATE,
    inner_steps=num_updates,
    outer_rounds=num_rounds,
    accountant="rdp"
)
print("noise_multiplier : ", nm)

args = {
    "loss": bloss,
    "optimizer_class": torch.optim.SGD,
    "learning_rate": LR,
    "num_updates": num_updates,
    "nrounds": num_rounds,
    "device": device,
    "seed": hargs.seed
}

results_all_reps = []
# We set model and dataloaders to be the same for each rep
global_init = BaselineModel()
torch.manual_seed(hargs.seed) 

# We run FedAvg wo rPDP
test_dls, training_dls = [], []
for i in range(NUM_CLIENTS): # NUM_CLIENTS
    test_data = FedHeartDisease(center=i, train=False)
    test_dl = dl(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_dls.append(test_dl)

    train_data = FedHeartDisease(center=i, train=True)
    train_dl = dl(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    training_dls.append(train_dl)

args["training_dataloaders"] = training_dls
args["test_dataloaders"] = test_dls
args["client_sample_rate"] = CLIENT_SAMPLE_RATE
args["dp_target_epsilon"] = [None] * NUM_CLIENTS
current_args = copy.deepcopy(args)
current_args["model"] = copy.deepcopy(global_init)

s = FedAvg(**current_args, log=False)
cm, perf = s.run(metric=metric, device=device)
mean_perf = np.mean(perf[-3:])
print(f"Mean performance without rPDP, Mean Perf={mean_perf:.4f}")
results_all_reps.append({"perf": perf, "mean_perf": mean_perf, "e": "PrivacyFree", "d": None, "nm": None, "norm": None, "bs": BATCH_SIZE, "seed": hargs.seed})

results = pd.DataFrame.from_dict(results_all_reps)
results.to_csv(os.path.join(hargs.outdir, f"results_heart_disease_{hargs.seed}.csv"), index=False)

del training_dls

# We run FedAvg with rPDP
training_dls = []
for i in range(NUM_CLIENTS): # NUM_CLIENTS
    train_data = FedHeartDisease(center=i, train=True)
    train_dl = dl(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    training_dls.append(train_dl)
    del train_dl

args["training_dataloaders"] = training_dls
args["learning_rate"] = LR_DP
args["dp_target_delta"] = target_delta
args["dp_noise_multiplier"] = nm
args["dp_max_grad_norm"] = grad_norm

curve_fn = get_sample_rate_curve(
    target_delta = target_delta,
    noise_multiplier=nm,
    num_updates=num_updates,
    num_rounds=num_rounds,
    client_rate = CLIENT_SAMPLE_RATE
)

for ename in epsilons:
    name, p_id = ename.split('-')
    e = np.array([BoundedFunc(GENERATE_EPSILONS[name](len(train_dl.dataset), SETTINGS[name][int(p_id)])) for train_dl in training_dls])
    
    # We run FedAvg with rPDP (StrongForAll)
    temp_e = copy.deepcopy(e)
    min_epsilon = min([min(elements) for elements in temp_e])
    for i in range(NUM_CLIENTS):
        temp_e[i][:] = min_epsilon

    privacy_engine = PrivacyEngine(accountant='rdp')
    privacy_engine.sample_rate_fn = curve_fn 
    current_args = copy.deepcopy(args)
    current_args["model"] = copy.deepcopy(global_init)
    current_args["dp_target_epsilon"] = temp_e
    current_args["privacy_engine"] = privacy_engine
    
    s = FedAvg(**current_args, log=False)
    # testing ...
    expected_batch_size = max(1, int(sum(s.models_list[0].privacy_engine.sample_rate)))
    print(f"sample_rate: {min(s.models_list[0].privacy_engine.sample_rate)} / {max(s.models_list[0].privacy_engine.sample_rate)}, expected_batch_size: {expected_batch_size} / {len(s.models_list[0]._train_dl.dataset)}")
    # testing end 
    cm, perf = s.run(metric=metric, device=device)
    mean_perf = np.mean(perf[-3:])
    print(f"Mean performance of StrongForAll, eps={min_epsilon}, delta={target_delta}, Perf={mean_perf:.4f}")
    results_all_reps.append({"perf": perf, "mean_perf": round(mean_perf,4), "e": f"{ename}-StrongForAll", "d": target_delta, "nm": nm, "norm": grad_norm, "bs": expected_batch_size, "seed": hargs.seed})
    del privacy_engine, s, cm, mean_perf

    results = pd.DataFrame.from_dict(results_all_reps)
    results.to_csv(os.path.join(hargs.outdir, f"results_heart_disease_{hargs.seed}.csv"), index=False)
    
    # We run FedAvg with rPDP
    print(f" We run FedAvg with rPDP ({ename}) ...")
    privacy_engine = PrivacyEngine(accountant='pers_rdp')
    privacy_engine.sample_rate_fn = curve_fn 

    current_args = copy.deepcopy(args)
    current_args["model"] = copy.deepcopy(global_init)
    current_args["dp_target_epsilon"] = e
    current_args["privacy_engine"] = privacy_engine

    s = FedAvg(**current_args, log=False)
    # testing ...
    expected_batch_size = max(1, int(sum(s.models_list[0].privacy_engine.sample_rate)))
    print(f"sample_rate: {min(s.models_list[0].privacy_engine.sample_rate)} / {max(s.models_list[0].privacy_engine.sample_rate)}, expected_batch_size: {expected_batch_size} / {len(s.models_list[0]._train_dl.dataset)}")
    # testing end 
    cm, perf = s.run(metric=metric, device=device)
    mean_perf = np.mean(perf[-3:])
    print(f"Mean performance of {name}, min_eps={min(e[0]):.4f}, max_eps={max(e[0]):.4f}, delta={target_delta}, Perf={mean_perf:.4f}, seed={hargs.seed}")
    results_all_reps.append({"perf": perf, "mean_perf": round(mean_perf,4), "e": f"{ename}-Ours", "d": target_delta, "nm": nm, "norm": grad_norm, "bs": expected_batch_size, "seed": hargs.seed})
    del privacy_engine, s, cm, mean_perf

    results = pd.DataFrame.from_dict(results_all_reps)
    results.to_csv(os.path.join(hargs.outdir, f"results_heart_disease_{hargs.seed}.csv"), index=False)

    # We run FedAvg with rPDP (Dropout)
    print(" We run FedAvg with rPDP (Dropout) ...")
    temp_e = copy.deepcopy(e)
    mean_epsilon = np.mean([np.mean(elements) for elements in temp_e])
    for i in range(NUM_CLIENTS):
        mask = temp_e[i] < mean_epsilon
        temp_e[i][mask] = 0
        temp_e[i][~mask] = mean_epsilon
        
    privacy_engine = PrivacyEngine(accountant="rdp")
    privacy_engine.sample_rate_fn = curve_fn 
    current_args = copy.deepcopy(args)
    current_args["model"] = copy.deepcopy(global_init)
    current_args["dp_target_epsilon"] = temp_e
    current_args["privacy_engine"] = privacy_engine
    
    s = FedAvg(**current_args, log=False)
    # testing ...
    expected_batch_size = max(1, int(sum(s.models_list[0].privacy_engine.sample_rate)))
    print(f"sample_rate: {min(s.models_list[0].privacy_engine.sample_rate)} / {max(s.models_list[0].privacy_engine.sample_rate)}, expected_batch_size: {expected_batch_size} / {len(s.models_list[0]._train_dl.dataset)}")
    # testing end 
    cm, perf = s.run(metric=metric, device=device)
    mean_perf = np.mean(perf[-3:])
    print(f"Mean performance of Dropout, eps={mean_epsilon}, delta={target_delta}, Perf={mean_perf:.4f}")
    results_all_reps.append({"perf": perf, "mean_perf": round(mean_perf,4), "e": f"{ename}-Dropout", "d": target_delta, "nm": nm, "norm": grad_norm, "bs": expected_batch_size, "seed": hargs.seed})
    del privacy_engine, s, cm, mean_perf

    results = pd.DataFrame.from_dict(results_all_reps)
    results.to_csv(os.path.join(hargs.outdir, f"results_heart_disease_{hargs.seed}.csv"), index=False)