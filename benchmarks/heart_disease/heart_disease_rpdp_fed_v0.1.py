import copy
from itertools import product

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
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    CLIENT_SAMPLE_RATE,
    BaselineModel,
    BaselineLoss,
    FedHeartDisease,
    get_nb_max_rounds,
    metric,
)
from fedrpdp.strategies import FedAvg
from fedrpdp.utils import evaluate_model_on_tests
from fedrpdp.utils.rpdp_utils import (
    get_sample_rate_curve,
    MultiLevels, 
    MixGauss, 
    Pareto,
)

torch.multiprocessing.set_sharing_strategy("file_system")

n_repetitions = 5
LR = 0.01
LR_DP = 0.01
num_updates = 50
num_rounds = 15
bloss = BaselineLoss()
# We init the strategy parameters to the following default ones

args = {
    "loss": bloss,
    "optimizer_class": torch.optim.SGD,
    "learning_rate": LR,
    "num_updates": num_updates,
    "nrounds": num_rounds,
}

# personalized epsilons
# different distributions & different settings
STYLES = ["ThreeLevels", "BoundedPareto", "BoundedMixGauss"]
GENERATE_EPSILONS = {
    "ThreeLevels": lambda n, params: MultiLevels(3, *params, n),
    "BoundedPareto": lambda n, params: Pareto(*params, n), 
    "BoundedMixGauss": lambda n, params: MixGauss(*params, n),
}
SETTINGS = {
    "ThreeLevels": [[[0.7,0.2,0.1], [0.1, 0.4, 1.0]]],
    "BoundedPareto": [[10.0, 0.1]], 
    "BoundedMixGauss": [[[0.7,0.2,0.1], [(0.1, 0.05), (0.4, 0.05), (1.0, 0.1)]]],
}
BoundedFunc = lambda values: [min(max(x, 0.1), 1.0) for x in values]
epsilons = []
for name in STYLES:
    epsilons.extend([f"{name}-{_}" for _ in range(len(SETTINGS[name]))])

# deltas = [10 ** (-i) for i in range(1, 5)]
deltas = [1e-3]
noise_multipliers = [5.0]

START_SEED = 42
seeds = np.arange(START_SEED, START_SEED + n_repetitions).tolist()

test_dls, training_dls = [], []
for i in range(NUM_CLIENTS):
    test_data = FedHeartDisease(center=i, train=False)
    test_dl = dl(
        test_data,
        batch_size=len(test_data),
        shuffle=False,
        num_workers=0,
        # collate_fn=collate_fn,
    )
    test_dls.append(test_dl)

    train_data = FedHeartDisease(center=i, train=True)
    train_dl = dl(
        train_data,
        batch_size=len(train_data),
        shuffle=False,
        num_workers=0,
        # collate_fn=collate_fn,
    )
    training_dls.append(train_dl)

results_all_reps = []
nmdelta_list = list(product(noise_multipliers, deltas))
for se in seeds:
    # We set model and dataloaders to be the same for each rep
    global_init = BaselineModel()
    torch.manual_seed(se) 

    args["training_dataloaders"] = training_dls
    args["test_dataloaders"] = test_dls
    args["client_sample_rate"] = CLIENT_SAMPLE_RATE
    args["dp_target_epsilon"] = [None] * NUM_CLIENTS
    current_args = copy.deepcopy(args)
    current_args["model"] = copy.deepcopy(global_init)

    # We run FedAvg wo rPDP
    s = FedAvg(**current_args, log=False)
    cm, mean_perf = s.run(metric)
    print(f"Mean performance without rPDP, Perf={mean_perf:.4f}")
    results_all_reps.append({"perf": mean_perf, "e": None, "d": None, "nm": None, "norm": None, "seed": se})

    # We run FedAvg with rPDP
    for nm, d in nmdelta_list:
        curve_fn = get_sample_rate_curve(
            target_delta = d,
            noise_multiplier=nm,
            num_updates=num_updates,
            num_rounds=num_rounds,
            client_rate = CLIENT_SAMPLE_RATE
        )

        for ename in epsilons:
            name, p_id = ename.split('-')
            print(f"name: {name}, delta={d}, nm={nm}")
            e = [BoundedFunc(GENERATE_EPSILONS[name](len(train_dl.dataset), SETTINGS[name][int(p_id)])) for train_dl in training_dls]

    #         # We run FedAvg with rPDP
            privacy_engine = PrivacyEngine(accountant='pers_rdp')
            privacy_engine.sample_rate_fn = curve_fn # TODO: make it as an internal func of PrivacyEngine

            current_args = copy.deepcopy(args)
            current_args["model"] = copy.deepcopy(global_init)
            current_args["learning_rate"] = LR_DP
            current_args["dp_target_epsilon"] = e
            current_args["dp_target_delta"] = d
            current_args["dp_noise_multiplier"] = nm
            current_args["dp_max_grad_norm"] = 1.1
            current_args["privacy_engine"] = privacy_engine

            s = FedAvg(**current_args, log=False)
            cm, mean_perf = s.run(metric)
            print(f"Mean performance of {name}, min_eps={min(e[0]):.4f}, max_eps={max(e[0]):.4f}, delta={d}, Perf={mean_perf:.4f}, seed={se}")
            results_all_reps.append({"perf": round(mean_perf, 4), "e": ename, "d": d, "nm": nm, "norm": 1.1, "seed": se})
            del privacy_engine, s, cm, mean_perf


        # We run FedAvg with rPDP (WeakForAll)
        privacy_engine = PrivacyEngine(accountant='rdp')
        privacy_engine.sample_rate_fn = curve_fn 
        current_args = copy.deepcopy(args)
        current_args["model"] = copy.deepcopy(global_init)
        current_args["learning_rate"] = LR_DP
        current_args["dp_target_epsilon"] = [1.0] * NUM_CLIENTS
        current_args["dp_target_delta"] = d
        current_args["dp_noise_multiplier"] = nm
        current_args["dp_max_grad_norm"] = 1.1
        current_args["privacy_engine"] = privacy_engine
        # We run FedAvg with DP
        s = FedAvg(**current_args, log=False)
        cm, mean_perf = s.run(metric)
        print(f"Mean performance of WeakForAll, eps=1.0, delta={d}, Perf={mean_perf:.4f}")
        results_all_reps.append({"perf": round(mean_perf,4), "e": "WeakForAll(1.0)", "d": d, "nm": nm, "norm": 1.1, "seed": se})
        del privacy_engine, s, cm, mean_perf

        # We run FedAvg with rPDP (StrongForAll)
        privacy_engine = PrivacyEngine(accountant='rdp')
        privacy_engine.sample_rate_fn = curve_fn 
        current_args = copy.deepcopy(args)
        current_args["model"] = copy.deepcopy(global_init)
        current_args["learning_rate"] = LR_DP
        current_args["dp_target_epsilon"] = [0.1] * NUM_CLIENTS
        current_args["dp_target_delta"] = d
        current_args["dp_noise_multiplier"] = nm
        current_args["dp_max_grad_norm"] = 1.1
        current_args["privacy_engine"] = privacy_engine
        # We run FedAvg with DP
        s = FedAvg(**current_args, log=False)
        cm, mean_perf = s.run(metric)
        print(f"Mean performance of StrongForAll, eps=0.1, delta={d}, Perf={mean_perf:.4f}")
        results_all_reps.append({"perf": round(mean_perf,4), "e": "StrongForAll(0.1)", "d": d, "nm": nm, "norm": 1.1, "seed": se})
        del privacy_engine, s, cm, mean_perf

results = pd.DataFrame.from_dict(results_all_reps)
results.to_csv("results_fed_heart_disease.csv", index=False)
