import copy
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import multiprocessing as mp
print(f"num of CPU: {mp.cpu_count()}")

from torch.utils.data import DataLoader as dl
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'
import sys
sys.path.append('../..')

from fedrpdp import PrivacyEngine
from fedrpdp.datasets.fed_mnist import (
    DATA_DIR,
    NUM_CLIENTS,
    CLIENT_SAMPLE_RATE,
    FedMnist, # 切分原数据的操作全包装在class里
    BaselineModel,
    BaselineLoss,
    metric
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

project_abspath = os.path.abspath(os.path.join(os.getcwd(),"../.."))
data_path = os.path.join(project_abspath, DATA_DIR)
print('Project Path: ', project_abspath)
print('Dataset Path: ', data_path)

n_repetitions = 5
LR = 0.1
LR_DP = 0.1
BATCH_SIZE = 64
num_updates = 50
num_rounds = 20
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
    "ThreeLevels": [[[0.7,0.2,0.1], [0.5, 1.0, 5.0]]],
    "BoundedPareto": [[2.5, 0.5]], 
    "BoundedMixGauss": [[[0.7,0.2,0.1], [(0.5, 0.05), (1.0, 0.5), (5.0, 1.0)]]],
}
BoundedFunc = lambda values: [min(max(x, 0.1), 5.0) for x in values]
epsilons = []
for name in STYLES:
    epsilons.extend([f"{name}-{_}" for _ in range(len(SETTINGS[name]))])

# deltas = [10 ** (-i) for i in range(1, 5)]
deltas = [1e-5]
noise_multipliers = [2.0, 5.0, 10.0]

START_SEED = 42
seeds = np.arange(START_SEED, START_SEED + n_repetitions).tolist()

if __name__ == "__main__":
    train_datas, test_datas = [], []
    for i in range(NUM_CLIENTS): # NUM_CLIENTS
        test_data = FedMnist(center=i, train=False, data_path=data_path)
        train_data = FedMnist(center=i, train=True, data_path=data_path)

        train_datas.append(train_data)
        test_datas.append(test_data)

    # We run FedAvg with rPDP
    nm, d = 5.0, 1e-5
    curve_fn = get_sample_rate_curve(
        target_delta = d,
        noise_multiplier=nm,
        num_updates=num_updates,
        num_rounds=num_rounds,
        client_rate = CLIENT_SAMPLE_RATE
    )
    name = "ThreeLevels"
    e = [BoundedFunc(GENERATE_EPSILONS[name](len(train_set), SETTINGS[name][0])) for train_set in train_datas]
    privacy_engine = PrivacyEngine(accountant='pers_rdp')
    privacy_engine.sample_rate_fn = curve_fn 
    
    for num_workers in range(0, 2):
        test_dls, training_dls = [], []
        for i in range(NUM_CLIENTS): # NUM_CLIENTS
            test_dl = dl(
                test_datas[i],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=num_workers,
            )
            test_dls.append(test_dl)

            train_dl = dl(
                train_datas[i],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=num_workers,
            )
            training_dls.append(train_dl)

        # We set model and dataloaders to be the same for each rep
        global_init = BaselineModel()

        args["training_dataloaders"] = training_dls
        args["test_dataloaders"] = test_dls
        args["client_sample_rate"] = CLIENT_SAMPLE_RATE

        current_args = copy.deepcopy(args)
        current_args["model"] = copy.deepcopy(global_init)
        current_args["learning_rate"] = LR_DP
        current_args["dp_target_epsilon"] = e
        current_args["dp_target_delta"] = d
        current_args["dp_noise_multiplier"] = nm
        current_args["dp_max_grad_norm"] = 1.1
        current_args["privacy_engine"] = privacy_engine

        s = FedAvg(**current_args, log=False)

        start = time.time()
        for epoch in range(1, 3):
            for i, data in enumerate(s.models_list[0]._train_dl, 0):
                pass
        end = time.time()

        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
