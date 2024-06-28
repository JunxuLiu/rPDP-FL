import argparse
import copy
import datetime
import importlib
import numpy as np
import os
import pandas as pd
import warnings # ignore warnings for clarity
warnings.simplefilter("ignore")

import torch
from torch.utils.data import DataLoader

from configs.config_utils import read_config, get_config_file_path
from myopacus import PrivacyEngine
from myopacus.strategies import FedAvg
from myopacus.accountants.rpdp_utils import GENERATE_EPSILONS_FUNC

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='heart_disease')
parser.add_argument("--gpuid", type=int, default=7,
                    help="Index of the GPU device.")
parser.add_argument("--seed", type=int, default=41, 
                    help="random seed")
args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def set_random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
set_random_seed(args.seed)
device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
module_name = f"datasets.fed_{args.dataset}"
try:
    dataset_modules = importlib.import_module(module_name)
    FedClass = dataset_modules.FedClass
    RawClass = dataset_modules.RawClass
    BaselineModel = dataset_modules.BaselineModel
    BaselineLoss = dataset_modules.BaselineLoss
    Optimizer = dataset_modules.Optimizer
    metric = dataset_modules.metric
    
except ModuleNotFoundError as e:
    print(f'{module_name} import failed: {e}')

project_abspath = os.path.abspath(os.path.join(os.getcwd(),".."))
dict = read_config(get_config_file_path(dataset_name=f"fed_{args.dataset}", debug=False))
# save_dir
save_dir = os.path.join(project_abspath, dict["save_dir"])
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_filename = os.path.join(save_dir, f"results_fedavg_rpdp_{args.dataset}_{args.seed}.csv")

NUM_CLIENTS = dict["fedavg"]["num_clients"]
NUM_STEPS = dict["fedavg"]["num_steps"]
NUM_ROUNDS = dict["fedavg"]["num_rounds"]
CLIENT_RATE = dict["fedavg"]["client_rate"]
BATCH_SIZE = dict["fedavg"]["batch_size"]
LR = dict["fedavg"]["learning_rate"]

LR_DP = dict["dpfedavg"]["learning_rate"]
MAX_GRAD_NORM = dict["dpfedavg"]["max_grad_norm"]
TARGET_DELTA = dict["dpfedavg"]["target_delta"]
MAX_PHYSICAL_BATCH_SIZE = dict["dpfedavg"]["max_physical_batch_size"]

""" Prepare local datasets """
# data_dir
if args.dataset == "heart_disease":
    data_path = os.path.join(project_abspath, dict["dataset_dir"])
else:
    data_path = os.path.join(project_abspath, dict["dataset_dir"][f"iid_{NUM_CLIENTS}"]) # data_path = os.path.join(project_abspath, dict["dataset_dir"][f"niid_{NUM_CLIENTS}"])

rawdata = RawClass(data_path=data_path)
test_dls, training_dls = [], []
for i in range(NUM_CLIENTS): # NUM_CLIENTS
    train_data = FedClass(rawdata=rawdata, center=i, train=True)
    train_dl = DataLoader(train_data, batch_size=len(train_data))
    training_dls.append(train_dl)

    test_data = FedClass(rawdata=rawdata, center=i, train=False)
    test_dl = DataLoader(test_data, batch_size=BATCH_SIZE)
    test_dls.append(test_dl)

""" Prepare model and loss """
# We set model and dataloaders to be the same for each rep
global_init = BaselineModel.to(device)
criterion = BaselineLoss()

training_args = {
    "training_dataloaders": training_dls,
    "test_dataloaders": test_dls,
    "loss": criterion,
    "optimizer_class": Optimizer,
    "learning_rate": LR_DP,
    "num_steps": NUM_STEPS,
    "num_rounds": NUM_ROUNDS,
    "client_rate": CLIENT_RATE,
    "device": device,
    "metric": metric,
    "seed": args.seed
}

""" Prepare personalized epsilons """
# different distributions & different settings
SETTINGS = dict["rpdpfedavg"]["settings"]
MIN_EPSILON, MAX_EPSILON = dict["rpdpfedavg"]["min_epsilon"], dict["rpdpfedavg"]["max_epsilon"]
BoundedFunc = lambda values: np.array([min(max(x, MIN_EPSILON), MAX_EPSILON) for x in values])
epsilons = []
for name in GENERATE_EPSILONS_FUNC.keys():
    epsilons.extend([f"{name}-{_}" for _ in range(len(SETTINGS[name]))])

for ename in epsilons:
    name, p_id = ename.split('-')
    target_epsilons = np.array([BoundedFunc(GENERATE_EPSILONS_FUNC[name](len(train_dl.dataset), SETTINGS[name][int(p_id)])) for train_dl in training_dls])

    # We run FedAvg with rPDP
    print(f" We run FedAvg with rPDP ({ename}) ...")
    set_random_seed(args.seed)
    privacy_engine = PrivacyEngine(accountant="fed_rdp", n_clients=NUM_CLIENTS)
    privacy_engine.prepare_fedrpdp(
        num_steps = NUM_STEPS,
        num_rounds = NUM_ROUNDS,
        client_rate = CLIENT_RATE,
        target_epsilons = target_epsilons,
        target_delta = TARGET_DELTA,
        max_epsilon = MAX_EPSILON,
        max_grad_norm = MAX_GRAD_NORM,
        max_physical_batch_size = MAX_PHYSICAL_BATCH_SIZE
    )
    current_args = copy.deepcopy(training_args)
    current_args["model"] = copy.deepcopy(global_init)
    current_args["privacy_engine"] = privacy_engine

    s = FedAvg(**current_args, log=False)
    cm, perf = s.run()
    mean_perf = np.mean(perf[-3:])
    expected_batch_size = [int(sum(acct.sample_rate)) for acct in s.privacy_engine.accountant.accountants]

    print(f"Mean performance of {name}, min_eps={min(target_epsilons[0]):.4f}, max_eps={max(target_epsilons[0]):.4f}, delta={TARGET_DELTA}, Perf={mean_perf:.4f}, seed={args.seed}")
    results_dict = [{
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mean_perf": round(mean_perf,4), "perf": perf, 
        "e": f"{ename}-Ours", 
        "d": TARGET_DELTA, 
        "nm": round(s.privacy_engine.default_noise_multiplier, 2), 
        "norm": MAX_GRAD_NORM, 
        "bs": expected_batch_size, 
        "lr": LR_DP,
        "num_clients": NUM_CLIENTS,
        "client_rate": CLIENT_RATE}]
    results = pd.DataFrame.from_dict(results_dict)
    results.to_csv(save_filename, mode='a', index=False)
    del privacy_engine, s, cm, mean_perf

    # We run FedAvg with rPDP (StrongForAll)
    set_random_seed(args.seed)
    min_epsilon = min([min(per_client_epsilons) for per_client_epsilons in target_epsilons])
    print(min_epsilon)

    privacy_engine = PrivacyEngine(accountant="fed_rdp", n_clients=NUM_CLIENTS)
    privacy_engine.prepare_feddp(
        num_steps = NUM_STEPS,
        num_rounds = NUM_ROUNDS,
        sample_rate = BATCH_SIZE / min([len(train_dl.dataset) for train_dl in training_dls]),
        client_rate = CLIENT_RATE,
        target_epsilon = min_epsilon,
        target_delta = TARGET_DELTA,
        max_grad_norm = MAX_GRAD_NORM,
        max_physical_batch_size = MAX_PHYSICAL_BATCH_SIZE
    )
    current_args = copy.deepcopy(training_args)
    current_args["model"] = copy.deepcopy(global_init)
    current_args["privacy_engine"] = privacy_engine
    
    s = FedAvg(**current_args, log=False)
    cm, perf = s.run()
    mean_perf = np.mean(perf[-3:])
    expected_batch_size = [int(acct.sample_rate * len(train_dl.dataset)) for acct, train_dl in zip(s.privacy_engine.accountant.accountants, training_dls)]
    
    print(f"Mean performance of StrongForAll, eps={min_epsilon}, delta={TARGET_DELTA}, Perf={mean_perf:.4f}")
    results_dict = [{
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mean_perf": round(mean_perf,4), "perf": perf, 
        "e": f"{ename}-StrongForAll({min_epsilon})", 
        "d": TARGET_DELTA, 
        "nm": round(s.privacy_engine.default_noise_multiplier, 2), 
        "norm": MAX_GRAD_NORM, 
        "bs": expected_batch_size, 
        "lr": LR_DP,
        "num_clients": NUM_CLIENTS,
        "client_rate": CLIENT_RATE}]
    results = pd.DataFrame.from_dict(results_dict)
    results.to_csv(save_filename, mode='a', index=False, header=False)
    del privacy_engine, s, cm, mean_perf

    # We run FedAvg with rPDP (Dropout)
    print(" We run FedAvg with rPDP (Dropout) ...")
    set_random_seed(args.seed)
    temp_epsilons = copy.deepcopy(target_epsilons)
    mean_epsilon = np.mean([np.mean(per_client_epsilons) for per_client_epsilons in temp_epsilons])
    for i in range(NUM_CLIENTS):
        mask = temp_epsilons[i] < mean_epsilon
        temp_epsilons[i][mask] = 0
        temp_epsilons[i][~mask] = mean_epsilon
        
    privacy_engine = PrivacyEngine(accountant="fed_rdp", n_clients=NUM_CLIENTS)
    privacy_engine.prepare_fedrpdp(
        num_steps = NUM_STEPS,
        num_rounds = NUM_ROUNDS,
        client_rate = CLIENT_RATE,
        target_epsilons = temp_epsilons,
        target_delta = TARGET_DELTA,
        max_epsilon = MAX_EPSILON,
        max_grad_norm = MAX_GRAD_NORM,
        max_physical_batch_size = MAX_PHYSICAL_BATCH_SIZE
    )
    current_args = copy.deepcopy(training_args)
    current_args["model"] = copy.deepcopy(global_init)
    current_args["privacy_engine"] = privacy_engine

    s = FedAvg(**current_args, log=False)
    cm, perf = s.run()
    mean_perf = np.mean(perf[-3:])
    expected_batch_size = [int(sum(acct.sample_rate)) for acct in s.privacy_engine.accountant.accountants]

    print(f"Mean performance of Dropout, eps={mean_epsilon}, delta={TARGET_DELTA}, Perf={mean_perf:.4f}")
    results_dict = [{
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mean_perf": round(mean_perf,4), "perf": perf, 
        "e": f"{ename}-Dropout({mean_epsilon})", 
        "d": TARGET_DELTA, 
        "nm": round(s.privacy_engine.default_noise_multiplier, 2), 
        "norm": MAX_GRAD_NORM, 
        "bs": expected_batch_size, 
        "lr": LR_DP,
        "num_clients": NUM_CLIENTS,
        "client_rate": CLIENT_RATE}]
    results = pd.DataFrame.from_dict(results_dict)
    results.to_csv(save_filename, mode='a', index=False, header=False)
    del privacy_engine, s, cm, mean_perf