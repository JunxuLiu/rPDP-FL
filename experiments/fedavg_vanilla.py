import argparse
import datetime
import importlib
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import warnings
warnings.simplefilter("ignore")

from configs.config_utils import read_config, get_config_file_path
from myopacus.strategies import FedAvg

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
save_filename = os.path.join(save_dir, f"results_fedavg_{args.dataset}_{args.seed}.csv")

NUM_CLIENTS = dict["fedavg"]["num_clients"]
NUM_STEPS = dict["fedavg"]["num_steps"]
NUM_ROUNDS = dict["fedavg"]["num_rounds"]
CLIENT_RATE = dict["fedavg"]["client_rate"]
BATCH_SIZE = dict["fedavg"]["batch_size"]
LR = dict["fedavg"]["learning_rate"]

# data_dir
if args.dataset == "heart_disease":
    data_path = os.path.join(project_abspath, dict["dataset_dir"])
else:
    data_path = os.path.join(project_abspath, dict["dataset_dir"][f"iid_{NUM_CLIENTS}"])
    
rawdata = RawClass(data_path=data_path)
training_dls, test_dls = [], []
for i in range(NUM_CLIENTS):
    train_ds = FedClass(rawdata=rawdata, center=i, train=True)
    training_dls.append(DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True))

    test_ds = FedClass(rawdata=rawdata, center=i, train=False)
    test_dls.append(DataLoader(test_ds, batch_size=BATCH_SIZE))

""" Prepare model and loss """
model = BaselineModel.to(device)
criterion = BaselineLoss()

results_all_reps = []
current_args = {
    "training_dataloaders": training_dls,
    "test_dataloaders": test_dls,
    "model": model,
    "loss": criterion,
    "optimizer_class": Optimizer,
    "learning_rate": LR,
    "num_steps": NUM_STEPS,
    "num_rounds": NUM_ROUNDS,
    "client_rate": CLIENT_RATE,
    "device": device,
    "metric": metric,
    "seed": args.seed
}
# ======== Start Training ==========
s = FedAvg(**current_args, log=False)
cm, perf = s.run()
mean_perf = np.mean(perf[-3:])
print(f"Mean performance of vanilla FedAvg, Perf={mean_perf:.4f}")
record_row = [{
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "mean_perf": round(mean_perf, 4), "perf": perf, 
    "e": None, 
    "d": None, 
    "nm": None, 
    "norm": None, 
    "bs": BATCH_SIZE,
    "lr": LR,
    "num_clients": NUM_CLIENTS,
    "client_rate": CLIENT_RATE
}]
results = pd.DataFrame.from_dict(record_row)
results.to_csv(save_filename, mode='a', index=False)
# ======== End Training ============
