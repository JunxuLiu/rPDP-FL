import argparse
import copy
import datetime
import importlib
import numpy as np
import os
import pandas as pd
import warnings
warnings.simplefilter("ignore")

import torch
from torch.utils.data import DataLoader
from configs.config_utils import read_config, get_config_file_path
from myopacus import PrivacyEngine
from myopacus.accountants.rpdp_utils import GENERATE_EPSILONS_FUNC

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='fed_heart_disease')
parser.add_argument("--gpuid", type=int, default=7,
    help="Index of the GPU device.")
parser.add_argument("--seed", type=int, default=41, 
                    help="random seed")
args = parser.parse_args()

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
save_filename = os.path.join(save_dir, f"results_sgd_rpdp_{args.dataset}_{args.seed}.csv")

""" Prepare training datasets """
# data_dir
if args.dataset == "heart_disease":
    data_path = os.path.join(project_abspath, dict["dataset_dir"])
else:
    data_path = os.path.join(project_abspath, dict["dataset_dir"]["iid_10"])

NUM_LABELS = dict["num_labels"]
BATCH_SIZE = dict["sgd"]["batch_size"]
NUM_STEPS = dict["dpsgd"]["num_steps"]
LR_DP = dict["rpdpsgd"]["learning_rate"]
TARGET_DELTA = dict["dpsgd"]["target_delta"]
MAX_GRAD_NORM = dict["dpsgd"]["max_grad_norm"]
MAX_PHYSICAL_BATCH_SIZE = dict["dpsgd"]["max_physical_batch_size"]

rawdata = RawClass(data_path=data_path)
train_pooled = FedClass(rawdata=rawdata, train=True, pooled=True)
test_pooled = FedClass(rawdata=rawdata, train=False, pooled=True)
train_dataloader = DataLoader(train_pooled, batch_size=len(train_pooled))
test_dataloader = DataLoader(test_pooled, batch_size=BATCH_SIZE)
print(len(train_pooled), len(test_pooled))

""" Prepare personalized epsilons """
# different distributions & different settings
SETTINGS = dict["rpdpfedavg"]["settings"]
MIN_EPSILON, MAX_EPSILON = dict["rpdpfedavg"]["min_epsilon"], dict["rpdpfedavg"]["max_epsilon"]
BoundedFunc = lambda values: np.array([min(max(x, MIN_EPSILON), MAX_EPSILON) for x in values])
epsilons = []
for name in GENERATE_EPSILONS_FUNC.keys():
    epsilons.extend([f"{name}-{_}" for _ in range(len(SETTINGS[name]))])
print(epsilons)

# ======== Start Training ==========
results_all_reps = []
for ename in epsilons:
    name, p_id = ename.split('-')
    target_epsilons = np.array(BoundedFunc(GENERATE_EPSILONS_FUNC[name](len(train_pooled), SETTINGS[name][int(p_id)])))
    
    print(f" We run SGD with rPDP ({ename}) ...")
    model = copy.deepcopy(BaselineModel).to(device)
    criterion = BaselineLoss()
    optimizer = Optimizer(model.parameters(), lr=LR_DP)
    privacy_engine = PrivacyEngine(accountant="fed_rdp", n_clients=1)
    model, optimizer, train_dataloader = privacy_engine.make_private_with_personalization(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        num_steps=NUM_STEPS,
        target_epsilons=target_epsilons,
        target_delta=TARGET_DELTA,
        max_epsilon=MAX_EPSILON,
        max_grad_norm=MAX_GRAD_NORM,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE
    )
    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")

    current_batch_size, i = 0, 0
    test_accs = []
    train_loader_iter = iter(train_dataloader)
    while i < NUM_STEPS:
        model.train()
        try:
            data, target = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_dataloader)
            data, target = next(train_loader_iter)      
        current_batch_size += len(target) 
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(input=output, target=target)
        loss.backward()
        optimizer.step()

        if len(privacy_engine.accountant) and (i == privacy_engine.accountant.history[-1][-1] - 1):
            i += 1
            current_batch_size = 0
            total_correct, total_points = 0, 0
            with torch.no_grad():
                model.eval()
                for data, target in iter(test_dataloader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    test_correct = metric(y_true=target.detach().cpu().numpy(), y_pred=output.detach().cpu().numpy())
                    total_correct += test_correct
                    total_points += len(target)

                test_accs.append(round(total_correct/total_points, 4))
                print(f"Step {i}: test_correct = {total_correct}/{total_points}, " \
                      f"test_acc = {round(total_correct/total_points, 4)}")
    
    record_row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mean_pref(last5)": np.mean(test_accs[-5:]),
        "e": {ename}, "d": TARGET_DELTA, 
        "nm": optimizer.noise_multiplier, 
        "norm": MAX_GRAD_NORM,
        "bs": max(1, int(sum(privacy_engine.accountant.sample_rate))),
        "lr": LR_DP,
        "pref": test_accs}
    results_all_reps.append(record_row)
    del privacy_engine, model, criterion, optimizer
            
results = pd.DataFrame.from_dict(results_all_reps)
results.to_csv(save_filename, mode='a', index=False)