import argparse
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
from myopacus.utils.batch_memory_manager import BatchMemoryManager

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='heart_disease')
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
save_filename = os.path.join(save_dir, f"results_dp_sgd_{args.dataset}_{args.seed}.csv")

""" Prepare training datasets """
# data_dir
if args.dataset == "heart_disease":
    data_path = os.path.join(project_abspath, dict["dataset_dir"])
else:
    data_path = os.path.join(project_abspath, dict["dataset_dir"]["iid_10"])

BATCH_SIZE = dict["sgd"]["batch_size"]
NUM_STEPS = dict["dpsgd"]["num_steps"]
LR_DP = dict["dpsgd"]["learning_rate"]
TARGET_EPSILON = dict["dpsgd"]["target_epsilon"]
TARGET_DELTA = dict["dpsgd"]["target_delta"]
MAX_GRAD_NORM = dict["dpsgd"]["max_grad_norm"]
MAX_PHYSICAL_BATCH_SIZE = dict["dpsgd"]["max_physical_batch_size"]

rawdata = RawClass(data_path=data_path)
train_pooled = DataLoader(
    FedClass(rawdata=rawdata, train=True, pooled=True),
    batch_size=BATCH_SIZE,
)
test_pooled = DataLoader(
    FedClass(rawdata=rawdata, train=False, pooled=True),
    batch_size=BATCH_SIZE,
)
print(len(train_pooled.dataset), len(test_pooled.dataset))

""" Prepare model and loss """
model = BaselineModel.to(device)
criterion = BaselineLoss()
optimizer = Optimizer(model.parameters(), lr=LR_DP)

""" Prepare privacy engine """
privacy_engine = PrivacyEngine(accountant="rdp")
model, optimizer, train_pooled = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_pooled,
    epochs=NUM_STEPS,
    target_epsilon=TARGET_EPSILON,
    target_delta=TARGET_DELTA,
    max_grad_norm=MAX_GRAD_NORM
)
print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")
if MAX_PHYSICAL_BATCH_SIZE > 0:
    with BatchMemoryManager(
            data_loader=train_pooled,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer) as memory_safe_train_loader:
        train_pooled = memory_safe_train_loader

# ======== Start Training ==========
test_accs = []
current_batch_size, i = 0, 0
train_loader_iter = iter(train_pooled)
while i < NUM_STEPS:
    try:
        batch = next(train_loader_iter)
    except StopIteration:
        train_loader_iter = iter(train_pooled)
        batch = next(train_loader_iter)
    
    model.train()
    optimizer.zero_grad()
    batch = tuple(t.to(device) for t in batch)
    current_batch_size += len(batch[-1]) 

    if len(batch) == 2: # for other datasets
        logits = model(batch[0])
        loss = criterion(logits, batch[1])

    elif len(batch) == 4: # for snli dataset
        inputs = {'input_ids':    batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels':         batch[3]}
        outputs = model(**inputs) # output = loss, logits, hidden_states, attentions
        loss, logits = outputs[:2]

    loss.backward()
    optimizer.step()

    if i == privacy_engine.accountant.history[-1][-1] - 1:
        i += 1
        current_batch_size = 0

        model.eval()
        total_correct, total_points = 0, 0
        with torch.no_grad():
            for batch in iter(test_pooled):
                batch = tuple(t.to(device) for t in batch)
                if len(batch) == 2: # for other datasets
                    logits = model(batch[0])
                    loss = criterion(logits, batch[1])

                elif len(batch) == 4: # for snli dataset
                    inputs = {'input_ids':    batch[0],
                                'attention_mask': batch[1],
                                'token_type_ids': batch[2],
                                'labels':         batch[3]}
                    outputs = model(**inputs) # output = loss, logits, hidden_states, attentions
                    loss, logits = outputs[:2]

                test_correct = metric(y_true=batch[-1].detach().cpu().numpy(), y_pred=logits.detach().cpu().numpy())
                total_correct += test_correct
                total_points += len(batch[-1])

        test_accs.append(round(total_correct/total_points, 4))
        print(f"Step {i}: test_correct = {total_correct}/{total_points}, "\
                f"test_acc = {round(total_correct/total_points, 4)}")

record_row = [{
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "mean_pref(last5)": np.mean(test_accs[-5:]),
    "e": TARGET_EPSILON, "d": TARGET_DELTA, 
    "nm": optimizer.noise_multiplier, 
    "norm": MAX_GRAD_NORM,
    "bs": BATCH_SIZE,
    "lr": LR_DP,
    "pref": test_accs}]
results = pd.DataFrame.from_dict(record_row)
results.to_csv(save_filename, mode='a', index=False)
