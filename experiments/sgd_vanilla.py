import argparse
import datetime
import importlib
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from configs.config_utils import read_config, get_config_file_path
import warnings # ignore warnings for clarity
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='heart_disease')
parser.add_argument("--gpuid", type=int, default=7,
                    help="Index of the GPU device.")
parser.add_argument("--seed", type=int, default=41, 
                    help="random seed")
args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.manual_seed(args.seed) 
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
save_filename = os.path.join(save_dir, f"results_sgd_{args.dataset}_{args.seed}.csv")

""" Prepare training datasets """
# data_dir
if args.dataset == "heart_disease":
    data_path = os.path.join(project_abspath, dict["dataset_dir"])
else:
    data_path = os.path.join(project_abspath, dict["dataset_dir"]["iid_10"])

NUM_EPOCHS = dict["sgd"]["num_epochs"]
BATCH_SIZE = dict["sgd"]["batch_size"]
LR = dict["sgd"]["learning_rate"]
LOGGING_INTERVAL = dict["sgd"]["logging_interval"]

rawdata = RawClass(data_path=data_path)
train_pooled = DataLoader(
    FedClass(rawdata=rawdata, train=True, pooled=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_pooled = DataLoader(
    FedClass(rawdata=rawdata, train=False, pooled=True),
    batch_size=BATCH_SIZE,
    shuffle=False,
)
print(len(train_pooled.dataset), len(test_pooled.dataset))

""" Prepare model and loss """
model = BaselineModel.to(device)
optimizer = Optimizer(model.parameters(), lr=LR)
bloss = BaselineLoss()

# ======== Start Training ==========

results_all_reps, test_accs = [], []
for epoch in range(NUM_EPOCHS):
    losses = []
    for i, batch in enumerate(iter(train_pooled)):
        model.train()
        optimizer.zero_grad()
        batch = tuple(t.to(device) for t in batch)

        if len(batch) == 2: # for other datasets
            logits = model(batch[0])
            loss = bloss(logits, batch[1])

        elif len(batch) == 4: # for snli dataset
            inputs = {'input_ids':    batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels':         batch[3]}
            outputs = model(**inputs) # output = loss, logits, hidden_states, attentions
            loss, logits = outputs[:2]

        loss.backward()
        optimizer.step()

        train_metric = metric(batch[-1].detach().cpu().numpy(), logits.detach().cpu().numpy())
        train_loss = loss.item()
        losses.append(train_loss)
        
        if i % LOGGING_INTERVAL == 0:
            train_loss = np.mean(losses)
            model.eval()
            total_correct, total_points = 0, 0
            with torch.no_grad():
                for batch in iter(test_pooled):
                    batch = tuple(t.to(device) for t in batch)
                    if len(batch) == 2: # for other datasets
                        logits = model(batch[0])
                        loss = bloss(logits, batch[1]).item()

                    elif len(batch) == 4: # for snli dataset
                        inputs = {'input_ids':    batch[0],
                                    'attention_mask': batch[1],
                                    'token_type_ids': batch[2],
                                    'labels':         batch[3]}
                        outputs = model(**inputs) # output = loss, logits, hidden_states, attentions
                        loss, logits = outputs[:2]
                        
                    correct = metric(y_true=batch[-1].detach().cpu().numpy(), y_pred=logits.detach().cpu().numpy())
                    total_correct += correct
                    total_points += len(batch[-1])
                    
            print(f"Epoch={epoch}, Step={i}, perf={total_correct} / {total_points} ({total_correct/total_points:.4f}%), train_loss={train_loss}")
            test_accs.append(round(total_correct/total_points, 4))

record_row = [{
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "mean_pref(last5)": np.mean(test_accs[-5:]),
    "e": None, 
    "d": None, 
    "nm": None, 
    "norm": None, 
    "bs": BATCH_SIZE, 
    "lr": LR,
    "pref": test_accs
}]
results = pd.DataFrame.from_dict(record_row)
results.to_csv(save_filename, mode='a', index=False)
