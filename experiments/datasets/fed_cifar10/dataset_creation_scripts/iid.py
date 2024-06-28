import numpy as np
import os
from torchvision.datasets import CIFAR10

import datasets

NUM_LABELS = 10
NUM_CLIENTS = 10

data_path = os.path.join(datasets.RAW_DATA_DIR, "cifar10")
train_data = CIFAR10(data_path, train=True, download=True)
data, target = np.array(train_data.data), np.array(train_data.targets)

# Create federated datasets
save_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(),"..")), f"iid_{NUM_CLIENTS}")
if not (os.path.exists(save_path)):
    os.mkdir(save_path)

num_examples = len(train_data)
num_examples_per_client = num_examples // NUM_CLIENTS
print(num_examples)

perm = np.random.permutation(num_examples)
for cid in range(NUM_CLIENTS):
    indices = np.array(perm[cid * num_examples_per_client : (cid+1) * num_examples_per_client]).astype(int)
    client_X = data[indices]
    client_y = target[indices]
    combined = list(zip(client_X, client_y)) 

    cname = 'client{:d}'.format(cid) 
    np.save(os.path.join(save_path, f"{cname}.npy"), combined)