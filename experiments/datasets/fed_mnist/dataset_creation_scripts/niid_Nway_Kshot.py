import numpy as np
import os
import random

from torchvision.datasets import MNIST

WAY = 5
SHOT = 1200
NUM_LABELS = 10
NUM_CLIENTS = 10

dataset_abspath = os.path.abspath(os.path.join(os.getcwd(),"../.."))
data_path = os.path.join(dataset_abspath, "mnist")
if not os.path.exists(data_path):
    os.mkdir(data_path)
train_data = MNIST(data_path, train=True, download=True)
data, target = train_data.data, train_data.targets

# Create federated datasets
save_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(),"..")), f"niid_{NUM_CLIENTS}_{WAY}way_{SHOT}shot")
if not (os.path.exists(save_path)):
    os.mkdir(save_path)

NUM_SHARDS = NUM_CLIENTS * WAY 
NUM_SHARDS_PER_LABEL = len(data) // NUM_LABELS // SHOT

data_tr, data_ts = [], []
target_tr, target_ts = [], []

for i in range(NUM_LABELS):
    idx = target==i
    data_tr.append(list(data[idx]))
    target_tr.append(list(target[idx]))

while(True):
    ###### SPLIT DATA #######
    X = [[] for _ in range(NUM_CLIENTS)]
    y = [[] for _ in range(NUM_CLIENTS)]

    idx = np.zeros(NUM_LABELS, dtype=np.int64)
    shards_index = list(range(NUM_SHARDS))
    user_labels_count = {i: [] for i in range(NUM_CLIENTS)}
    FINDED = True
    for user in range(NUM_CLIENTS):
        if user == NUM_CLIENTS-1:
            rand_set = set(np.array(shards_index))
            labels = [idx // NUM_SHARDS_PER_LABEL for idx in rand_set] #
            if len(set(labels)) < WAY:
                print('Failed!')
                FINDED = False
                break
        
        else:
            trials = 0
            while(True):
                rand_set = set(np.random.choice(shards_index, WAY, replace=False))
                if len(rand_set) > WAY: 
                    continue

                labels = [idx // NUM_SHARDS_PER_LABEL for idx in rand_set] # 
                if len(set(labels)) == WAY:
                    break
                trials += 1
                if trials > 100:
                    print('trials > 100, NOT FINDED.')
                    FINDED = False
                    break
        
        if FINDED == False:
            break

        shards_index = list(set(shards_index) - rand_set) 

        for idx_lbl, lbl in enumerate(labels):
            if len(X[user]) == 0:
                X[user] = data_tr[lbl][idx[lbl]:idx[lbl]+SHOT]
                y[user] = [lbl] * SHOT
            else:
                X[user].extend(data_tr[lbl][idx[lbl]:idx[lbl]+SHOT])
                y[user].extend([lbl] * SHOT)
            idx[lbl] += SHOT
        print(user, np.unique(y[user]))
    
    if FINDED:
        break
    
for cid in range(NUM_CLIENTS):
    cname = 'client{:d}'.format(cid) 
    client_X = [e.numpy() for e in X[cid]]
    client_y = y[cid]

    combined = list(zip(client_X, client_y)) 
    random.shuffle(combined)
    print(cname, len(combined))
    np.save(os.path.join(save_path, f"{cname}.npy"), combined)
    