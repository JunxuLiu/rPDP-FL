from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import os

WAY = 5
SHOT = 1200
NUM_LABELS = 10
NUM_CLIENTS = 10

DATA_ROOT = '/data/privacyGroup/liujunxu/datasets/mnist/'

train_data = MNIST(DATA_ROOT, train=True, download=True, 
                transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
test_data = MNIST(DATA_ROOT, train=False, download=True, 
                transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
data, target = train_data.data, train_data.targets

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

    # 长度为10的数组: 指示每个标签已分配出去的样本数
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
            idx[lbl] += SHOT # 记录当前标签已分配的样本数
        print(user, np.unique(y[user]))
    
    if FINDED:
        break
    
# Create federated datasets
save_dir = f"../niid_{WAY}way_{SHOT}shot"
if not (os.path.exists(save_dir)):
    os.mkdir(save_dir)
    
for cid in range(NUM_CLIENTS):
    cname = 'client{:d}'.format(cid) 
    client_X = [e.numpy() for e in X[cid]]
    client_y = y[cid]

    combined = list(zip(client_X, client_y)) 
    random.shuffle(combined)
    print(cname, len(combined))
    np.save(os.path.join(save_dir, f"{cname}.npy"), combined)
    