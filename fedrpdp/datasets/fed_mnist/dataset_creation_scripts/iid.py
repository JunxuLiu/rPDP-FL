from torchvision.datasets import MNIST
import numpy as np
import os

NUM_LABELS = 10
NUM_CLIENTS = 50

DATA_ROOT = '/data/privacyGroup/liujunxu/datasets/mnist/'

train_data = MNIST(DATA_ROOT, train=True, download=True)
test_data = MNIST(DATA_ROOT, train=False, download=True)
data, target = train_data.data, train_data.targets

# Create federated datasets
save_dir = f"../iid_{NUM_CLIENTS}"
if not (os.path.exists(save_dir)):
    os.mkdir(save_dir)

num_examples = len(train_data)
num_examples_per_client = num_examples // NUM_CLIENTS
perm = np.random.permutation(num_examples)
for cid in range(NUM_CLIENTS):
    indices = np.array(perm[cid * num_examples_per_client : (cid+1) * num_examples_per_client])
    client_X = data.numpy()[perm[cid * num_examples_per_client : (cid+1) * num_examples_per_client]]
    client_y = target.numpy()[perm[cid * num_examples_per_client : (cid+1) * num_examples_per_client]]
    combined = list(zip(client_X, client_y)) 

    cname = 'client{:d}'.format(cid) 
    np.save(os.path.join(save_dir, f"{cname}.npy"), combined)