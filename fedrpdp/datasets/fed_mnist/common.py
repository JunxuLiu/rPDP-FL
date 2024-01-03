import torch

from .dataset import FedMnist
DATA_DIR = {
    'non-iid': 'fedrpdp/datasets/fed_mnist/niid_5way_1200shot',
    'iid_10': 'fedrpdp/datasets/fed_mnist/iid_10',
    'iid_50': 'fedrpdp/datasets/fed_mnist/iid_50',
}
CLIENT_SAMPLE_RATE = 0.8
BATCH_SIZE = 4
NUM_EPOCHS_POOLED = 50
LR = 0.001
Optimizer = torch.optim.Adam

FedClass = FedMnist