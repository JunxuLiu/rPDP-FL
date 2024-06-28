import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
CIFAR10_TRANSFORM = Compose([ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)])

class Cifar10Raw(Dataset):

    def __init__(
        self,
        data_path,
        X_dtype=torch.float32,
        y_dtype=torch.float32,
        train_fraction: float = 0.66,
        transform: bool = True
    ):
        if not (os.path.exists(data_path)):
            raise ValueError(f"The string {data_path} is not a valid path.")
        self.data_dir = Path(data_path)

        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.transform = transform
        self.num_clients = len(os.listdir(data_path))

        self.features = []
        self.labels = []
        self.sets = []

        for idx in range(self.num_clients):
            client_name = f"client{idx}"
            client_data = np.load(os.path.join(data_path, f"{client_name}.npy"), allow_pickle=True)

            features = [sample[0] for sample in client_data]
            labels = [sample[1] for sample in client_data]

            nb = len(client_data)
            indices_train, indices_test = train_test_split(
                np.arange(nb),
                test_size=1.0 - train_fraction,
                train_size=train_fraction,
                random_state=43,
                shuffle=True
            )
            sets = np.array(["train"] * nb)
            sets[indices_test] = "test"

            self.features.append(features)
            self.labels.append(labels)
            self.sets.append(sets)

        # for center_data_file in self.data_dir.glob("*.npy"):
        #     center_name = os.path.basename(center_data_file).split(".")[0]
        #     center_data = np.load(center_data_file, allow_pickle=True)

        #     center_X = [sample[0] for sample in center_data]
        #     center_y = [sample[1] for sample in center_data]
            
        #     self.features.extend(center_X)
        #     self.labels.extend(center_y)

        #     self.centers += [self.centers_number[center_name]] * len(center_X)

        #     # proposed modification to introduce shuffling before splitting the center
        #     nb = len(center_X)

        #     indices_train, indices_test = train_test_split(
        #         np.arange(nb),
        #         test_size=1.0 - train_fraction,
        #         train_size=train_fraction,
        #         random_state=43,
        #         shuffle=True
        #     )

        #     for i in np.arange(nb):
        #         if i in indices_test:
        #             self.sets.append("test")
        #         else:
        #             self.sets.append("train")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        assert idx < len(self.features), "Index out of range."
        
        X, y = self.features[idx], self.labels[idx]
        if self.transform is not None:
            X = self.transform(X)
        return X, y


class FedCifar10(Dataset):
    def __init__(
        self,
        rawdata: Cifar10Raw,
        center: int = 0,
        train: bool = True,
        pooled: bool = False,
        transform: bool = True
    ):
        assert center in range(rawdata.num_clients)

        self.transform = transform
        self.chosen_centers = [center]
        if pooled:
            self.chosen_centers = range(rawdata.num_clients)

        features, labels, sets = [], [], []
        for idx in self.chosen_centers:
            features.extend(rawdata.features[idx])
            labels.extend(rawdata.labels[idx])
            sets.extend(rawdata.sets[idx])

        chosen_sets = "train" if train else "test"
        self.features = [fp for idx, fp in enumerate(features) if sets[idx] == chosen_sets]
        self.labels = [fp for idx, fp in enumerate(labels) if sets[idx] == chosen_sets]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        assert idx < len(self.features), "Index out of range."
        X, y = self.features[idx], self.labels[idx]
        if self.transform:
            X = CIFAR10_TRANSFORM(X)
        return X, y