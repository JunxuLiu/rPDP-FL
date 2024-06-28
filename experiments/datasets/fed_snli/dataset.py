import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

class SNLIRaw(Dataset):
    def __init__(
        self,
        data_path: str,
        train_fraction: float = 0.95,
        dtype: torch.dtype = torch.Tensor
    ):
        if not (os.path.exists(data_path)):
            raise ValueError(f"The string {data_path} is not a valid path.")
        
        self.data_dir = Path(data_path)
        self.dtype = dtype
        self.features, self.sets = [], []
        
        self.num_clients = len(os.listdir(data_path))
        for idx in range(self.num_clients):
            client_name = f"client{idx}"
            # client_data_file = self.data_dir.glob(f"{client_name}.npy")
            center_data = np.load(os.path.join(data_path, f"{client_name}.npy"), allow_pickle=True)
        
            features = [(
                torch.from_numpy(sample[0]), 
                torch.from_numpy(sample[1]),
                torch.from_numpy(sample[2]),
                torch.tensor(sample[3]))
                for sample in center_data
            ]

            nb = len(center_data)
            indices_train, indices_test = train_test_split(
                np.arange(nb),
                test_size=1.0 - train_fraction,
                train_size=train_fraction,
                random_state=43,
                shuffle=True
            )
            # print(len(indices_train), len(indices_test))
            sets = np.array(["train"] * nb)
            sets[indices_test] = "test"

            self.features.append(features)
            self.sets.append(sets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        assert idx < len(self.features), "Index out of range."
        X = self.features[idx]
        return X
    
class FedSNLI(Dataset):
    def __init__(
        self,
        rawdata: SNLIRaw,
        center: int = 0,
        train: bool = True,
        pooled: bool = False
    ):
        assert center in range(rawdata.num_clients)

        self.chosen_centers = [center]
        if pooled:
            self.chosen_centers = range(rawdata.num_clients)

        features, sets = [], []
        for idx in self.chosen_centers:
            features.extend(rawdata.features[idx])
            sets.extend(rawdata.sets[idx])

        chosen_sets = "train" if train else "test"
        self.features = [fp for idx, fp in enumerate(features) if sets[idx] == chosen_sets]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        assert idx < len(self.features), "Index out of range."
        X = self.features[idx]
        return X