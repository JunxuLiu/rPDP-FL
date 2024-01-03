import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

# from ...utils.flamby_utils import check_dataset_from_config
from . import common

class MnistRaw(Dataset):
    """Pytorch dataset containing all the features, labels and
    metadata for the Heart Disease dataset.

    Parameters
    ----------
    X_dtype : torch.dtype, optional
        Dtype for inputs `X`. Defaults to `torch.float32`.
    y_dtype : torch.dtype, optional
        Dtype for labels `y`. Defaults to `torch.int64`.
    debug : bool, optional,
        Whether or not to use only the part of the dataset downloaded in
        debug mode. Defaults to False.
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.

    Attributes
    ----------
    data_dir: str
        Where data files are located
    labels : pd.DataFrame
        The labels as a dataframe.
    features: pd.DataFrame
        The features as a dataframe.
    centers: list[int]
        The list with the center ids associated with the dataframes.
    sets: list[str]
        For each sample if it is from the train or the test.
    X_dtype: torch.dtype
        The dtype of the X features output
    y_dtype: torch.dtype
        The dtype of the y label output
    debug: bool
        Whether or not we use the dataset with only part of the features
    normalize: bool
        Whether or not to normalize the features. We use the corresponding
        training client to compute the mean and std per feature used to
        normalize.
        Defaults to True.
    """

    def __init__(
        self,
        X_dtype=torch.float32,
        y_dtype=torch.float32,
        debug=False,
        data_path=None,
        normalize=True,
        transform=True,
    ):
        """See description above"""
        if data_path is None:
            dict = check_dataset_from_config("fed_mnist", debug)
            self.data_dir = Path(dict["dataset_path"])
        else:
            if not (os.path.exists(data_path)):
                raise ValueError(f"The string {data_path} is not a valid path.")
            self.data_dir = Path(data_path)

        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.debug = debug
        self.transform = transform
        
        self.NUM_CLIENTS = len(os.listdir(data_path))
        self.centers_number = {f"client{i}": i for i in range(self.NUM_CLIENTS)}

        self.features = []
        self.labels = []
        self.centers = []
        self.sets = []

        self.train_fraction = 0.66

        for center_data_file in self.data_dir.glob("*.npy"):
            
            center_name = os.path.basename(center_data_file).split(".")[0]
            center_data = np.load(center_data_file, allow_pickle=True)

            center_X = [sample[0] for sample in center_data]
            center_y = [sample[1] for sample in center_data]
            
            self.features.extend(center_X)
            self.labels.extend(center_y)

            self.centers += [self.centers_number[center_name]] * len(center_X)

            # proposed modification to introduce shuffling before splitting the center
            nb = len(center_X)

            indices_train, indices_test = train_test_split(
                np.arange(nb),
                test_size=1.0 - self.train_fraction,
                train_size=self.train_fraction,
                random_state=43,
                shuffle=True
            )

            for i in np.arange(nb):
                if i in indices_test:
                    self.sets.append("test")
                else:
                    self.sets.append("train")

        
        # self.features = [
        #     torch.from_numpy(i.astype(np.float32)).reshape(1,28,28)
        #     for i in self.features
        # ]
        # self.labels = torch.from_numpy(np.array(self.labels))
        # Per-center Normalization much needed
        # self.centers_stats = {}
        # for center in range(NUM_CLIENTS):
        #     # We normalize on train only
        #     to_select = [
        #         (self.sets[idx] == "train") and (self.centers[idx] == center)
        #         for idx, _ in enumerate(self.features)
        #     ]
            # features_center = [
            #     fp for idx, fp in enumerate(self.features) if to_select[idx]
            # ]
            # features_tensor_center = torch.cat(
            #     [features_center[i][None, :] for i in range(len(features_center))],
            #     axis=0,
            # )
            # mean_of_features_center = features_tensor_center.mean(axis=0)
            # std_of_features_center = features_tensor_center.std(axis=0)
            # self.centers_stats[center] = {
            #     "mean": mean_of_features_center,
            #     "std": std_of_features_center,
            # }

        # We finally broadcast the means and stds over all datasets
        # self.mean_of_features = torch.zeros((len(self.features), 13), dtype=self.X_dtype)
        # self.std_of_features = torch.ones((len(self.features), 13), dtype=self.X_dtype)
        # for i in range(self.mean_of_features.shape[0]):
        #     self.mean_of_features[i] = self.centers_stats[self.centers[i]]["mean"]
        #     self.std_of_features[i] = self.centers_stats[self.centers[i]]["std"]

        # We normalize on train only for pooled as well
        # to_select = [(self.sets[idx] == "train") for idx, _ in enumerate(self.features)]
        # features_train = [fp for idx, fp in enumerate(self.features) if to_select[idx]]
        # features_tensor_train = torch.cat(
        #     [features_train[i][None, :] for i in range(len(features_train))], axis=0
        # )
        # self.mean_of_features_pooled_train = features_tensor_train.mean(axis=0)
        # self.std_of_features_pooled_train = features_tensor_train.std(axis=0)

        # We convert everything back into lists

        # self.mean_of_features = torch.split(self.mean_of_features, 1)
        # self.std_of_features = torch.split(self.std_of_features, 1)
        # self.mean_of_features_pooled_train = [
        #     self.mean_of_features_pooled_train for i in range(len(self.features))
        # ]
        # self.std_of_features_pooled_train = [
        #     self.std_of_features_pooled_train for i in range(len(self.features))
        # ]
        # self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        assert idx < len(self.features), "Index out of range."
        
        X, y = self.features[idx], self.labels[idx]
        if self.transform:
            mnist_transform = Compose([ToTensor(), Normalize(0.5, 0.5)])
            X = mnist_transform(X)
        return X, y


class FedMnist(MnistRaw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for Heart Disease federated classification.
    One can instantiate this dataset with train or test data coming from either
    of the 4 centers it was created from or all data pooled.
    The train/test split are arbitrarily fixed.

    Parameters
    ----------
    center : int, optional
        Default to 0
    train : bool, optional
        Default to True
    pooled : bool, optional
        Whether to take all data from the 2 centers into one dataset, by
        default False
    X_dtype : torch.dtype, optional
        Dtype for inputs `X`. Defaults to `torch.float32`.
    y_dtype : torch.dtype, optional
        Dtype for labels `y`. Defaults to `torch.int64`.
    debug : bool, optional,
        Whether or not to use only the part of the dataset downloaded in
        debug mode. Defaults to False.
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.
    normalize: bool
        Whether or not to normalize the features. We use the corresponding
        training client to compute the mean and std per feature used to
        normalize. When using pooled=True, we use the training part of the full
        dataset to compute the statistics, in order to reflect the differences
        between available informations in FL and pooled mode. Defaults to True.
    """

    def __init__(
        self,
        center: int = 0,
        train: bool = True,
        pooled: bool = False,
        X_dtype: torch.dtype = torch.Tensor,
        y_dtype: torch.dtype = torch.Tensor,
        debug: bool = False,
        data_path: str = None,
        normalize: bool = True,
        transform: bool = True
    ):
        """Cf class description"""

        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            debug=debug,
            data_path=data_path,
            normalize=normalize,
            transform=transform
        )
        assert center in range(self.NUM_CLIENTS)

        self.chosen_centers = [center]
        if pooled:
            self.chosen_centers = range(self.NUM_CLIENTS)
            # We set the apropriate statistics
            self.mean_of_features = self.mean_of_features_pooled_train
            self.std_of_features = self.std_of_features_pooled_train

        if train:
            self.chosen_sets = ["train"]
        else:
            self.chosen_sets = ["test"]

        to_select = [
            (self.sets[idx] in self.chosen_sets)
            and (self.centers[idx] in self.chosen_centers)
            for idx, _ in enumerate(self.features)
        ]

        self.features = [fp for idx, fp in enumerate(self.features) if to_select[idx]]
        self.sets = [fp for idx, fp in enumerate(self.sets) if to_select[idx]]
        self.labels = [fp for idx, fp in enumerate(self.labels) if to_select[idx]]
        self.centers = [fp for idx, fp in enumerate(self.centers) if to_select[idx]]
        # self.mean_of_features = [
        #     fp for idx, fp in enumerate(self.mean_of_features) if to_select[idx]
        # ]
        # self.std_of_features = [
        #     fp for idx, fp in enumerate(self.std_of_features) if to_select[idx]
        # ]
            