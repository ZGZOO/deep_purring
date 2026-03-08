from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_loaders(train_features: pd.DataFrame, val_features: pd.DataFrame, test_features: pd.DataFrame,
                train_labels: pd.DataFrame, val_labels: pd.DataFrame, test_labels: pd.DataFrame,
                batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create torch DataLoaders for the train, validation, and test sets.

    :return: train_loader, val_loader, test_loader
    """
    # tensors
    train_features_tensor = torch.tensor(train_features.values, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)
    val_features_tensor = torch.tensor(val_features.values, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels.values, dtype=torch.long)
    test_features_tensor = torch.tensor(test_features.values, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.long)

    # datasets
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
