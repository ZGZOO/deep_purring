from typing import Any, Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from provided_embeddings_models.constants import RANDOM_STATE
from provided_embeddings_models.tasks import TaskType


def split_embeddings(all_embeddings: pd.DataFrame, label_col: TaskType,
                     val_size: float = 0.1, test_size: float = 0.1) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split embeddings into train, validation, and test sets, grouped by cat_id so that all
    recordings from a given cat land in exactly one split (no data leakage across cats).
    Combined size of the validation and test sets must be less than the entire dataset.

    :param all_embeddings: pandas dataframe with all embeddings; must contain a 'cat_id' column
    :param label_col: name of target column
    :param val_size: size of validation set. Must be float in (0, 1). Defaults to 0.1.
    :param test_size: size of test set. Must be float in (0, 1). Defaults to 0.1.
    :return: train_features, val_features, test_features, train_targets, val_targets, test_targets
    """
    assert label_col in all_embeddings.columns
    assert 0 < val_size < 1
    assert 0 < test_size < 1
    assert 0 < val_size + test_size < 1

    features = all_embeddings.drop(columns=label_col)
    targets = all_embeddings[label_col]
    groups = features["cat_id"]

    # Stage 1: train vs temp (val + test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss1.split(features, targets, groups=groups))

    train_features, train_targets = features.iloc[train_idx], targets.iloc[train_idx]
    temp_features, temp_targets = features.iloc[temp_idx], targets.iloc[temp_idx]
    temp_groups = groups.iloc[temp_idx]

    # Stage 2: val vs test from temp
    new_test_size = test_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=new_test_size, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss2.split(temp_features, temp_targets, groups=temp_groups))

    val_features, val_targets = temp_features.iloc[val_idx], temp_targets.iloc[val_idx]
    test_features, test_targets = temp_features.iloc[test_idx], temp_targets.iloc[test_idx]

    for df in (train_features, val_features, test_features):
        df.drop(columns=["cat_id"], inplace=True)

    return train_features, val_features, test_features, train_targets, val_targets, test_targets


def build_preprocessing_pipeline(
        n_components: int | None = None,
        random_state: int | None = None,
) -> Pipeline:
    """
    Build a scikit-learn Pipeline for preprocessing embeddings.

    :param n_components: if provided, appends a PCA step with this many components; if None, PCA is omitted.
    :param random_state: random state for pipeline steps (for reproducibility)
    :return: unfitted sklearn Pipeline ready for fit_transform / transform calls.
    """
    # TODO: For the end-to-end pipeline, add a CNN embedding extractor (to be implemented in cnn_embeddings.py)
    #  before existing steps so raw features pass through the CNN before StandardScaler/PCA.

    steps: List[Tuple[str, Any]] = []
    steps.append(("scaler", StandardScaler()))
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=random_state)))
    return Pipeline(steps)


def get_loaders(train_features: pd.DataFrame, val_features: pd.DataFrame, test_features: pd.DataFrame,
                train_labels: pd.DataFrame, val_labels: pd.DataFrame, test_labels: pd.DataFrame,
                batch_size: int, regression: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create torch DataLoaders for the train, validation, and test sets.

    :param train_features: training features
    :param val_features: validation features
    :param test_features: test features
    :param train_labels: training labels
    :param val_labels: validation labels
    :param test_labels: test labels
    :param batch_size: batch size
    :param regression: if True, cast labels to float32 (for MSE loss); otherwise long (for CrossEntropy).
    :return: train_loader, val_loader, test_loader
    """
    dtype = torch.float32 if regression else torch.long

    # tensors — accept both pd.DataFrame/Series and numpy arrays
    train_features_tensor = torch.tensor(np.asarray(train_features), dtype=torch.float32)
    train_labels_tensor = torch.tensor(np.asarray(train_labels), dtype=dtype)
    val_features_tensor = torch.tensor(np.asarray(val_features), dtype=torch.float32)
    val_labels_tensor = torch.tensor(np.asarray(val_labels), dtype=dtype)
    test_features_tensor = torch.tensor(np.asarray(test_features), dtype=torch.float32)
    test_labels_tensor = torch.tensor(np.asarray(test_labels), dtype=dtype)

    # datasets
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
