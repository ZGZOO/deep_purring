from typing import Any, Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from provided_embeddings_models.constants import RANDOM_STATE
from provided_embeddings_models.tasks import TaskType


def split_embeddings(all_embeddings: pd.DataFrame, label_col: TaskType,
                     val_size: float = 0.1, test_size: float = 0.1, regression: bool = False) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split embeddings into train, validation, and test sets.
    Combined size of the validation and test sets must be less than the entire dataset.

    :param all_embeddings: pandas dataframe with all embeddings
    :param label_col: name of target column
    :param val_size: size of validation set. Must be float in (0, 1). Defaults to 0.1.
    :param test_size: size of test set. Must be float in (0, 1). Defaults to 0.1.
    :param regression: if False, use stratified sampling instead of random sampling. Defaults to False.
    :return: train_features, val_features, test_features, train_targets, val_targets, test_targets
    """
    assert label_col in all_embeddings.columns
    assert 0 < val_size < 1
    assert 0 < test_size < 1
    assert 0 < val_size + test_size < 1

    features = all_embeddings.drop(columns=label_col)
    targets = all_embeddings[label_col]

    # First split: training set and a temp set (val + test)
    if regression:
        train_features, temp_features, train_targets, temp_targets = train_test_split(
            features, targets, test_size=(val_size + test_size), random_state=RANDOM_STATE, shuffle=True
        )
    else:
        train_features, temp_features, train_targets, temp_targets = train_test_split(
            features, targets, test_size=(val_size + test_size), random_state=RANDOM_STATE, shuffle=True,
            stratify=targets
        )

    # Adjust val and test sizes for second split
    new_test_size = test_size / (val_size + test_size)

    # Second split: val set and test set from temp set
    if regression:
        val_features, test_features, val_targets, test_targets = train_test_split(
            temp_features, temp_targets, test_size=new_test_size, random_state=RANDOM_STATE, shuffle=True
        )
    else:
        val_features, test_features, val_targets, test_targets = train_test_split(
            temp_features, temp_targets, test_size=new_test_size, random_state=RANDOM_STATE, shuffle=True,
            stratify=temp_targets
        )

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
