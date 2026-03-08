from typing import Tuple

import pandas as pd

from provided_embeddings_models.constants import RANDOM_STATE
from sklearn.model_selection import train_test_split


def split_embeddings(all_embeddings: pd.DataFrame, label_col: str,
                     val_size: float = 0.1, test_size: float = 0.1) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split embeddings into train, validation, and test sets.
    Combined size of the validation and test sets must be less than the entire dataset.

    :param all_embeddings: pandas dataframe with all embeddings
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

    # First split: training set and a temp set (val + test)
    train_features, temp_features, train_targets, temp_targets = train_test_split(
        features, targets, test_size=(val_size + test_size), random_state=RANDOM_STATE, shuffle=True,
        stratify=targets
    )

    # Adjust val and test sizes for second split
    new_test_size = test_size / (val_size + test_size)

    # Second split: val set and test set from temp set
    val_features, test_features, val_targets, test_targets = train_test_split(
        temp_features, temp_targets, test_size=new_test_size, random_state=RANDOM_STATE, shuffle=True,
        stratify=temp_targets
    )

    return train_features, val_features, test_features, train_targets, val_targets, test_targets
