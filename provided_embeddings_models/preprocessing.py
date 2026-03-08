from typing import Any, Tuple, List

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from provided_embeddings_models.constants import RANDOM_STATE
from sklearn.model_selection import train_test_split


def split_embeddings(all_embeddings: pd.DataFrame, label_col: str,
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
        steps_before: List[Tuple[str, Any]] | None = None,
        steps_after: List[Tuple[str, Any]] | None = None,
) -> Pipeline:
    """
    Build a scikit-learn Pipeline for preprocessing embeddings.

    :param n_components: if provided, appends a PCA step with this many components; if None, PCA is omitted.
    :param steps_before: optional list of (name, transformer) tuples prepended before StandardScaler.
    :param steps_after: optional list of (name, transformer) tuples appended after PCA.
    :return: unfitted sklearn Pipeline ready for fit_transform / transform calls.
    """
    steps: List[Tuple[str, Any]] = []
    if steps_before:
        steps.extend(steps_before)
    steps.append(("scaler", StandardScaler()))
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components)))
    if steps_after:
        steps.extend(steps_after)
    return Pipeline(steps)
