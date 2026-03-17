from typing import Any

import pandas as pd
from pandas import DataFrame, Series

from provided_embeddings_models.constants import EMBEDDINGS_DIR, AGE_GROUP_CATEGORIES, GENDER_CATEGORIES


def load_embeddings_data(filename: str) -> pd.DataFrame:
    """
    Load dataset from pretrained embeddings from Van Toor paper, and update it as follows:
        - 'target' column is updated to 'age' to allow multiple possible target labels.
        - 'age_group' column is added for classification task.

    :param filename: name of file in `EMBEDDINGS_DIR` containing dataset
    :return: pandas DataFrame containing embeddings
    """
    df = pd.read_csv(EMBEDDINGS_DIR / filename)
    df.rename(columns={'target': 'age'}, inplace=True)
    df['age_group'] = df['age'].apply(_age_to_age_group)
    return df


# TODO: Implement load_audio_data() in a new audio_loading.py module. It should:
#  - load raw .wav files from AUDIO_DIR,
#  - resample to 96kHz (max sample rate in the dataset),
#  - generate spectrograms via spectrogram.py,
#  - extract embeddings via cnn_embeddings.py,
#  - return a DataFrame in the same format as load_embeddings_data().


def _age_to_age_group(age: float) -> str:
    if age <= 1:
        return AGE_GROUP_CATEGORIES[0]
    elif age <= 10:
        return AGE_GROUP_CATEGORIES[1]
    else:
        return AGE_GROUP_CATEGORIES[2]


def clean_embeddings_for_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove metadata and label columns other than 'age_group', remove rows with missing values, and map string age groups to integer indices.

    :param df: pandas DataFrame containing embeddings
    :return: new DataFrame with cleaned embeddings
    """
    label_col = 'age_group'
    new_df = _clean_embeddings(df, label_col)

    # encode age group as ints
    new_df['age_group'] = new_df['age_group'].apply(_age_group_str_to_index)
    return new_df


def _age_group_str_to_index(age_group: str) -> int:
    return AGE_GROUP_CATEGORIES.index(age_group)


def clean_embeddings_for_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove metadata and label columns other than 'age' and remove rows with missing values.

    :param df: pandas DataFrame containing embeddings
    :return: new DataFrame with cleaned embeddings
    """
    label_col = 'age'
    new_df = _clean_embeddings(df, label_col)
    return new_df


def clean_embeddings_for_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove metadata and label columns other than 'gender', remove rows with unknown gender or missing values, and map string genders to integer indices.

    :param df: pandas DataFrame containing embeddings
    :return: new DataFrame with cleaned embeddings
    """
    label_col = 'gender'
    new_df = _clean_embeddings(df, label_col)

    # remove unknown gender
    new_df = new_df[new_df['gender'] != GENDER_CATEGORIES[-1]]

    # encode gender as ints
    new_df['gender'] = new_df['gender'].apply(_gender_str_to_index)
    return new_df


def _gender_str_to_index(gender: str) -> int:
    return GENDER_CATEGORIES.index(gender)


def _clean_embeddings(df: DataFrame, label_col: str) -> Series | DataFrame | Any:
    # original features are numbered 0-N
    new_df = df.drop(columns=[col for col in df.columns if col not in [label_col, "cat_id"] and not col.isnumeric()])

    # remove any rows with missing values
    bad = new_df.isna().any(axis=1)
    new_df = new_df[~bad]
    return new_df
