from typing import Tuple

from constants import *
from provided_embeddings_models.preprocessing import split_embeddings, build_preprocessing_pipeline
from util import *

# load dataset into pandas from csv
data = load_embeddings_data(YAMNET_FILENAME)
print(data.head())


def get_age_group_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    age_group_cleaned = clean_embeddings_for_age_group(data)
    # print(age_group_cleaned.head())
    return split_embeddings(age_group_cleaned, 'age_group', val_size=0.12, regression=False)


def get_age_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    age_cleaned = clean_embeddings_for_age(data)
    # print(age_cleaned.head())
    return split_embeddings(age_cleaned, 'age', val_size=0.12, regression=True)


def get_sex_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sex_cleaned = clean_embeddings_for_sex(data)
    # print(sex_cleaned.head())
    return split_embeddings(sex_cleaned, 'sex', val_size=0.12, regression=False)


# age group isolated
train_features, val_features, test_features, train_labels, val_labels, test_labels = get_age_group_split()

# age (continuous) isolated
# train_features, val_features, test_features, train_labels, val_labels, test_labels = get_age_split()

# sex (M/F) isolated
# train_features, val_features, test_features, train_labels, val_labels, test_labels = get_sex_split()

print(f'Train features: {train_features.shape}')
print(f'Train labels: {train_labels.shape}')
print(f'Val features: {val_features.shape}')
print(f'Val labels: {val_labels.shape}')
print(f'Test features: {test_features.shape}')
print(f'Test labels: {test_labels.shape}')

# preprocess features
pipeline = build_preprocessing_pipeline(n_components=32)
train_features_processed = pipeline.fit_transform(train_features)
val_features_processed = pipeline.transform(val_features)
test_features_processed = pipeline.transform(test_features)
print(f'Train features range: {train_features_processed.min():.3f} - {train_features_processed.max():.3f}')
print(f'Val features range: {val_features_processed.min():.3f} - {val_features_processed.max():.3f}')
print(f'Test features range: {test_features_processed.min():.3f} - {test_features_processed.max():.3f}')
print(f'Train features pca: {train_features_processed.shape}')
print(f'Val features pca: {val_features_processed.shape}')
print(f'Test features pca: {test_features_processed.shape}')
