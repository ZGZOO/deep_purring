from constants import *
from provided_embeddings_models.preprocessing import split_embeddings, build_preprocessing_pipeline
from util import *

# load dataset into pandas from csv
data = load_embeddings_data(YAMNET_FILENAME)
# print(data.head())

# age group isolated
# age_group_cleaned = clean_embeddings_for_age_group(data)
# print(age_group_cleaned.head())

# age (continuous) isolated
age_cleaned = clean_embeddings_for_age(data)
# print(age_cleaned.head())

# sex (M/F) isolated
# sex_cleaned = clean_embeddings_for_sex(data)
# print(sex_cleaned.head())

# train/val/test split
(train_features, val_features, test_features,
 train_labels, val_labels, test_labels) = split_embeddings(age_cleaned, 'age', val_size=0.12, regression=True)
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
