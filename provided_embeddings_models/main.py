from constants import *
from provided_embeddings_models.training import split_embeddings
from util import *

# load dataset into pandas from csv
data = load_embeddings_data(VGGISH_FILENAME)
# print(data.head())

# age group isolated
age_group_cleaned = clean_embeddings_for_age_group(data)
# print(age_group_cleaned.head())

# age (continuous) isolated
# age_cleaned = clean_embeddings_for_age(data)
# print(age_cleaned.head())

# sex (M/F) isolated
# sex_cleaned = clean_embeddings_for_sex(data)
# print(sex_cleaned.head())

(train_features, val_features, test_features,
 train_labels, val_labels, test_labels) = split_embeddings(age_group_cleaned, 'age_group', val_size=0.12)
print(f'Train features: {train_features.shape}')
print(f'Train labels: {train_labels.shape}')
print(f'Val features: {val_features.shape}')
print(f'Val labels: {val_labels.shape}')
print(f'Test features: {test_features.shape}')
print(f'Test labels: {test_labels.shape}')
