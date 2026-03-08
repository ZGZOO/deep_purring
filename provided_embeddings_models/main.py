from constants import *
from util import *

# load dataset into pandas from csv
data = load_embeddings_data(VGGISH_FILENAME)
print(data.head())

# age group isolated
age_group_cleaned = clean_embeddings_for_age_group(data)
print(age_group_cleaned.head())

# age (continuous) isolated
age_cleaned = clean_embeddings_for_age(data)
print(age_cleaned.head())

# sex (M/F) isolated
sex_cleaned = clean_embeddings_for_sex(data)
print(sex_cleaned.head())
