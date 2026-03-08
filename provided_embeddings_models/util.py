import pandas as pd

from provided_embeddings_models.constants import EMBEDDINGS_DIR, AGE_GROUP_CATEGORIES


def load_embeddings_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(EMBEDDINGS_DIR / filename)
    df.rename(columns={'target': 'age'}, inplace=True)
    df['age_group'] = df['age'].apply(_age_to_age_group)
    return df


def _age_to_age_group(age: float) -> str:
    if age <= 1:
        return AGE_GROUP_CATEGORIES[0]
    elif age <= 10:
        return AGE_GROUP_CATEGORIES[1]
    else:
        return AGE_GROUP_CATEGORIES[2]
