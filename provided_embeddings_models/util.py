import pandas as pd

from provided_embeddings_models.constants import EMBEDDINGS_DIR


def load_embeddings_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(EMBEDDINGS_DIR / filename)