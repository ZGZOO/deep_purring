# load dataset into pandas from csv
from constants import *
import pandas as pd

data = pd.read_csv(EMBEDDINGS_DIR / "vggish_looped_embeddings.csv")
print(data.head())
