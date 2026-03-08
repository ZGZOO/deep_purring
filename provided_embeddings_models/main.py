from constants import *
from util import *

# load dataset into pandas from csv
data = load_embeddings_data(VGGISH_FILENAME)
print(data.head())
