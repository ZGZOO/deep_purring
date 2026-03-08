from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "feline-age-prediction" / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"
AUDIO_DIR = DATASET_DIR / "raw_audio"

PERCH_FILENAME = "perch_looped_embeddings.csv"
VGGISH_FILENAME = "vggish_looped_embeddings.csv"
YAMNET_FILENAME = "yamnet_looped_embeddings.csv"

RANDOM_SEED = 42