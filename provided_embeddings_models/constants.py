from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "feline-age-prediction" / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"
AUDIO_DIR = DATASET_DIR / "raw_audio"
# TODO: Add spectrogram generation constants (e.g. SAMPLE_RATE, HOP_LENGTH, N_MELS) for use in new spectrogram.py module
# TODO: Add CNN model/checkpoint path constant(s) for use in a new cnn_embeddings.py module

# Precomputed datasets provided by paper
PERCH_FILENAME = "perch_looped_embeddings.csv"
VGGISH_FILENAME = "vggish_looped_embeddings.csv"
YAMNET_FILENAME = "yamnet_looped_embeddings.csv"

RANDOM_STATE = 42
BATCH_SIZE = 32

AGE_GROUP_CATEGORIES = ['Kitten', 'Adult', 'Senior']
NUM_AGE_GROUPS = len(AGE_GROUP_CATEGORIES)

GENDER_CATEGORIES = ['M', 'F', 'X']

# Output models
MODEL_DIR = Path(__file__).parent / "models"
