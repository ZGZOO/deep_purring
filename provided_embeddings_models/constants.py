from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "feline-age-prediction" / "dataset"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"
AUDIO_DIR = DATASET_DIR / "raw_audio"
# Subfolder for YAMNet-style looped audio (used for our spectrogram+CNN pipeline)
AUDIO_LOOPED_YAMNET_DIR = AUDIO_DIR / "AudioLoopedYAMNet"

# Spectrogram generation (for spectrogram.py)
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64

# Fixed window length for CNN (longer -> truncate, shorter -> loop to fit)
TARGET_DURATION_SEC = 1.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION_SEC)

# Our CNN embedding output dimension (feeds into CatMLP)
EMBED_DIM = 128

# CNN checkpoint for deployment / precomputed embeddings (optional)
CNN_CHECKPOINT_DIR = Path(__file__).parent / "models"
CNN_CHECKPOINT_FILENAME = "cnn_embedding.pt"

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
