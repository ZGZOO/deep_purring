# Route A: Our spectrogram + CNN + MLP (from raw audio)

This pipeline uses **our own** spectrogram and CNN on the paper's raw wav files, then feeds the embedding into Brian's CatMLP. No paper embeddings (YAMNet/VGGish/Perch) are used.

## What was added (branch `jenny-spectrogram-cnn-embeddings`)

| File | Purpose |
|------|--------|
| `spectrogram.py` | Waveform → mel-spectrogram (torchaudio) |
| `cnn_embeddings.py` | `CatEmbeddingCNN`: spectrogram → embedding (128-dim) |
| `audio_loading.py` | Load wav, parse labels from filename, `AudioSpectrogramDataset` |
| `main_audio.py` | Train CNN + MLP end-to-end from raw audio |
| `constants.py` | Added `SAMPLE_RATE`, `N_FFT`, `HOP_LENGTH`, `N_MELS`, `EMBED_DIM`, `AUDIO_LOOPED_YAMNET_DIR` |

## How to run

1. Install deps (adds `torchaudio`):  
   `pip install -r requirements.txt`

2. Ensure the dataset is present:  
   `feline-age-prediction/dataset/raw_audio/AudioLoopedYAMNet/` should contain `.wav` files.

3. Train from raw audio (age_group by default):  
   `python -m provided_embeddings_models.main_audio`

   This will:
   - Load wavs from `AudioLoopedYAMNet`
   - Compute mel-spectrograms
   - Train CNN + MLP (Brian's) end-to-end
   - Save the CNN to `provided_embeddings_models/models/cnn_embedding.pt`

4. To use a different task (`gender` or `age`), edit `TASK` in `main_audio.py`.

## Optional: precompute embeddings and use existing `main.py`

After training the CNN, you can precompute embeddings for all wavs and then run the usual MLP training (with PCA, Optuna, etc.):

```python
from provided_embeddings_models.audio_loading import load_audio_data
from provided_embeddings_models.cnn_embeddings import CatEmbeddingCNN
from provided_embeddings_models.constants import CNN_CHECKPOINT_DIR, CNN_CHECKPOINT_FILENAME
import torch

cnn = CatEmbeddingCNN()
cnn.load_state_dict(torch.load(CNN_CHECKPOINT_DIR / CNN_CHECKPOINT_FILENAME))
cnn.eval()
df = load_audio_data(cnn_module=cnn)
# Then use df with clean_embeddings_for_* and train_model_main() from main.py
```
