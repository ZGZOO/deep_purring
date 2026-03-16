"""
Load raw .wav files from the dataset, parse labels from filenames, and provide
a PyTorch Dataset that yields (mel-spectrogram, label) for training CNN+MLP.

Filename convention (from paper): e.g. 0.5Y-022A-F1-01.wav
  - age (years): 0.5
  - cat_id: 022A
  - gender: F (M/F/X)
"""

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from provided_embeddings_models.constants import (
    AUDIO_LOOPED_YAMNET_DIR,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    AGE_GROUP_CATEGORIES,
    GENDER_CATEGORIES,
)
from provided_embeddings_models.spectrogram import waveform_to_mel_spectrogram
from provided_embeddings_models.tasks import TaskType


def parse_wav_metadata(filename: str) -> Optional[dict]:
    """
    Parse age and gender from a wav filename.
    Example: 0.5Y-022A-F1-01.wav -> age=0.5, gender='F'

    :param filename: e.g. 0.5Y-022A-F1-01.wav
    :return: dict with keys age (float), gender (str), age_group (str), or None if parse fails
    """
    # Match: digits.digitsY or digitsY at start, then later M, F, or X before .wav
    age_match = re.match(r"^([\d.]+)Y", filename)
    gender_match = re.search(r"([MFX])(?=[-\d.]*\.wav)", filename)
    if not age_match or not gender_match:
        return None
    age = float(age_match.group(1))
    gender = gender_match.group(1)
    age_group = _age_to_age_group_str(age)
    return {"age": age, "gender": gender, "age_group": age_group}


def _age_to_age_group_str(age: float) -> str:
    if age <= 1:
        return AGE_GROUP_CATEGORIES[0]
    elif age <= 10:
        return AGE_GROUP_CATEGORIES[1]
    return AGE_GROUP_CATEGORIES[2]


def load_wav(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load a .wav file and return a mono waveform at the given sample rate (Librosa).

    :param path: path to .wav file
    :param sample_rate: target sample rate (resampled if needed)
    :return: waveform numpy array shape (samples,), float32 in [-1, 1]
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required. Install with: pip install librosa")

    waveform, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return waveform.astype(np.float32)


def fix_audio_length(
    waveform: np.ndarray,
    target_samples: int = TARGET_SAMPLES,
) -> np.ndarray:
    """
    Bring waveform to fixed length for the CNN window.
    - Longer: truncate to target_samples.
    - Shorter: loop (repeat) the audio to fill the window instead of padding with silence.

    :param waveform: (samples,) float array
    :param target_samples: desired length in samples
    :return: (target_samples,) float array
    """
    n = len(waveform)
    if n >= target_samples:
        return waveform[:target_samples].copy()
    # Loop: repeat until at least target_samples, then take first target_samples
    repeats = (target_samples + n - 1) // n
    looped = np.tile(waveform, repeats)
    return looped[:target_samples].copy()


class AudioSpectrogramDataset(Dataset):
    """
    Dataset that yields (mel-spectrogram, label) for training.
    Labels are derived from wav filenames (age, gender, age_group).
    """

    def __init__(
        self,
        audio_dir: Path,
        task: TaskType,
        transform=None,
    ):
        """
        :param audio_dir: directory containing .wav files (e.g. AUDIO_LOOPED_YAMNET_DIR)
        :param task: 'age_group' | 'gender' | 'age' — which label to return
        :param transform: callable waveform -> spec; default is waveform_to_mel_spectrogram
        """
        self.audio_dir = Path(audio_dir)
        self.task = task
        self.transform = transform or waveform_to_mel_spectrogram

        self.samples: list[Tuple[Path, dict]] = []
        for path in sorted(self.audio_dir.glob("*.wav")):
            meta = parse_wav_metadata(path.name)
            if meta is None:
                continue
            if task == "gender" and meta["gender"] == "X":
                continue
            self.samples.append((path, meta))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, meta = self.samples[idx]
        waveform = load_wav(path)
        waveform = fix_audio_length(waveform)
        spec = self.transform(waveform)

        if self.task == "age_group":
            label = AGE_GROUP_CATEGORIES.index(meta["age_group"])
        elif self.task == "gender":
            label = GENDER_CATEGORIES.index(meta["gender"])
        else:
            label = float(meta["age"])

        label_t = torch.tensor(label, dtype=torch.long if self.task != "age" else torch.float32)
        return spec, label_t


def load_audio_data(
    audio_dir: Optional[Path] = None,
    cnn_module=None,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Load raw .wav files, compute mel-spectrograms, run through the provided CNN,
    and return a DataFrame in the same format as load_embeddings_data() (columns
    0..embed_dim-1, age, age_group, gender) so the rest of the pipeline can be reused.

    Use this when you have a trained/frozen CNN and want to precompute embeddings
    to train the MLP with the existing train_model_main().

    :param audio_dir: directory of .wav files; default AUDIO_LOOPED_YAMNET_DIR
    :param cnn_module: CatEmbeddingCNN (or callable) in eval mode; if None, embeddings are not computed (not implemented).
    :param device: device to run CNN on
    :return: DataFrame with numeric embedding columns + age, age_group, gender
    """
    if audio_dir is None:
        audio_dir = AUDIO_LOOPED_YAMNET_DIR
    if cnn_module is None:
        raise ValueError("load_audio_data requires a trained CNN to compute embeddings")

    cnn_module = cnn_module.eval()
    if device is None:
        device = next(cnn_module.parameters()).device

    rows = []
    for path in sorted(Path(audio_dir).glob("*.wav")):
        meta = parse_wav_metadata(path.name)
        if meta is None or meta["gender"] == "X":
            continue
        waveform = load_wav(path)
        waveform = fix_audio_length(waveform)
        spec = waveform_to_mel_spectrogram(waveform).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = cnn_module(spec).squeeze(0).cpu().numpy()
        row = {i: emb[i] for i in range(len(emb))}
        row["age"] = meta["age"]
        row["age_group"] = meta["age_group"]
        row["gender"] = meta["gender"]
        rows.append(row)

    df = pd.DataFrame(rows)
    # Column names 0,1,...,embed_dim-1 for compatibility with clean_embeddings_*
    return df
