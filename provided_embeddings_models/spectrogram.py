"""
Convert raw audio waveform to mel-spectrogram for use by our CNN.

Uses Librosa (main audio processing library) for loading and mel-spectrogram.
Output is PyTorch tensor so the rest of the pipeline (CNN) stays on GPU if needed.
"""

from typing import Union

import numpy as np
import torch

from provided_embeddings_models.constants import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
)


def waveform_to_mel_spectrogram(
    waveform: Union[torch.Tensor, np.ndarray],
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> torch.Tensor:
    """
    Convert a mono waveform to a mel-spectrogram (log-mel) using Librosa.

    :param waveform: shape (samples,) or (batch, samples); float in [-1, 1]
    :param sample_rate: sample rate of the waveform
    :param n_fft: FFT window size
    :param hop_length: hop length between frames
    :param n_mels: number of mel filterbanks
    :return: mel-spectrogram tensor, shape (n_mels, time) or (batch, n_mels, time)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for spectrogram. Install with: pip install librosa")

    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    single = waveform.ndim == 1
    if single:
        waveform = waveform[np.newaxis, :]  # (1, samples)

    out = []
    for i in range(waveform.shape[0]):
        mel = librosa.feature.melspectrogram(
            y=waveform[i],
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-9)
        out.append(log_mel)

    log_mel = np.stack(out).astype(np.float32)
    log_mel_t = torch.from_numpy(log_mel)

    if single:
        log_mel_t = log_mel_t.squeeze(0)  # (n_mels, time)
    return log_mel_t
