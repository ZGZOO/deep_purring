"""
Demo: predict cat age group from your own audio (one wav file).

Usage:
  python -m provided_embeddings_models.demo_audio path/to/your_cat_meow.wav

Uses the same pipeline as training: load wav -> fix length (truncate/loop) -> mel spectrogram -> CNN -> MLP -> age group.
"""

import argparse
import sys
from pathlib import Path

import torch

from provided_embeddings_models.audio_loading import load_wav, fix_audio_length
from provided_embeddings_models.constants import (
    MODEL_DIR,
    AGE_GROUP_CATEGORIES,
)
from provided_embeddings_models.main_audio import SpectrogramToLogits
from provided_embeddings_models.spectrogram import waveform_to_mel_spectrogram
from provided_embeddings_models.tasks import TASK_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Predict cat age group from a wav file.")
    parser.add_argument(
        "wav_path",
        type=Path,
        help="Path to your .wav file (e.g. your own cat meow recording).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / "age_group_spectrogram_model.pt",
        help="Path to saved full model (default: models/age_group_spectrogram_model.pt)",
    )
    args = parser.parse_args()

    wav_path = args.wav_path
    if not wav_path.exists():
        print(f"Error: file not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    # Load model
    model_path = args.model
    if not model_path.exists():
        print(
            f"Error: model not found: {model_path}\n"
            "Run 'python -m provided_embeddings_models.main_audio' first to train and save the model.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = SpectrogramToLogits("age_group", TASK_CONFIGS["age_group"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval()

    # Load your audio -> fix length -> spectrogram
    waveform = load_wav(wav_path)
    waveform = fix_audio_length(waveform)
    spec = waveform_to_mel_spectrogram(waveform).unsqueeze(0)  # (1, n_mels, time)

    # Predict
    with torch.no_grad():
        logits = model(spec).squeeze(0)
    probs = torch.softmax(logits, dim=0)
    pred_idx = int(logits.argmax().item())
    pred_label = AGE_GROUP_CATEGORIES[pred_idx]

    print(f"Input: {wav_path}")
    print(f"Predicted age group: {pred_label}")
    print("Probabilities:")
    for name, p in zip(AGE_GROUP_CATEGORIES, probs.tolist()):
        print(f"  {name}: {p:.2%}")


if __name__ == "__main__":
    main()
