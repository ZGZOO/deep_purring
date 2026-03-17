from __future__ import annotations

import os
import uuid
from pathlib import Path

import librosa
import numpy as np
import torch
from flask import Flask, render_template, request, url_for
from torch import nn
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get(
        "BRANCH_CNN_PATH",
        str(BASE_DIR / "saved_models" / "small_cnn_seed42.pt"),
    )
)
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 2.0
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024
AGE_GROUP_LABELS = ["Kitten", "Adult", "Senior"]
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def load_branch_model() -> SmallCNN | None:
    if not MODEL_PATH.exists():
        return None

    branch_model = SmallCNN()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    branch_model.load_state_dict(state_dict)
    branch_model.to(DEVICE)
    branch_model.eval()
    return branch_model


model = load_branch_model()


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def load_audio_fixed(path: Path) -> np.ndarray:
    target_len = int(SAMPLE_RATE * DURATION)
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")

    return y


def audio_to_mel(y: np.ndarray) -> np.ndarray:
    spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mel = librosa.power_to_db(spectrogram, ref=np.max).astype(np.float32)
    return (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)


def predict_age_group(audio_path: Path) -> tuple[str, int, list[float]]:
    if model is None:
        raise FileNotFoundError(
            f"Branch CNN checkpoint not found at {MODEL_PATH}. "
            "Place small_cnn_seed42.pt there or set BRANCH_CNN_PATH."
        )

    y = load_audio_fixed(audio_path)
    mel = audio_to_mel(y)
    features = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(features).squeeze(0)
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()

    class_idx = int(logits.argmax().item())
    label = AGE_GROUP_LABELS[class_idx]
    return label, class_idx, probabilities.tolist()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        audio_file = request.files.get("audio")

        if audio_file is None or audio_file.filename == "":
            error = "Choose an audio file or record one first."
        elif not allowed_file(audio_file.filename):
            error = "Unsupported file type. Use wav, mp3, ogg, flac, m4a, or webm."
        else:
            safe_name = secure_filename(audio_file.filename)
            suffix = Path(safe_name).suffix.lower() or ".wav"
            saved_name = f"{uuid.uuid4().hex}{suffix}"
            save_path = UPLOAD_DIR / saved_name
            audio_file.save(save_path)

            try:
                label, class_idx, probabilities = predict_age_group(save_path)
                result = {
                    "label": label,
                    "class_idx": class_idx,
                    "model_name": MODEL_PATH.name,
                    "probabilities": [
                        {
                            "label": class_label,
                            "value": float(score),
                            "percent": round(float(score) * 100, 1),
                        }
                        for class_label, score in zip(AGE_GROUP_LABELS, probabilities)
                    ],
                    "audio_url": url_for("static", filename=f"uploads/{saved_name}"),
                    "filename": safe_name,
                }
            except Exception as exc:
                error = f"Prediction failed: {exc}"

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
