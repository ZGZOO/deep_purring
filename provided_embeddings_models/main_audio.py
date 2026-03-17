"""
Train the full pipeline from raw audio: Spectrogram -> CNN -> Embedding -> MLP (Route A).

Uses the same task configs and CatMLP head as main.py, but the input is
mel-spectrogram from our own audio loading (paper's wav files). The CNN is
trained jointly with the MLP.

Run test set only (no training):
  python -m provided_embeddings_models.main_audio --test-only
"""

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from provided_embeddings_models.audio_loading import (
    AudioSpectrogramDataset,
    AUDIO_LOOPED_YAMNET_DIR,
)
from provided_embeddings_models.constants import (
    AGE_GROUP_CATEGORIES,
    RANDOM_STATE,
    BATCH_SIZE,
    EMBED_DIM,
    MODEL_DIR,
)
from provided_embeddings_models.cnn_embeddings import CatEmbeddingCNN
from provided_embeddings_models.lightning_model import CatMLP
from provided_embeddings_models.tasks import TaskType, TASK_CONFIGS, TaskConfig

pl.seed_everything(RANDOM_STATE, workers=True)


class SpectrogramToLogits(pl.LightningModule):
    """
    Combines our CNN (spectrogram -> embedding) with Brian's CatMLP (embedding -> logits).
    Trained end-to-end. Step logic and self.log() live here so Trainer manages this module.
    """

    def __init__(self, task: TaskType, cfg: TaskConfig):
        super().__init__()
        self.task = task
        self.cfg = cfg
        self.lr = 1e-3
        self.cnn = CatEmbeddingCNN()
        self.mlp = CatMLP(
            input_size=EMBED_DIM,
            hidden_sizes=[128, 128],
            output_size=cfg.output_size,
            lr=self.lr,
            task=task,
        )

    def forward(self, spec):
        emb = self.cnn(spec)
        return self.mlp(emb)

    def training_step(self, batch, batch_idx):
        spec, y = batch
        logits = self(spec)
        loss_fn = self.mlp.loss_fn
        if self.task == "age":
            loss = loss_fn(logits.squeeze(-1), y)
        else:
            loss = loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        spec, y = batch
        logits = self(spec)
        loss_fn = self.mlp.loss_fn
        metric = self.mlp.metric
        if self.task == "age":
            loss = loss_fn(logits.squeeze(-1), y)
            metric.update(logits.squeeze(-1), y)
        else:
            loss = loss_fn(logits, y)
            metric.update(logits.argmax(dim=1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_metric", metric, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        spec, y = batch
        logits = self(spec)
        metric = self.mlp.metric
        if self.task == "age":
            metric.update(logits.squeeze(-1), y)
        else:
            metric.update(logits.argmax(dim=1), y)
        self.log("test_metric", metric, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def _print_and_save_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print classification report + confusion matrix and save plot."""
    print("\n" + "=" * 60)
    print("age_group — Classification Report")
    print("=" * 60)
    print(
        classification_report(
            y_true, y_pred,
            target_names=AGE_GROUP_CATEGORIES,
            digits=2,
        )
    )
    cm = confusion_matrix(y_true, y_pred)
    print("age_group — Confusion Matrix (rows=actual, cols=predicted)")
    print(AGE_GROUP_CATEGORIES)
    print(cm)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=AGE_GROUP_CATEGORIES,
            yticklabels=AGE_GROUP_CATEGORIES,
            ax=ax, cbar_kws={"label": "count"},
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("age_group — Confusion Matrix")
        out_path = MODEL_DIR / "age_group_confusion_matrix.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"Saved confusion matrix plot to {out_path}")
    except Exception as e:
        print(f"Could not save plot (install matplotlib, seaborn for plot): {e}")


def run_test_only(model_path: Path | None = None) -> None:
    """Load saved model, run on the same cat_id-grouped test set, print report and save plot."""
    TASK: TaskType = "age_group"
    task_config = TASK_CONFIGS[TASK]

    if model_path is None:
        model_path = MODEL_DIR / "age_group_spectrogram_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train first: python -m provided_embeddings_models.main_audio"
        )

    # Same dataset and split as in main() so we evaluate on the same test set
    dataset = AudioSpectrogramDataset(AUDIO_LOOPED_YAMNET_DIR, task=TASK)
    n = len(dataset)
    groups = np.array([dataset.samples[i][1]["cat_id"] for i in range(n)])
    X_dummy = np.arange(n)
    y_dummy = np.zeros(n)

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_STATE)
    _, temp_idx = next(gss.split(X_dummy, y_dummy, groups=groups))
    gss2 = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=RANDOM_STATE + 1)
    _, test_idx_rel = next(
        gss2.split(X_dummy[temp_idx], y_dummy[temp_idx], groups=groups[temp_idx])
    )
    test_idx = temp_idx[test_idx_rel]
    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = SpectrogramToLogits(TASK, task_config)
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            spec, y = batch
            logits = model(spec)
            preds = logits.argmax(dim=1).numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    print(f"Test set size: {len(y_true)} samples")
    _print_and_save_report(y_true, y_pred)


def main():
    TASK: TaskType = "age_group"
    task_config = TASK_CONFIGS[TASK]
    es_mode: Literal["max", "min"] = "max" if task_config.direction == "maximize" else "min"

    dataset = AudioSpectrogramDataset(AUDIO_LOOPED_YAMNET_DIR, task=TASK)
    n = len(dataset)
    # Group by cat_id so the same cat is only in one of train/val/test
    groups = np.array([dataset.samples[i][1]["cat_id"] for i in range(n)])
    X_dummy = np.arange(n)
    y_dummy = np.zeros(n)

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss.split(X_dummy, y_dummy, groups=groups))
    gss2 = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=RANDOM_STATE + 1)
    val_idx_rel, test_idx_rel = next(
        gss2.split(
            X_dummy[temp_idx], y_dummy[temp_idx],
            groups=groups[temp_idx],
        )
    )
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = SpectrogramToLogits(TASK, task_config)

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[EarlyStopping(monitor="val_metric", patience=10, mode=es_mode)],
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # Collect test predictions and print confusion matrix + classification report (like teammate's)
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            spec, y = batch
            spec = spec.to(device)
            logits = model(spec)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    _print_and_save_report(y_true, y_pred)

    # Save CNN (for load_audio_data / precompute embeddings)
    ckpt_path = MODEL_DIR / "cnn_embedding.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cnn.state_dict(), ckpt_path)
    print(f"Saved CNN to {ckpt_path}")

    # Save full model (CNN+MLP) for demo: one wav -> age group prediction
    full_model_path = MODEL_DIR / "age_group_spectrogram_model.pt"
    torch.save(model.state_dict(), full_model_path)
    print(f"Saved full model for demo to {full_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run test set only.")
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Load saved model and run on test set only (no training). Prints classification report and confusion matrix.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to saved model .pt (default: models/age_group_spectrogram_model.pt). Used with --test-only.",
    )
    args = parser.parse_args()

    if args.test_only:
        run_test_only(model_path=args.model)
    else:
        main()
