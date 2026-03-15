"""
Train the full pipeline from raw audio: Spectrogram -> CNN -> Embedding -> MLP (Route A).

Uses the same task configs and CatMLP head as main.py, but the input is
mel-spectrogram from our own audio loading (paper's wav files). The CNN is
trained jointly with the MLP.
"""

from typing import Literal

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split

from provided_embeddings_models.audio_loading import (
    AudioSpectrogramDataset,
    AUDIO_LOOPED_YAMNET_DIR,
)
from provided_embeddings_models.constants import (
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


def main():
    TASK: TaskType = "age_group"
    task_config = TASK_CONFIGS[TASK]
    es_mode: Literal["max", "min"] = "max" if task_config.direction == "maximize" else "min"

    dataset = AudioSpectrogramDataset(AUDIO_LOOPED_YAMNET_DIR, task=TASK)
    n = len(dataset)
    train_len = int(0.8 * n)
    val_len = int(0.1 * n)
    test_len = n - train_len - val_len
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

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
    main()
