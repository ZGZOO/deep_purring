import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import F1Score, MeanAbsoluteError

from provided_embeddings_models.tasks import TaskType


class CatMLP(pl.LightningModule):
    def __init__(
            self,
            input_size: int,
            hidden_sizes: list,
            output_size: int,
            lr: float,
            task: TaskType,
    ):
        """
        Lightning module for MLP to predict cats' age group, gender, or age from input embeddings of their meows.

        :param input_size: input embedding size
        :param hidden_sizes: hidden layer sizes
        :param output_size: output embedding size
        :param lr: learning rate
        :param task: task type for the model ('age', 'gender', 'age_group')
        """
        super().__init__()
        self.lr = lr
        self.task = task

        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)

        if task == "age":
            self.loss_fn = nn.MSELoss()
            self.metric = MeanAbsoluteError()
        elif task == "age_group":
            self.loss_fn = nn.CrossEntropyLoss()
            self.metric = F1Score(task="multiclass", num_classes=3, average="macro")
        elif task == "gender":
            self.loss_fn = nn.CrossEntropyLoss()
            self.metric = F1Score(task="binary", average="macro")
        else:
            raise NotImplementedError(f'Task {task} is not implemented')

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.task == "age":
            loss = self.loss_fn(logits.squeeze(-1), y)
        else:
            loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.task == "age":
            loss = self.loss_fn(logits.squeeze(-1), y)
            self.metric.update(logits.squeeze(-1), y)
        else:
            loss = self.loss_fn(logits, y)
            preds = logits.argmax(dim=1)
            self.metric.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_metric", self.metric, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.task == "age":
            self.metric.update(logits.squeeze(-1), y)
        else:
            preds = logits.argmax(dim=1)
            self.metric.update(preds, y)
        self.log("test_metric", self.metric, on_epoch=True, on_step=False)
        # TODO: Accumulate preds and targets across batches (e.g., self.test_preds/self.test_targets lists) to enable
        #  epoch-level metric computation in on_test_epoch_end().

    # TODO: Implement on_test_epoch_end() to compute and log extended metrics from accumulated preds/targets:
    #   - age_group / gender: sklearn.metrics.classification_report (full per-class precision, recall, F1)
    #   - age: RMSE (sqrt of MSE) and QWK (sklearn.metrics.cohen_kappa_score with weights='quadratic', applied to
    #          rounded integer predictions)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
