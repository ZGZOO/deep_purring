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
        self.test_preds: list = []
        self.test_targets: list = []

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
            self.test_preds.append(logits.squeeze(-1).detach().cpu())
        else:
            preds = logits.argmax(dim=1)
            self.metric.update(preds, y)
            self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        self.log("test_metric", self.metric, on_epoch=True, on_step=False)

    def on_test_epoch_end(self):
        from sklearn.metrics import classification_report, cohen_kappa_score, root_mean_squared_error

        all_preds = torch.cat(self.test_preds).numpy()
        all_targets = torch.cat(self.test_targets).numpy()

        if self.task == "age_group":
            from provided_embeddings_models.constants import AGE_GROUP_CATEGORIES
            print(classification_report(all_targets, all_preds, target_names=AGE_GROUP_CATEGORIES))
        elif self.task == "gender":
            from provided_embeddings_models.constants import GENDER_CATEGORIES
            print(classification_report(all_targets, all_preds, target_names=GENDER_CATEGORIES[:-1]))
        elif self.task == "age":
            rmse = root_mean_squared_error(all_targets, all_preds)
            qwk = cohen_kappa_score(
                all_targets.round().astype(int),
                all_preds.round().astype(int),
                weights="quadratic",
            )
            print(f"RMSE: {rmse:.4f}  QWK: {qwk:.4f}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
