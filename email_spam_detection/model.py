import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import AutoModelForSequenceClassification

from .settings import ModelConfig, TrainingConfig


class EmailSpamDetector(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            ModelConfig.model,
            num_labels=1,
            torch_dtype=torch.float,
        )

        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(),
                torchmetrics.Precision(),
                torchmetrics.Recall(),
                torchmetrics.F1(1),
            ]
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.validation_metrics = metrics.clone(prefix='validation_')

    def forward(self, **input):
        return self.model(**input)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        metrics_output = self.train_metrics(
            outputs.logits.squeeze(), batch['labels'].int()
        )
        self.log_dict(metrics_output, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss

        metrics_output = self.validation_metrics(
            outputs.logits.squeeze(), batch['labels'].int()
        )
        self.log_dict(metrics_output, on_step=False, on_epoch=True)
        self.log(
            'validation_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=TrainingConfig.learning_rate,
            betas=(
                TrainingConfig.adam_beta_1,
                TrainingConfig.adam_beta_2,
            ),
            weight_decay=TrainingConfig.l2_norm,
        )
