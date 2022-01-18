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
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, **input):
        return self.model(**input)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        self.train_acc(outputs.logits.squeeze(), batch['labels'].int())
        self.log(
            'train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss

        self.val_acc(outputs.logits.squeeze(), batch['labels'].int())
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

        return {'val_loss': val_loss}

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
