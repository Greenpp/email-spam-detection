import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    StochasticWeightAveraging,
)

from email_spam_detection.dataset.datamodule import EmailSpamDataModule
from email_spam_detection.model import EmailSpamDetector
from email_spam_detection.settings import RANDOM_SEED, TrainingConfig

pl.seed_everything(RANDOM_SEED)
datamodule = EmailSpamDataModule(
    fold_num=TrainingConfig.cv_fold,
    input_label=TrainingConfig.input_label,
    batch_size=TrainingConfig.batch_size,
)
model = EmailSpamDetector()

trainer = pl.Trainer(
    gpus=1,
    max_epochs=TrainingConfig.epochs,
    log_every_n_steps=10,
    precision=16,
    callbacks=[
        StochasticWeightAveraging(),
        ModelCheckpoint(save_top_k=1),
    ],
)
trainer.fit(
    model,
    datamodule=datamodule,
)
