import pytorch_lightning as pl
from email_spam_detection.dataset.datamodule import EmailSpamDataModule
from email_spam_detection.model import EmailSpamDetector
from email_spam_detection.settings import (
    RANDOM_SEED,
    TrainingConfig,
    WandbConfig,
)
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger

if __name__ == '__main__':
    pl.seed_everything(RANDOM_SEED)

    datamodule = EmailSpamDataModule(
        fold_num=TrainingConfig.cv_fold,
        input_label='msg',
        batch_size=TrainingConfig.batch_size,
    )
    model = EmailSpamDetector()

    logger = WandbLogger(
        project=WandbConfig.project_name,
        name=WandbConfig.run_name,
        group=WandbConfig.group_name,
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=TrainingConfig.epochs,
        logger=logger,
        precision=16,
        callbacks=[
            StochasticWeightAveraging(),
            ModelCheckpoint(save_top_k=0),
        ],
    )
    trainer.fit(
        model,
        datamodule=datamodule,
    )
