import pytorch_lightning as pl
import wandb
from email_spam_detection.dataset.datamodule import EmailSpamDataModule
from email_spam_detection.model import EmailSpamDetector
from email_spam_detection.settings import RANDOM_SEED
from pytorch_lightning.loggers import WandbLogger

if __name__ == '__main__':
    pl.seed_everything(RANDOM_SEED)

    datamodule = EmailSpamDataModule(0)
    model = EmailSpamDetector()

    logger = WandbLogger(
        project='email-spam-detection',
        name='training-test-1'
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=2,
        logger=logger,
    )
    trainer.fit(
        model,
        datamodule=datamodule,
    )
