import multiprocessing
import os

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger

from email_spam_detection.dataset.datamodule import EmailSpamDataModule
from email_spam_detection.model import EmailSpamDetector
from email_spam_detection.settings import (
    RANDOM_SEED,
    DataConfig,
    TrainingConfig,
)


class Worker:
    def __init__(
        self,
        queue: multiprocessing.Queue,
        process: multiprocessing.Process,
    ) -> None:
        self.queue = queue
        self.process = process


class WorkerData:
    def __init__(
        self,
        fold: int,
        sweep_id,
        sweep_run_name,
        config,
    ) -> None:
        self.fold = fold
        self.sweep_id = sweep_id
        self.sweep_run_name = sweep_run_name
        self.config = config


class DoneData:
    def __init__(
        self,
        validation_accuracy: float,
    ) -> None:
        self.validation_accuracy = validation_accuracy


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(sweep_q, worker_q):
    reset_wandb_env()

    worker_data: WorkerData = worker_q.get()

    pl.seed_everything(RANDOM_SEED)
    run_name = "{}-{}".format(worker_data.sweep_run_name, worker_data.fold)
    config = worker_data.config
    wandb.init(
        group=worker_data.sweep_id,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=config,
    )

    logger = WandbLogger()

    datamodule = EmailSpamDataModule(
        fold_num=worker_data.fold,
        input_label=wandb.config.input_label,
        batch_size=TrainingConfig.batch_size,
    )
    model = EmailSpamDetector()

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=TrainingConfig.epochs,
        logger=logger,
        log_every_n_steps=10,
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

    validation_accuracy = wandb.run.summary['validation_Accuracy']
    wandb.finish()
    sweep_q.put(DoneData(validation_accuracy))


if __name__ == '__main__':
    sweep_q = multiprocessing.Queue()
    workers: list[Worker] = []
    for fold in range(DataConfig.cv_splits):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=train, kwargs={'sweep_q': sweep_q, 'worker_q': q}
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    sweep_run = wandb.init()

    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    metrics = []
    for fold in range(DataConfig.cv_splits):
        worker = workers[fold]

        worker.queue.put(
            WorkerData(
                sweep_id=sweep_id,
                fold=fold,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
            )
        )

        result: DoneData = sweep_q.get()
        worker.process.join()
        metrics.append(result.validation_accuracy)

    sweep_run.log({'validation_accuracy': sum(metrics) / len(metrics)})
