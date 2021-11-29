import pickle as pkl

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..settings import LocationConfig
from .dataloader import EmailSpamCollator
from .dataset import EmailSpamDataset


class EmailSpamDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fold_num: int,
        batch_size: int = 16,
        input_label: str = 'subject',
    ):
        self.input_label = input_label
        self.fold_num = fold_num
        self.batch_size = batch_size
        self.collator = EmailSpamCollator()

    def prepare_data(self) -> None:
        train_idx, test_idx = self._load_indices()

        self.train_dataset = EmailSpamDataset(train_idx, self.input_label)
        self.val_dataset = EmailSpamDataset(test_idx, self.input_label)

    def _load_indices(self) -> tuple[np.ndarray, np.ndarray]:
        with open(LocationConfig.cv_indices_file, 'rb') as f:
            indices = pkl.load(f)

        return indices[self.fold_num]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=4,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            collate_fn=self.collator,
            num_workers=4,
        )
