from pathlib import Path

import numpy as np
import pandas as pd
from email_spam_detection.settings import DataConfig, LocationConfig
from torch.utils.data import Dataset


class EmailSpamDataset(Dataset):
    def __init__(self, idx: np.ndarray, input_label: str) -> None:
        data = pd.read_json(LocationConfig.processed_data_file)

        self.X = data[input_label][idx].tolist()
        self.y = data[DataConfig.class_label][idx].tolist()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index) -> tuple[str, int]:
        return self.X[index], self.y[index]
