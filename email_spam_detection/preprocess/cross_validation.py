import pickle as pkl

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..settings import RANDOM_SEED, DataConfig, LocationConfig


def generate_cross_validation_labels():
    processed_data = pd.read_json(LocationConfig.processed_data_file)

    skf = StratifiedKFold(
        DataConfig.cv_splits,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    indices = list(
        skf.split(
            processed_data,
            processed_data[DataConfig.class_label],
        )
    )

    with open(LocationConfig.cv_indices_file, 'wb') as f:
        pkl.dump(indices, f, pkl.HIGHEST_PROTOCOL)
