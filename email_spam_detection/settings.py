RANDOM_SEED = 42


class LocationConfig:
    tmp_dir = '.tmp'
    raw_data_dir = 'data/raw'
    extracted_data_file = 'data/extracted.json'
    processed_data_file = 'data/processed.json'
    cv_indices_file = 'data/cv_indices.bin'


class ModelConfig:
    model = 'distilbert-base-uncased'


class DataConfig:
    class_label = 'spam'


class TrainingConfig:
    cv_splits = 5