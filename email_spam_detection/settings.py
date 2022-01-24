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
    cv_splits = 5


class TrainingConfig:
    input_label = 'msg'
    learning_rate = 0.00005839
    epochs = 10
    batch_size = 256
    cv_fold = 0

    adam_beta_1 = 0.9125
    adam_beta_2 = 0.996

    l2_norm = 1.230e-7


class WandbConfig:
    project_name = 'email-spam-detection'
    run_name = 'test-1'
    group_name = 'test-1'
