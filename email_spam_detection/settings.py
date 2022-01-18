RANDOM_SEED = 42


class LocationConfig:
    tmp_dir = '.tmp'
    raw_data_dir = 'data/raw'
    extracted_data_file = 'data/extracted.json'
    processed_data_file = 'data/processed.json'
    cv_indices_file = 'data/cv_indices.bin'


class ModelConfig:
    model = 'distilbert-base-uncased'
    # model = 'xlnet-base-cased'


class DataConfig:
    class_label = 'spam'
    cv_splits = 5


class TrainingConfig:
    learning_rate = 10e-5
    epochs = 10
    batch_size = 128
    cv_fold = 0

    adam_beta_1 = 0.9
    adam_beta_2 = 0.999

    l2_norm = 0.0


class WandbConfig:
    project_name = 'email-spam-detection'
    run_name = f'beta1={TrainingConfig.adam_beta_1}:beta2={TrainingConfig.adam_beta_2}:l2={TrainingConfig.l2_norm}:fold={TrainingConfig.cv_fold}'
    group_name = f'beta1={TrainingConfig.adam_beta_1}:beta2={TrainingConfig.adam_beta_2}:l2={TrainingConfig.l2_norm}'
