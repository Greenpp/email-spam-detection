import pandas as pd

from ..settings import LocationConfig


def preprocess() -> None:
    data = pd.read_json(LocationConfig.extracted_data_file)

    subjects = _extract_subjects(data)
    data['subject'] = subjects

    data.to_json(LocationConfig.processed_data_file)


def _extract_subjects(data: pd.DataFrame) -> pd.Series:
    subjects = data['text'].apply(
        lambda t: list(filter(lambda l: l.startswith('Subject:'), t.split('\n')))[0][9:]
    )

    return subjects
