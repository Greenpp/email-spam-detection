from pathlib import Path

import pandas as pd

from ..settings import LocationConfig


def extract_subjects() -> None:
    extracted_file = Path(LocationConfig.extracted_data_file)

    emails = pd.read_json(extracted_file)
    emails['subject'] = emails['text'].apply(
        lambda t: list(filter(lambda l: l.startswith('Subject:'), t.split('\n')))[0][9:]
    )
    subjects = emails.drop(columns=['text'])
    subjects.to_json(LocationConfig.subjects_data_file)
