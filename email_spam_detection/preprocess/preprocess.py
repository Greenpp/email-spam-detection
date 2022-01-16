import re

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from ..settings import RANDOM_SEED, LocationConfig


def preprocess() -> None:
    data = pd.read_json(LocationConfig.extracted_data_file)

    subjects = _extract_subjects(data)
    data['subject'] = subjects

    data['response'] = data['subject'].apply(lambda x: x.startswith('re :'))
    data['forward'] = data['subject'].apply(lambda x: x.startswith('fw :'))
    data['om'] = data['text'].apply(lambda x: '- original message -' in x)
    data['forward2'] = data['text'].apply(lambda x: '- forwarded by' in x)

    msgs = _extract_msgs(data)
    data['msg'] = msgs

    marker_pattern = re.compile(r'^.*?:')
    data['markers'] = data['msg'].apply(
        lambda m: all([re.match(marker_pattern, l) is None for l in m.split('\n')])
    )

    clean_data = data[
        ~data['response']
        & ~data['forward']
        & ~data['om']
        & ~data['forward2']
        & ~data['markers']
    ][['subject', 'msg', 'spam']]

    sampler = RandomUnderSampler(random_state=RANDOM_SEED)
    clean_resampled_data, _ = sampler.fit_resample(clean_data, clean_data['spam'])

    clean_resampled_data.to_json(LocationConfig.processed_data_file)


def _extract_subjects(data: pd.DataFrame) -> pd.Series:
    subjects = data['text'].apply(
        lambda t: list(filter(lambda l: l.startswith('Subject:'), t.split('\n')))[0][9:]
    )

    return subjects


def _extract_msgs(data: pd.DataFrame) -> pd.Series:
    msgs = data['text'].apply(
        lambda t: '\n'.join(
            filter(lambda l: not l.startswith('Subject:'), t.split('\n'))
        )
    )

    return msgs
