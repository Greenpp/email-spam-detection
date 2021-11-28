import itertools
import shutil
import tarfile
from pathlib import Path
from typing import Generator

import pandas as pd
from tqdm import tqdm

from ..settings import LocationConfig


def _get_tmp_dir_path() -> Path:
    tmp_dir = Path(LocationConfig.tmp_dir)

    return tmp_dir


def create_tmp_dir() -> None:
    tmp_dir = _get_tmp_dir_path()
    tmp_dir.mkdir(parents=True, exist_ok=True)


def cleanup() -> None:
    tmp_dir = _get_tmp_dir_path()
    shutil.rmtree(tmp_dir)


def extract_data() -> None:
    raw_data_dir = Path(LocationConfig.raw_data_dir)
    tmp_dir = _get_tmp_dir_path()

    for file in raw_data_dir.glob('*.tar.gz'):
        with tarfile.open(file, 'r:gz') as t:
            t.extractall(tmp_dir)


def _get_spam_files() -> Generator[Path, None, None]:
    tmp_dir = _get_tmp_dir_path()

    return tmp_dir.glob('**/spam/*.txt')


def _get_ham_files() -> Generator[Path, None, None]:
    tmp_dir = _get_tmp_dir_path()

    return tmp_dir.glob('**/ham/*.txt')


def merge_data() -> None:
    spam_files = list(_get_spam_files())
    ham_files = list(_get_ham_files())

    texts = []
    is_spam = []

    files_with_labels = itertools.chain(
        zip(itertools.repeat(0), ham_files),
        zip(itertools.repeat(1), spam_files),
    )

    for label, file in tqdm(
        files_with_labels,
        total=(len(spam_files) + len(ham_files)),
    ):
        with open(file, encoding='latin1') as f:
            try:
                text = f.read()
            except Exception as e:
                print(file)
                raise e

        texts.append(text)
        is_spam.append(label)

    df = pd.DataFrame(
        {
            'text': texts,
            'spam': is_spam,
        }
    )
    df.to_json(LocationConfig.extracted_data_file)
