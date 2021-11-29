import torch
from transformers import AutoTokenizer, DataCollatorWithPadding

from ..settings import ModelConfig


class EmailSpamCollator:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(ModelConfig.model)
        self.collator = DataCollatorWithPadding(self.tokenizer)

    def __call__(self, batch):
        texts, labels = zip(*batch)

        batch = self.collator(
            self.tokenizer(list(texts), truncation=True, max_length=64)
        )
        batch['labels'] = torch.tensor(labels, dtype=torch.float)

        return batch
