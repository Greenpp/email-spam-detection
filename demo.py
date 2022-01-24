import numpy as np
from gradio import Interface, outputs
from transformers import AutoTokenizer

from email_spam_detection.model import EmailSpamDetector
from email_spam_detection.settings import ModelConfig

tokenizer = AutoTokenizer.from_pretrained(ModelConfig.model)
model = EmailSpamDetector.load_from_checkpoint('./best-model/epoch=9-step=339.ckpt')
model.eval()


def detect_spam(msg: str) -> dict:
    encoded_msg = tokenizer.encode(
        msg,
        truncation=True,
        max_length=64,
        return_tensors='pt',
    )

    pred = model.model.forward(encoded_msg)
    is_spam_logit = pred['logits'].item()
    is_spam = max(0.0, min(1.0, is_spam_logit))
    is_ham = 1.0 - is_spam

    return {'spam': is_spam, 'ham': is_ham}


Interface(detect_spam, 'text', outputs.Label()).launch()
