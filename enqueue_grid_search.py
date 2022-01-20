import itertools
import subprocess

params = [
    ['subject', 'msg', 'text'],  # input label
    [0.9, 0.8, 0.7],  # beta 1
    [0.999, 0.99, 0.9],  # beta 2
    [0.0, 0.1, 0.2],  # l2
    [0, 1, 2, 3, 4],  # fold
]
for lab, b1, b2, l2, fold in itertools.product(*params):
    subprocess.run(
        [
            'dvc',
            'exp',
            'run',
            '--queue',
            '--set-param',
            f'email_spam_detection/settings.py:TrainingConfig.input_label={lab}',
            '--set-param',
            f'email_spam_detection/settings.py:TrainingConfig.adam_beta_1={b1}',
            '--set-param',
            f'email_spam_detection/settings.py:TrainingConfig.adam_beta_2={b2}',
            '--set-param',
            f'email_spam_detection/settings.py:TrainingConfig.l2_norm={l2}',
            '--set-param',
            f'email_spam_detection/settings.py:TrainingConfig.cv_fold={fold}',
        ]
    )
