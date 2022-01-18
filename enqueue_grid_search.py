import itertools
import subprocess

params = [
    [0.9, 0.85, 0.8],  # beta 1
    [0.999, 0.99, 0.9],  # beta 2
    [0.0, 0.05, 0.1],  # l2
    [0, 1, 2, 3, 4],  # fold
]
for b1, b2, l2, fold in itertools.product(*params.values()):
    subprocess.run(
        [
            'dvc',
            'exp',
            'run',
            '--queue',
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
