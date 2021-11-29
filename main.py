# %%
from email_spam_detection.dataset.datamodule import EmailSpamDataModule

# %%
dm = EmailSpamDataModule(0)
dm.prepare_data()
# %%
tr = dm.train_dataloader()
# %%
next(iter(tr))
# %%
len(tr)
# %%
