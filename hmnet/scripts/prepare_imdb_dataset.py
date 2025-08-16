# %%
import os
import torch
from hmnet.models.tokenizer import ByteTokenizer
import pandas as pd
import kagglehub
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence

max_length = 300
# Download latest version
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
# %%
df = pd.read_csv(
    path + "/IMDB Dataset.csv",
    encoding="utf-8",
)
df.columns = ["text", "label"]  # Ensure correct column names
# %%
tokenizer = ByteTokenizer()

df["text"] = df["text"].apply(lambda x: x[-max_length:] if len(x) > max_length else x)
# Encode
input_ids = tokenizer.encode(
    df["text"].tolist(), add_bos=False, add_eos=True, add_cls=True
)
input_ids = [torch.tensor(ids["input_ids"]) for ids in input_ids]
assert not any(
    (0 in ids for ids in input_ids)
), "Warning: Some input_ids contain 0, which may indicate padding or special tokens."


# input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
# mask = (input_ids != 0).long()  # Create mask for padding

# %%
labels = df["label"].apply(lambda x: 1 if x == "positive" else 0).to_list()

dataset_tensor = {
    "input_ids": input_ids,
    "labels": torch.tensor(labels, dtype=torch.uint8).unsqueeze(-1),
}

# %%
import os



# %%
os.chdir("/workspace")
torch.load("data/imdb_dataset.pt")
os.makedirs("data", exist_ok=True)
print(input_ids[:10])

# %%
