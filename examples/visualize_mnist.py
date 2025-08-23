# %%
import os
import torch

os.chdir("/workspace")
results = torch.load("results/imdb_predictions.pt")

results["dechunked_boundary_preds"] = results["dechunked_boundary_preds"].reshape(
    -1, 1, 7, 7
)
# results["dechunked_boundary_preds"] = (
#     results["dechunked_boundary_preds"]
#     .repeat_interleave(7, dim=2)
#     .repeat_interleave(7, dim=3)
# )

# %%
results["dechunked_boundary_preds"]


# %%
results["dechunked_attn_preds"]

# %%
