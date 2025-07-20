# %%
import torch
ckpt =torch.load("../ckpts/hnet_2stage_XL.pt", map_location="cpu")
# %%
for k, v in ckpt.items():
    print(k, v.shape)