# %%
import os
from pkg_resources import FileMetadata
from regex import F
import torch
from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hmnet.models.config_hmnet import HMNetConfig


def main(
    ckpt_path: str = "ckpts/hnet_1stage_L.pt",
):
    state_dict = torch.load(ckpt_path)
    return state_dict


# if __name__ == "__main__":
#     main()

# %%
os.chdir("/workspace")
state_dict = main()


# %%
for key in state_dict.keys():
    print(key)


# %%
import json
from hnet.models.config_hnet import HNetConfig
from hmnet.models.config_hmnet import HMNetConfig

with open("configs/hnet_1stage_L.json", "r") as f:
    config = json.load(f)
    hnet_config = HNetConfig(**config)

hnet_para = set(key.replace("backbone.", "") for key in state_dict.keys())

# %%
import yaml

with open("configs/hmnet_1stage_L.yaml", "r") as f:
    config = yaml.safe_load(f)
    hmnet_config = HMNetConfig.from_dict(config)

from hmnet.models.hmnet import HMNet

model = HMNet(hmnet_config)

# %% print HMNet Parameters
hmnet_para = set(name for name, _ in model.named_parameters())


# %%
hmnet_para - hnet_para
