# %% load hnet_2stage_XL
import torch
from hmnet.models.hmnet import HMNetForCausalLM
from hmnet.models.config_hmnet import HMNetConfig
from omegaconf import OmegaConf

hnet_2stage_XL_dict = torch.load(
    "/root/workspace/hmnet-archive/ckpts/hnet_2stage_XL.pt"
)


model_cfg = OmegaConf.load("/root/workspace/hmnet-archive/configs/hmnet_3stage_XL.yaml")
default_model_cfg = OmegaConf.structured(HMNetConfig)
merged_model_cfg = OmegaConf.merge(default_model_cfg, model_cfg)
hmnet_3stage_XL_config: HMNetConfig = OmegaConf.to_object(merged_model_cfg)

hmnet_3stage_XL = HMNetForCausalLM(config=hmnet_3stage_XL_config)

# %%
for key in hmnet_3stage_XL.state_dict().keys():
    print(key)
# %%
for key in hnet_2stage_XL_dict.keys():
    print(key)
# %%
main_network_str = "main_network."
hnet_2stage_XL_keys = hnet_2stage_XL_dict.keys()
hnet_2stage_to_3stage: dict[str, str] = {}
target = "backbone." + 3 * main_network_str + "layers."

for key in hnet_2stage_XL_keys:
    if key.startswith(target):
        layer_num = key[len(target) :].split(".")[0]
        if 0 <= int(layer_num) <= 7:
            new_layer_num = int(layer_num)
            new_key = key.replace(
                f"{target}{layer_num}",
                f"backbone.{2*main_network_str}encoder.layers.{new_layer_num}",
            )
            hnet_2stage_to_3stage[key] = new_key
        elif 8 <= int(layer_num) <= 18:
            new_layer_num = int(layer_num) - 8
            new_key = key.replace(
                f"{target}{layer_num}",
                f"backbone.{4*main_network_str}layers.{new_layer_num}",
            )
            hnet_2stage_to_3stage[key] = new_key
        elif 19 <= int(layer_num) <= 26:
            new_layer_num = int(layer_num) - 19
            new_key = key.replace(
                f"{target}{layer_num}",
                f"backbone.{2*main_network_str}decoder.layers.{new_layer_num}",
            )
            hnet_2stage_to_3stage[key] = new_key
    else:
        hnet_2stage_to_3stage[key] = key

# %%
set(hnet_2stage_to_3stage.values()) - set(hmnet_3stage_XL.state_dict().keys())
# %%
for key, value in hnet_2stage_to_3stage.items():
    try:
        hmnet_3stage_XL.state_dict()[value].copy_(hnet_2stage_XL_dict[key])
    except KeyError:
        print(f"KeyError: {key} not found in hnet_2stage_XL_dict")
    except RuntimeError as e:
        print(f"RuntimeError: {e}")

# %%
torch.save(
    hmnet_3stage_XL.state_dict(),
    "/root/workspace/hmnet-archive/ckpts/hmnet_3stage_XL_from_2stage_XL.pt",
)
