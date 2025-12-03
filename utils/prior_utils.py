from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

def is_base_model(cfg):
    if OmegaConf.select(cfg, "data.use_pred_depth") is True:
        return False
    if OmegaConf.select(cfg, "data.use_pred_normal") is True:
        return False

    return True

def calc_channels(cfg):    
    # Base RGB channels
    in_channels = 3

    # Older configs may not have relvant options, select() returns None if the option is missing
    if OmegaConf.select(cfg, "data.use_pred_depth") is True:
        in_channels += 1
    if OmegaConf.select(cfg, "data.use_pred_normal") is True:
        in_channels += 3

    return in_channels

def get_model_category(cfg):
    name = "model"

    if OmegaConf.select(cfg, "data.use_pred_depth") is True:
        name += "-depth"
    if OmegaConf.select(cfg, "data.use_pred_normal") is True:
        name += "-normal"
    if OmegaConf.select(cfg, "data.lora_finetune") is True:
        name += "-finetune"    

    return name

def graft_weights_with_channel_expansion(old_state_dict, new_model, old_cfg, new_cfg):
    new_state_dict = new_model.state_dict()

    # Iterate over all layers
    for name, new_param in new_state_dict.items():
        if name not in old_state_dict:
            # New LoRA parameters not in base model checkpoint. Skip to avoid KeyError.
            if "lora_" in name:
                continue
            print("Failed to find layer {} in HuggingFace model state_dict".format(name))
            raise Exception("Mismatched source model for graft")

        old_param = old_state_dict[name]

        # Directly copy tensors if matching in size (handles most layers)
        if (new_param.shape == old_param.shape):
            new_state_dict[name] = old_param.clone()
            continue

        # In theory we should only reach here for Conv2D layers, as such only need to handle weights, and these should only have extra channels in shape[1]
        if ('weight' in name):
            # Dimension check for Conv2D weights
            if new_param.dim() == 4 and old_param.dim() == 4:
                assert new_param.shape[0] == old_param.shape[0], "Grafting only supported for adding channels, not changing resolution"
                assert new_param.shape[1] > old_param.shape[1], "Cannot truncate channels during graft, can only add channels"

                new_weights = new_param.clone()
                new_weights[:, :old_param.shape[1], :, :] = old_param
                new_weights[:, old_param.shape[1]:, :, :] = 0.0

                new_state_dict[name] = new_weights
            else:
                 print(f"Warning: Skipping graft for {name} due to dimension mismatch")
        else:
            raise Exception("Failed layer graft")
            
    new_model.load_state_dict(new_state_dict)
    return new_model