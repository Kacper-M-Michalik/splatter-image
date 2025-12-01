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
    in_channels = 3

    if OmegaConf.select(cfg, "data.use_pred_depth") is True:
        in_channels += 1
    if OmegaConf.select(cfg, "data.use_pred_normal") is True:
        in_channels += 3

    return in_channels

def graft_weights_with_channel_expansion(old_state_dict, new_model, old_cfg, new_cfg):
    new_state_dict = new_model.state_dict()

    for name, param in new_state_dict.items():
        if name not in old_state_dict:
            print("Failed to find layer {} in HuggingFace model state_dict".format(name))
            raise "Mismatched source model for graft"

    print("\n DIRECT ACCESS\n")
    print(old_state_dict["network_with_offset.encoder.enc.128x128_conv.weight"].shape)
    print("\n NEW")
    print(new_state_dict["network_with_offset.encoder.enc.128x128_conv.weight"].shape)

    return new_model