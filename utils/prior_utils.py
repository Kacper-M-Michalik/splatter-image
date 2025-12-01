def calc_channels(cfg):    
    in_channels = 3

    if cfg.data.use_pred_depth:
        in_channels += 1
    if cfg.data.use_pred_normal:
        in_channels += 3

    return in_channels
    
