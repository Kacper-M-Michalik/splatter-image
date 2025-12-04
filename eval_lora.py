import torch, os, sys
from omegaconf import OmegaConf
from hydra import initialize, compose

from scene.gaussian_predictor_lora import GaussianSplatPredictor, merge_lora_weights
from splatter_datasets.dataset_factory import get_dataset
from eval import evaluate_dataset

def evaluate_lora(cfg_path, ckpt_path, outdir):
    # Loading Hydra config
    try:
        initialize(config_path="configs", version_base=None) # To handle Colab (when already initialised)
    except ValueError:
        pass 

    cfg = compose(config_name="default_config", overrides=["+dataset=cars_priors"])
    cfg.data.category = "cars_priors" 
    cfg.opt.lora_finetune = True 

    device = torch.device("cuda")
    os.makedirs(outdir, exist_ok=True)

    print("\n>> Building model")
    model = GaussianSplatPredictor(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    merge_lora_weights(model)
    model.eval()

    dataset = get_dataset(cfg, "val")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    scores = evaluate_dataset(model, loader, device=device,
                              model_cfg=cfg,
                              save_vis=1,
                              out_folder=outdir,
                              score_path=f"{outdir}/scores.txt")

    print(scores)

if __name__ == "__main__":
    evaluate_lora(cfg_path=sys.argv[1], ckpt_path=sys.argv[2], outdir=sys.argv[3])