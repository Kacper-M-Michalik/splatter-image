import argparse
import torch
import sys
import os
import json
from hydra import initialize, compose
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from scene.gaussian_predictor_lora import GaussianSplatPredictor, merge_lora_weights
from splatter_datasets.dataset_factory import get_dataset
from eval import evaluate_dataset

def main(args):
    print(f"\n[LoRA Evaluation] Config: {args.config_name} | Dataset: {args.dataset_override}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_folder, exist_ok=True)

    # Initialize Hydra and use source configs
    try:
        initialize(config_path="configs", version_base=None)
    except ValueError:
        pass # To handle Colab's re-initialization error

    overrides = []
    if args.dataset_override:
        overrides.append(f"+dataset={args.dataset_override}") # dynamically inject the dataset override
    
    # Add LoRA specific flags to ensure architecture compliance
    cfg = compose(config_name=args.config_name, overrides=overrides) # Add LoRA specific flags
    cfg.opt.lora_finetune = True 
    if args.dataset_override == "cars_priors": # To match architecture expected by checkpoint
        cfg.data.category = "cars_priors"
    model = GaussianSplatPredictor(cfg).to(device)
    if not os.path.exists(args.ckpt_path):
        print(f"Checkpoint not found at {args.ckpt_path}")
        sys.exit(1)
        
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False) # Ensure model loaded with frozen parameters
    merge_lora_weights(model)
    model.eval()

    dataset = get_dataset(cfg, "val")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


    scores = evaluate_dataset(
        model, 
        loader, 
        device=device, 
        model_cfg=cfg, 
        save_vis=args.save_vis, 
        out_folder=args.out_folder,
        score_path=os.path.join(args.out_folder, "scores.txt")
    )

    print(json.dumps(scores, indent=4))
    with open(os.path.join(args.out_folder, "metrics.json"), "w") as f:
        json.dump(scores, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA-adapted Splatter Image")
    
    # Required Arguments
    parser.add_argument("ckpt_path", type=str, help="Path to model_best.pth")
    parser.add_argument("out_folder", type=str, help="Output directory for results")
    
    # Optional Arguments (with defaults matching your workflow)
    parser.add_argument("--config_name", type=str, default="default_config", 
                        help="Base config filename (default: default_config)")
    parser.add_argument("--dataset_override", type=str, default="cars_priors",
                        help="Dataset override flag (default: cars_priors)")
    parser.add_argument("--save_vis", type=int, default=0, 
                        help="Number of examples to visualize")

    args = parser.parse_args()
    main(args)