import argparse
import json
import os
import sys
import tqdm
from omegaconf import OmegaConf

from huggingface_hub import hf_hub_download

import lpips as lpips_lib

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from splatter_datasets.dataset_factory import get_dataset
from utils.loss_utils import ssim as ssim_fn

class Metricator():
    def __init__(self, device):
        self.lpips_net = lpips_lib.LPIPS(net='vgg').to(device)
    def compute_metrics(self, image, target):
        lpips = self.lpips_net( image.unsqueeze(0) * 2 - 1, target.unsqueeze(0) * 2 - 1).item()
        psnr = -10 * torch.log10(torch.mean((image - target) ** 2, dim=[0, 1, 2])).item()
        ssim = ssim_fn(image, target).item()
        return psnr, ssim, lpips

@torch.no_grad()
def evaluate_dataset(model, dataloader, device, model_cfg, save_vis=0, score_path=None, out_folder=None):
    """
    Runs evaluation on the dataset passed in the dataloader. 
    Computes, prints and saves PSNR, SSIM, LPIPS.
    Args:
        save_vis: how many examples will have visualisations saved
        score_path: where to store scores
    """

    # Make folder for output images
    if save_vis > 0:
        os.makedirs(out_folder, exist_ok=True)

    if score_path is None:
       score_path = "scores.txt" 

    with open(score_path, "w+") as f:
        f.write("")

    # Some Settings
    bg_color = [1, 1, 1] if model_cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Instantiate metric calculator
    metricator = Metricator(device)

    psnr_all_examples_novel = []
    ssim_all_examples_novel = []
    lpips_all_examples_novel = []

    psnr_all_examples_cond = []
    ssim_all_examples_cond = []
    lpips_all_examples_cond = []

    # Go through all images for eval
    for d_idx, data in enumerate(tqdm.tqdm(dataloader)):
        psnr_all_renders_novel = []
        ssim_all_renders_novel = []
        lpips_all_renders_novel = []
        psnr_all_renders_cond = []
        ssim_all_renders_cond = []
        lpips_all_renders_cond = []

        # Move data to device
        data = {k: v.to(device) for k, v in data.items()}

        # Get quarternions for input images (typically 1)
        rot_transform_quats = data["source_cv2wT_quat"][:, :model_cfg.data.input_images]

        # ShapeNet and ObajaVerse renders use the same focal info within the dataset, CO3D does not, so requires reading said data
        if model_cfg.data.category == "hydrants" or model_cfg.data.category == "teddybears":
            focals_pixels_pred = data["focals_pixels"][:, :model_cfg.data.input_images, ...]
        else:
            focals_pixels_pred = None

        # Concatenate selected priors
        input_images = data["gt_images"][:, :model_cfg.data.input_images, ...]
        if "use_pred_depth" in model_cfg.data and model_cfg.data.use_pred_depth:
            assert model_cfg.data.category == "cars_priors", "Dataset does not have predicated maps!"
            input_images = torch.cat([input_images,
                            data["pred_depths"][:, :model_cfg.data.input_images, ...]],
                            dim=2)
        if "use_pred_normal" in model_cfg.data and model_cfg.data.use_pred_normal:
            assert model_cfg.data.category == "cars_priors", "Dataset does not have predicated maps!"
            input_images = torch.cat([input_images,
                            data["pred_normals"][:, :model_cfg.data.input_images, ...]],
                            dim=2)

        # Get camera to center depth
        if model_cfg.data.origin_distances:
            input_images = torch.cat([input_images,
                            data["origin_distances"][:, :model_cfg.data.input_images, ...]],
                            dim=2)

        # Pepare folders for output image
        example_id = dataloader.dataset.get_example_id(d_idx)
        if d_idx < save_vis:

            out_example_gt = os.path.join(out_folder, "{}_".format(d_idx) + example_id + "_gt")
            out_example = os.path.join(out_folder, "{}_".format(d_idx) + example_id)

            os.makedirs(out_example_gt, exist_ok=True)
            os.makedirs(out_example, exist_ok=True)

        # Batch has length 1, the first image is conditioning
        reconstruction = model(input_images,
                               data["view_to_world_transforms"][:, :model_cfg.data.input_images, ...],
                               rot_transform_quats,
                               focals_pixels_pred)

        # Calculate metrics
        # Go through all ground truth images
        for r_idx in range( data["gt_images"].shape[1]):
            if "focals_pixels" in data.keys():
                focals_pixels_render = data["focals_pixels"][0, r_idx]
            else:
                focals_pixels_render = None

            image = render_predicted({k: v[0].contiguous() for k, v in reconstruction.items()},
                                     data["world_view_transforms"][0, r_idx],
                                     data["full_proj_transforms"][0, r_idx], 
                                     data["camera_centers"][0, r_idx],
                                     background,
                                     model_cfg,
                                     focals_pixels=focals_pixels_render)["render"]

            if d_idx < save_vis:
                # vis_image_preds(reconstruction, out_example)
                torchvision.utils.save_image(image, os.path.join(out_example, '{0:05d}'.format(r_idx) + ".png"))
                torchvision.utils.save_image(data["gt_images"][0, r_idx, ...], os.path.join(out_example_gt, '{0:05d}'.format(r_idx) + ".png"))

            # Exclude blank images from metric computation 
            # -> To match batch sizes, if there are less renderes for a model than in other, we fill in batch with empty images, which we ignore here
            if not torch.all(data["gt_images"][0, r_idx, ...] == 0):
                psnr, ssim, lpips = metricator.compute_metrics(image, data["gt_images"][0, r_idx, ...])
                if r_idx < model_cfg.data.input_images:
                    psnr_all_renders_cond.append(psnr)
                    ssim_all_renders_cond.append(ssim)
                    lpips_all_renders_cond.append(lpips)
                else:
                    psnr_all_renders_novel.append(psnr)
                    ssim_all_renders_novel.append(ssim)
                    lpips_all_renders_novel.append(lpips)

        psnr_all_examples_cond.append(sum(psnr_all_renders_cond) / len(psnr_all_renders_cond))
        ssim_all_examples_cond.append(sum(ssim_all_renders_cond) / len(ssim_all_renders_cond))
        lpips_all_examples_cond.append(sum(lpips_all_renders_cond) / len(lpips_all_renders_cond))

        psnr_all_examples_novel.append(sum(psnr_all_renders_novel) / len(psnr_all_renders_novel))
        ssim_all_examples_novel.append(sum(ssim_all_renders_novel) / len(ssim_all_renders_novel))
        lpips_all_examples_novel.append(sum(lpips_all_renders_novel) / len(lpips_all_renders_novel))

        with open(score_path, "a+") as f:
            f.write("{}_".format(d_idx) + example_id + \
                    " " + str(psnr_all_examples_novel[-1]) + \
                    " " + str(ssim_all_examples_novel[-1]) + \
                    " " + str(lpips_all_examples_novel[-1]) + "\n")

    scores = {"PSNR_cond": sum(psnr_all_examples_cond) / len(psnr_all_examples_cond),
              "SSIM_cond": sum(ssim_all_examples_cond) / len(ssim_all_examples_cond),
              "LPIPS_cond": sum(lpips_all_examples_cond) / len(lpips_all_examples_cond),
              "PSNR_novel": sum(psnr_all_examples_novel) / len(psnr_all_examples_novel),
              "SSIM_novel": sum(ssim_all_examples_novel) / len(ssim_all_examples_novel),
              "LPIPS_novel": sum(lpips_all_examples_novel) / len(lpips_all_examples_novel)}

    return scores

@torch.no_grad()
def eval_robustness(model, dataloader, device, model_cfg, out_folder=None):
    """
    Evaluates robustness to shift and zoom
    """
    os.makedirs(out_folder, exist_ok=True)

    bg_color = [1, 1, 1] if model_cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    obj_idx = 98

    data = {k: v.unsqueeze(0) for k, v in dataloader.dataset[obj_idx].items()}
    data = {k: v.to(device) for k, v in data.items()}
    
    rot_transform_quats = data["source_cv2wT_quat"][:, :model_cfg.data.input_images]
    focals_pixels_pred = None

    input_images = data["gt_images"][:, :model_cfg.data.input_images, ...]
    example_id = dataloader.dataset.get_example_id(obj_idx)

    resize_to_128_transform = transforms.Resize(128, 
        interpolation=transforms.InterpolationMode.BILINEAR)

    for test_zoom_idx, crop_size in enumerate([-40, -30, -20, -10, 0, 10, 20, 30, 40]):

        # ================ zoom transforms ===============
        if crop_size >= 0:
            # crop the source images
            input_images = data["gt_images"][
                        0, :model_cfg.data.input_images,  
                        :, crop_size:model_cfg.data.training_resolution-crop_size, crop_size:model_cfg.data.training_resolution-crop_size]
        elif crop_size < 0:
            # pad only the source images
            padding_transform = transforms.Pad(padding=-crop_size,
                                                fill=1.0)
            input_images = padding_transform(data["gt_images"][0, :model_cfg.data.input_images])

        if crop_size != 0:
            input_images = resize_to_128_transform(input_images)
        

        # ================ shift transforms ===============
        x_shift = 0
        y_shift = crop_size
        padding_transform = transforms.Pad(padding=(abs(x_shift), abs(y_shift)),
                                                fill=1.0)

        padded_source  = padding_transform(data["gt_images"][0, :model_cfg.data.input_images])
        y_start = abs(y_shift) + y_shift
        x_start = abs(x_shift) + x_shift
        input_images = padded_source[ :, :, 
                                        y_start : model_cfg.data.training_resolution + y_start,
                                        x_start : model_cfg.data.training_resolution + x_start]

        input_images = input_images.unsqueeze(0)

        out_example_gt = os.path.join(out_folder, "{}_".format(test_zoom_idx) + example_id + "_gt")
        out_example = os.path.join(out_folder, "{}_".format(test_zoom_idx) + example_id)

        os.makedirs(out_example_gt, exist_ok=True)
        os.makedirs(out_example, exist_ok=True)

        # batch has length 1, the first image is conditioning
        reconstruction = model(input_images,
                                data["view_to_world_transforms"][:, :model_cfg.data.input_images, ...],
                                rot_transform_quats,
                                focals_pixels_pred)

        for r_idx in range( data["gt_images"].shape[1]):
            if "focals_pixels" in data.keys():
                focals_pixels_render = data["focals_pixels"][0, r_idx]
            else:
                focals_pixels_render = None
            image = render_predicted({k: v[0].contiguous() for k, v in reconstruction.items()},
                                        data["world_view_transforms"][0, r_idx],
                                        data["full_proj_transforms"][0, r_idx], 
                                        data["camera_centers"][0, r_idx],
                                        background,
                                        model_cfg,
                                        focals_pixels=focals_pixels_render)["render"]

            torchvision.utils.save_image(image, os.path.join(out_example, '{0:05d}'.format(r_idx) + ".png"))
            torchvision.utils.save_image(data["gt_images"][0, r_idx, ...], os.path.join(out_example_gt, '{0:05d}'.format(r_idx) + ".png"))

@torch.no_grad()
def main(dataset_name, experiment_path, device_idx, split='test', save_vis=0, out_folder=None):
    
    # Set device and random seed
    device = torch.device("cuda:{}".format(device_idx))
    torch.cuda.set_device(device)

    # Load pretrained model from HuggingFace if no local model specified
    if args.experiment_path is None:        
        # Eval run on the our new dataset with priors
        if dataset_name in ["cars_priors"]:
            cfg_path = hf_hub_download(repo_id="MVP-Group-Project/splatter-image-priors", 
                                    filename="model-depth-normal/config.yaml")
            model_path = hf_hub_download(repo_id="MVP-Group-Project/splatter-image-priors",
                                filename="model-depth-normal/model_best.pth")
        
        # Eval run on previous Splatter Image datasets
        else:            
            cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                                    filename="config_{}.yaml".format(dataset_name))
            if dataset_name in ["gso", "objaverse"]:
                model_name = "latest"
            else:
                model_name = dataset_name
            model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                                filename="model_{}.pth".format(model_name))
        
    else:
        cfg_path = os.path.join(experiment_path, ".hydra", "config.yaml")
        model_path = os.path.join(experiment_path, "model_latest.pth")
    
    # Load training configuration
    training_cfg = OmegaConf.load(cfg_path)

    # Check that training and testing datasets match if not using official models 
    if args.experiment_path is not None:
        if dataset_name == "gso":
            # GSO model must have been trained on objaverse
            assert training_cfg.data.category == "objaverse", "Model-dataset mismatch"
        else:
            assert training_cfg.data.category == dataset_name, "Model-dataset mismatch"

    # Load model
    model = GaussianSplatPredictor(training_cfg)
    ckpt_loaded = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model = model.to(device)
    model.eval()
    print('Loaded model!')

    # Override dataset in cfg if testing objaverse model
    if training_cfg.data.category == "objaverse" and split in ["test", "vis"]:
        training_cfg.data.category = "gso"
    # Instantiate dataset loader
    dataset = get_dataset(training_cfg, split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            persistent_workers=True, pin_memory=True, num_workers=1)
    
    scores = evaluate_dataset(model, dataloader, device, training_cfg, save_vis=save_vis, score_path=args.score_path, out_folder=out_folder)
    if split != 'vis':
        print(scores)
    return scores


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('dataset_name', type=str, help='Dataset to evaluate on', 
                        choices=['objaverse', 'gso', 'cars', 'chairs', 'hydrants', 'teddybears', 'nmr', 'cars_priors'])
    parser.add_argument('--experiment_path', type=str, default=None, help='Path to the parent folder of the model. \
                        If set to None, a pretrained model will be downloaded')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'vis', 'train'],
                        help='Split to evaluate on (default: test). \
                        Using vis renders loops and does not return scores - to be used for visualisation. \
                        You can also use this to evaluate on the training or validation splits.')
    parser.add_argument('--out_folder', type=str, default='out', help='Output folder to save renders (default: out)')
    parser.add_argument('--save_vis', type=int, default=0, help='Number of examples for which to save renders (default: 0)')
    parser.add_argument('--score_path', type=str, default=None, help='Full path for output file of inference scores (default: ./scores.txt)')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    dataset_name = args.dataset_name
    print("Evaluating on dataset {}".format(dataset_name))
    experiment_path = args.experiment_path
    if args.experiment_path is None:
        print("Will load a model released with the paper.")
    else:
        print("Loading a local model according to the experiment path")
    split = args.split
    if split == 'vis':
        print("Will not print or save scores. Use a different --split to return scores.")
    out_folder = args.out_folder
    save_vis = args.save_vis
    if save_vis == 0:
        print("Not saving any renders (only computing scores). To save renders use flag --save_vis")

    scores = main(dataset_name, experiment_path, 0, split=split, save_vis=save_vis, out_folder=out_folder)
    # Save scores to json in the experiment folder if appropriate split was used
    if split != "vis":
        if experiment_path is not None:
            score_out_path = os.path.join(experiment_path, 
                                   "{}_scores.json".format(split))
        else:
            score_out_path = "{}_{}_scores.json".format(dataset_name, split)
        with open(score_out_path, "w+") as f:
            json.dump(scores, f, indent=4)
            