import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from .dataset_readers import readCamerasWithPriorsFromTxt, readCamerasWithPriorsFromHF
from utils.general_utils import PILtoTorch, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World

from .shared_dataset import SharedDataset

# Efficiently convert a HuggingFace dataset into pandas dataframe grouped by uuid and then 
# sorted by frame_id 
def process_and_chunk(hf_dataset, uuids):
    df = hf_dataset.to_pandas()    
    df = df[df['uuid'].isin(uuids)]    
    grouped = {uuid: group.sort_values(by=['frame_id']) for uuid, group in df.groupby('uuid')}

    # Immediately deallocate the large dataframe to reduce memory usage
    del df
    return grouped

class SRNPriorsDataset(SharedDataset):
    def __init__(self, cfg, dataset_name="train"):
        super().__init__()
        self.cfg = cfg

        self.dataset_name = dataset_name
        if dataset_name == "vis":
            self.dataset_name = "test"            

        # Download ready dataset from HuggingFace
        print("Started downloading datasets")
        dataset_intrins = load_dataset(
            "MVP-Group-Project/srn_cars_priors",
            name="srn_cars_intrins", 
            split=self.dataset_name
        )
        dataset_poses = load_dataset(
            "MVP-Group-Project/srn_cars_priors",
            name="srn_cars_poses", 
            split=self.dataset_name
        )
        dataset_rgbs = load_dataset(
            "MVP-Group-Project/srn_cars_priors",
            name="srn_cars_rgbs", 
            split=self.dataset_name
        )
        dataset_depths = load_dataset(
            "MVP-Group-Project/srn_cars_priors",
            name="srn_cars_depths", 
            split=self.dataset_name
        )
        dataset_normals = load_dataset(
            "MVP-Group-Project/srn_cars_priors",
            name="srn_cars_normals", 
            split=self.dataset_name
        )
        print("Downloaded datasets")

        # Convert intrinsics files to a dataframe for performance reasons
        pre_dataset_intrins = dataset_intrins.to_pandas()
        pre_dataset_intrins.sort_values(by=["uuid"], ascending=[True], inplace=True)

        # Calculate dataset length
        assert len(dataset_poses) == len(dataset_rgbs)
        if cfg.data.subset != -1:
            assert cfg.data.subset > 0
            assert len(pre_dataset_intrins) >= cfg.data.subset
            self.subset_length = cfg.data.subset
        else:
            self.subset_length = len(pre_dataset_intrins)

        self.dataset_intrins = pre_dataset_intrins.iloc[:self.subset_length]
        uuids = set(self.dataset_intrins['uuid'].unique())    

        # Convert remaining HF datasets to dataframes indexed by uuid
        self.dataset_poses = process_and_chunk(dataset_poses, uuids)
        print("Converted poses")
        self.dataset_rgbs = process_and_chunk(dataset_rgbs, uuids)
        print("Converted rgbs")
        self.dataset_depths = process_and_chunk(dataset_depths, uuids)
        print("Converted depths")
        self.dataset_normals = process_and_chunk(dataset_normals, uuids)    
        print("Converted normals")

        print("Dataset intrin length: {}".format(self.subset_length))

        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)
        
        self.imgs_per_obj = self.cfg.opt.imgs_per_obj

        # In deterministic version the number of testing images
        # and number of training images are the same
        if self.cfg.data.input_images == 1:
            self.test_input_idxs = [64]
        elif self.cfg.data.input_images == 2:
            self.test_input_idxs = [64, 128]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.subset_length
    
    def get_example_id(self, index):
        return self.dataset_intrins.iloc[index]['uuid']
    
    def load_example_id(self, intrin_idx, trans = np.array([0.0, 0.0, 0.0]), scale=1.0):
        uuid = self.dataset_intrins.iloc[intrin_idx]['uuid']   

        if not hasattr(self, "all_rgbs"):
            self.all_poses = {}
            self.all_rgbs = {}
            self.all_depths = {}
            self.all_normals = {}
            self.all_world_view_transforms = {}
            self.all_view_to_world_transforms = {}
            self.all_full_proj_transforms = {}
            self.all_camera_centers = {}

        # Cache retrieved items
        if uuid not in self.all_rgbs.keys():            
            self.all_rgbs[uuid] = []
            self.all_depths[uuid] = []
            self.all_normals[uuid] = []
            self.all_world_view_transforms[uuid] = []
            self.all_full_proj_transforms[uuid] = []
            self.all_camera_centers[uuid] = []
            self.all_view_to_world_transforms[uuid] = []

            # Convert data from dataframes into appropriate datatypes and calculate 
            # relevant projections/transforms and metadata
            cam_infos = readCamerasWithPriorsFromHF(self.dataset_poses[uuid],
                                                    self.dataset_rgbs[uuid], 
                                                    self.dataset_depths[uuid], 
                                                    self.dataset_normals[uuid])
            
            # Remove dataframes with original dataset data as processed copies are now stored in memory
            del self.dataset_poses[uuid]
            del self.dataset_rgbs[uuid]
            del self.dataset_depths[uuid]
            del self.dataset_normals[uuid]

            for cam_info in cam_infos:
                R = cam_info.R
                T = cam_info.T

                assert cam_info.width == self.cfg.data.training_resolution
                assert cam_info.height == self.cfg.data.training_resolution

                # We store everything as uint8's due to memory usage problems, we recompute the floats on the go in __getitem__
                image = cam_info.rgb_image.view(3, cam_info.height, cam_info.width)
                self.all_rgbs[uuid].append(image)
                
                image = cam_info.depth_image.view(1, cam_info.height, cam_info.width)
                self.all_depths[uuid].append(image)
                
                image = cam_info.normal_image.view(3, cam_info.height, cam_info.width)
                self.all_normals[uuid].append(image) 
 
                world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

                self.all_world_view_transforms[uuid].append(world_view_transform)
                self.all_view_to_world_transforms[uuid].append(view_world_transform)
                self.all_full_proj_transforms[uuid].append(full_proj_transform)
                self.all_camera_centers[uuid].append(camera_center)
            
            self.all_world_view_transforms[uuid] = torch.stack(self.all_world_view_transforms[uuid])
            self.all_view_to_world_transforms[uuid] = torch.stack(self.all_view_to_world_transforms[uuid])
            self.all_full_proj_transforms[uuid] = torch.stack(self.all_full_proj_transforms[uuid])
            self.all_camera_centers[uuid] = torch.stack(self.all_camera_centers[uuid])
            
            self.all_rgbs[uuid] = torch.stack(self.all_rgbs[uuid])
            self.all_depths[uuid] = torch.stack(self.all_depths[uuid])
            self.all_normals[uuid] = torch.stack(self.all_normals[uuid])

    # Get permutation of images and associated data from dataset
    def __getitem__(self, index):  
        uuid = self.dataset_intrins.iloc[index]['uuid']
        self.load_example_id(index)

        if self.dataset_name == "train":
            frame_idxs = torch.randperm(
                    len(self.all_rgbs[uuid])
                    )[:self.imgs_per_obj]

            frame_idxs = torch.cat([frame_idxs[:self.cfg.data.input_images], frame_idxs], dim=0)

        else:
            input_idxs = self.test_input_idxs
            
            frame_idxs = torch.cat([torch.tensor(input_idxs), 
                                    torch.tensor([i for i in range(251) if i not in input_idxs])], dim=0) 

        # Dictionary expected as batch entry in prior/dataset processing scripts
        images_and_camera_poses = {
            "gt_images": (self.all_rgbs[uuid][frame_idxs].float() / 255.0).clamp(0.0, 1.0),
            "pred_depths": (self.all_depths[uuid][frame_idxs].float() / 255.0).clamp(0.0, 1.0),
            "pred_normals": (self.all_normals[uuid][frame_idxs].float() / 255.0).clamp(0.0, 1.0),
            "world_view_transforms": self.all_world_view_transforms[uuid][frame_idxs],
            "view_to_world_transforms": self.all_view_to_world_transforms[uuid][frame_idxs],
            "full_proj_transforms": self.all_full_proj_transforms[uuid][frame_idxs],
            "camera_centers": self.all_camera_centers[uuid][frame_idxs]
        }

        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])

        return images_and_camera_poses