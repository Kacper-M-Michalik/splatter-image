import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from .dataset_readers import readCamerasWithPriorsFromTxt, readCamerasWithPriorsFromHF
from utils.general_utils import PILtoTorch, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World

from .shared_dataset import SharedDataset

class SRNPriorsDataset(SharedDataset):
    def __init__(self, cfg, dataset_name="train"):
        super().__init__()
        self.cfg = cfg

        self.dataset_name = dataset_name
        if dataset_name == "vis":
            self.dataset_name = "test"            
        #if dataset_name == "val":
        #    self.dataset_name = "validation"

        # Download ready dataset from HuggingFace
        self.dataset_intrins = load_dataset(
            "MVP-Group-Project/srn_cars_priors",
            name="srn_cars_intrins", 
            split=self.dataset_name
        )
        self.dataset_poses = load_dataset(
            "MVP-Group-Project/srn_cars_priors",
            name="srn_cars_poses", 
            split=self.dataset_name
        )
        self.dataset_rgbs = load_dataset(
            "MVP-Group-Project/srn_cars_priors",
            name="srn_cars_rgbs", 
            split=self.dataset_name
        )
        self.dataset_depths = load_dataset(
            "MVP-Group-Project/srn_cars_priors",
            name="srn_cars_depths", 
            split=self.dataset_name
        )
        self.dataset_normals = load_dataset(
            "MVP-Group-Project/srn_cars_priors",
            name="srn_cars_normals", 
            split=self.dataset_name
        )

        self.dataset_intrins = self.dataset_intrins.to_pandas()
        self.dataset_poses = self.dataset_poses.to_pandas()
        self.dataset_rgbs = self.dataset_rgbs.to_pandas()
        self.dataset_depths = self.dataset_depths.to_pandas()
        self.dataset_normals = self.dataset_normals.to_pandas()

        self.dataset_intrins.sort_values(by=["uuid"], ascending=[True], inplace=True)
        self.dataset_poses.sort_values(by=["uuid", "frame_id"], ascending=[True, True], inplace=True)
        self.dataset_rgbs.sort_values(by=["uuid", "frame_id"], ascending=[True, True], inplace=True)
        self.dataset_depths.sort_values(by=["uuid", "frame_id"], ascending=[True, True], inplace=True)
        self.dataset_normals.sort_values(by=["uuid", "frame_id"], ascending=[True, True], inplace=True)

        assert len(self.dataset_poses) == len(self.dataset_rgbs)
        
        if cfg.data.subset != -1:
            assert cfg.data.subset > 0
            assert len(self.dataset_intrins) >= cfg.data.subset
            self.subset_length = cfg.data.subset
        else:
            self.subset_length = len(self.dataset_intrins)
        
        print("Dataset intrin length: {}".format(self.subset_length))
        print("Dataset image length: {}".format(self.dataset_poses))

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

    def load_example_id(self, intrin_idx, trans = np.array([0.0, 0.0, 0.0]), scale=1.0):
        uuid = self.dataset_intrins.iloc[intrin_idx]['uuid']        
        print(uuid)

        if not hasattr(self, "all_rgbs"):
            self.all_poses = {}
            self.all_rgbs = {}
            self.all_depths = {}
            self.all_normals = {}
            self.all_world_view_transforms = {}
            self.all_view_to_world_transforms = {}
            self.all_full_proj_transforms = {}
            self.all_camera_centers = {}

        if uuid not in self.all_rgbs.keys():
            self.all_poses[uuid] = []
            self.all_rgbs[uuid] = []
            self.all_depths[uuid] = []
            self.all_normals[uuid] = []
            self.all_world_view_transforms[uuid] = []
            self.all_full_proj_transforms[uuid] = []
            self.all_camera_centers[uuid] = []
            self.all_view_to_world_transforms[uuid] = []

            print("len poses: {}".format(len(self.dataset_poses[self.dataset_poses['uuid']==uuid])))
            print("len rgbs: {}".format(len(self.dataset_rgbs[self.dataset_rgbs['uuid']==uuid])))
            print("len depths: {}".format(len(self.dataset_depths[self.dataset_depths['uuid']==uuid])))
            print("len normals: {}".format(len(self.dataset_normals[self.dataset_normals['uuid']==uuid])))
            cam_infos = readCamerasWithPriorsFromHF(uuid, 
                                                    self.dataset_poses[self.dataset_poses['uuid']==uuid],
                                                    self.dataset_rgbs[self.dataset_rgbs['uuid']==uuid], 
                                                    self.dataset_depths[self.dataset_depths['uuid']==uuid], 
                                                    self.dataset_normals[self.dataset_normals['uuid']==uuid])

            for cam_info in cam_infos:
                R = cam_info.R
                T = cam_info.T

                assert cam_info.rgb_image.shape[1] == self.cfg.data.training_resolution
                assert cam_info.rgb_image.shape[2] == self.cfg.data.training_resolution
                image = (torch.from_numpy(cam_info.rgb_image) / 255.0).clamp(0.0, 1.0)
                print(image.shape)
                print(image)
                self.all_rgbs[uuid].append(image)
                
                assert cam_info.depth_image.shape[1] == self.cfg.data.training_resolution
                assert cam_info.depth_image.shape[2] == self.cfg.data.training_resolution
                image = (torch.from_numpy(cam_info.depth_image) / 255.0).clamp(0.0, 1.0)
                print(image.shape)
                print(image)
                self.all_depths[uuid].append(image)
                
                assert cam_info.normal_image.shape[1] == self.cfg.data.training_resolution
                assert cam_info.normal_image.shape[2] == self.cfg.data.training_resolution
                image = (torch.from_numpy(cam_info.normal_image) / 255.0).clamp(0.0, 1.0)
                print(image.shape)
                print(image)
                self.all_normals[uuid].append(image) 

                raise "test"

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
            self.all_poses[uuid] = torch.stack(self.all_poses[uuid])
            self.all_rgbs[uuid] = torch.stack(self.all_rgbs[uuid])
            self.all_depths[uuid] = torch.stack(self.all_depths[uuid])
            self.all_normals[uuid] = torch.stack(self.all_normals[uuid])

    def __getitem__(self, index):  
        uuid = self.dataset_intrins[index]['uuid']
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

        images_and_camera_poses = {
            "gt_images": self.all_rgbs[uuid][frame_idxs].clone(),
            "pred_depths": self.all_depths[uuid][frame_idxs].clone(),
            "pred_normals": self.all_normals[uuid][frame_idxs].clone(),
            "world_view_transforms": self.all_world_view_transforms[uuid][frame_idxs],
            "view_to_world_transforms": self.all_view_to_world_transforms[uuid][frame_idxs],
            "full_proj_transforms": self.all_full_proj_transforms[uuid][frame_idxs],
            "camera_centers": self.all_camera_centers[uuid][frame_idxs]
        }

        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])

        return images_and_camera_poses