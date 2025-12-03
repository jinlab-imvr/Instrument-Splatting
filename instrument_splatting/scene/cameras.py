#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View, getProjectionMatrix2
import math

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, K, image, gt_alpha_mask,
                 image_name, uid, world_view_transform = torch.eye(4) , zfar = None, znear = None, depth = None, semantic_mask = None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 keypoints_2d = None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        if FoVx is None:
            FoVx = focal2fov(K[0, 0], image.shape[2])
        self.FoVx = FoVx
        if FoVy is None:
            FoVy = focal2fov(K[1, 1], image.shape[1])
        self.FoVy = FoVy
        self.K = K
        self.image_name = image_name
        self.depth = depth

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        if self.original_image.shape[1] > 946:
            self.original_image = self.original_image[:, :self.original_image.shape[1] -40]
        self.original_depth = depth.to(self.data_device) if depth is not None else None

        self.keypoints_2d = torch.from_numpy(keypoints_2d).float().to(self.data_device) if keypoints_2d is not None else None
        self.original_semantic_mask = torch.from_numpy(semantic_mask).to(self.data_device) if semantic_mask is not None else None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # unit: mm
        if zfar is None:
            self.zfar = 500
            self.znear = 1
        else:
            self.zfar = zfar
            self.znear = znear

        self.trans = trans
        self.scale = scale
        if world_view_transform is not None:
            self.world_view_transform = world_view_transform.to(self.data_device)
        else:
            self.world_view_transform = torch.tensor(getWorld2View(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix2(znear=self.znear, zfar=self.zfar, K=K, h = self.image_height ,w = self.image_width).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

