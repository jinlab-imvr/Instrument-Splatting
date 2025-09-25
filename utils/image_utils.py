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
import numpy as np
import os
from PIL import Image

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_viewcam_from_npz(npz_path, obj2world = None, device='cuda'):
  
    T0 = torch.eye(4)
    T0[:3, :3] = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
    T0 = T0.to(device)
   
    data = np.load(npz_path)
    w2c = torch.from_numpy(data['extrinsic_mat']).float().to(device)
    cam_K = torch.from_numpy(data['intrinsic_mat']).float().to(device)
    # obj_pose = torch.from_numpy(data['object_poses'][-1][1]).float().to(device)
    if '001.npz' in npz_path:
        obj_pose = T0
    else:
        objects = [d for d in data['object_poses'] if 'Shape' in d[0]]
        assert len(objects) == 1
        object_pose = objects[0][1]
        obj_pose = torch.from_numpy(object_pose).float().to(device)
    obj_pose = T0.inverse() @ obj_pose
    if obj2world is not None:
        obj_pose = obj2world @ obj_pose
    # colored image, depth, silhouette
    rgb_file = npz_path.replace('.npz', '.png')
    img = torch.from_numpy(np.array(Image.open(rgb_file))).float().to(device) / 255.0
    depth = torch.from_numpy(data['depth_map']).float().to(device)
    depth = depth.clamp(0.0, 0.5)
    mask = (depth > 0).float()
    add = torch.tensor([[0,0,0,1]]).float().to(device)
    w2c = torch.cat([w2c, add], dim=0)  
    obj2cam = w2c @ obj_pose
    return img, depth, mask, obj2cam, cam_K 

# mesh_render_path = '/mnt/iMVR/shuojue/data/instrument_dataset_LND/wrist_mesh_render'
# npz_files = sorted([f for f in os.listdir(mesh_render_path) if f.endswith('.npz')])
# for npz_file in npz_files:
#     npz_path = os.path.join(mesh_render_path, npz_file)
#     load_viewcam_from_npz(npz_path, 'cuda')
    