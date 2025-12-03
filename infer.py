import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l1_loss_normalization

from renderer.gaussian_renderer import render_with_instrument_v3 as render
import sys
from instrument_splatting.scene import InstrumentScene
from utils.instrument import Instrument
from instrument_splatting import (
    optimizationParamTypeCallbacks,
    gaussianModel
)
from utils.loss_utils import ssim
import lpips
from utils.image_utils import psnr
from utils.camera_utils import Camera
import torch.autograd as autograd
import torch.nn.functional as F
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, load_viewcam_from_npz
from utils.graphics_utils import build_rotation, matrix_to_quaternion, rotation_matrix_to_angle_axis
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from torchvision.utils import save_image
from submodules.ml_aspanformer.src.ASpanFormer.aspanformer import ASpanFormer 
from submodules.ml_aspanformer.src.config.default import get_cfg_defaults
from submodules.ml_aspanformer.src.utils.misc import lower_config
from submodules.ml_aspanformer.demo import demo_utils
import numpy as np
import random
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import cv2
from collections import deque
from scipy.ndimage import label
from utils.loss_utils import dice as Dice
import json



remove_list = []
def pixel_to_camera_coords(pixel_coords, depth_map, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (pixel_coords[:, 0] - cx) * depth_map[pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)] / fx
    y = (pixel_coords[:, 1] - cy) * depth_map[pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)] / fy
    z = depth_map[pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)]
    return np.vstack((x, y, z)).T

def get_gripper_tips(instrument: Instrument, gripper_segmentation_mask, wrist_segmentation_mask):
    """
    Get the tips of the instrument from the segmentation mask
    """
    # Label connected components
    labeled_mask, num_features = label(gripper_segmentation_mask)
    if num_features ==1:
        gripper_coords = np.argwhere(labeled_mask == 1)
        centroid = np.mean(gripper_coords, axis=0)
        centered_coords = gripper_coords - centroid
        _, _, vh = np.linalg.svd(centered_coords)
        principal_axis = vh[0]
        # get the orthogonal vector of principal axis
        orthogonal_vector = np.array([principal_axis[1], -principal_axis[0]])
        # Project the centered coordinates onto the orthogonal vector
        projections = np.dot(centered_coords, orthogonal_vector)
        # projected points with coord < 0 belongs to left gripper
        left_gripper_coords = gripper_coords[projections < 0]
        right_gripper_coords = gripper_coords[projections >= 0]

    # Find the two largest components
    component_sizes = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
    largest_components = sorted(range(1, num_features + 1), key=lambda i: component_sizes[i - 1], reverse=True)[:2]

    # Get the coordinates of the two largest components
    gripper_coords1 = np.argwhere(labeled_mask == largest_components[0])
    gripper_coords2 = np.argwhere(labeled_mask == largest_components[1])

    # Compute the centroids of the two components
    centroid1 = np.mean(gripper_coords1, axis=0)
    centroid2 = np.mean(gripper_coords2, axis=0)

    # Center the coordinates by subtracting the centroids
    centered_coords1 = gripper_coords1 - centroid1
    centered_coords2 = gripper_coords2 - centroid2

    # Perform SVD on both components
    _, _, vh1 = np.linalg.svd(centered_coords1)
    _, _, vh2 = np.linalg.svd(centered_coords2)

    # The principal axes are the first right-singular vectors
    principal_axis1 = vh1[0]
    principal_axis2 = vh2[0]

    # Project the centered coordinates onto the principal axis
    projections1 = np.dot(centered_coords1, principal_axis1)
    projections2 = np.dot(centered_coords2, principal_axis2)

    # Find the indices of the maximum and minimum projections
    tip1_index = np.argmax(projections1)
    tip2_index = np.argmin(projections1)
    tip3_index = np.argmax(projections2)
    tip4_index = np.argmin(projections2)

    # Get the corresponding coordinates of the tips
    tip1 = gripper_coords1[tip1_index]
    tip2 = gripper_coords1[tip2_index]
    tip3 = gripper_coords2[tip3_index]
    tip4 = gripper_coords2[tip4_index]

    # Determine which tip is farthest from the wrist
    wrist_coords = np.argwhere(wrist_segmentation_mask > 0)
    wrist_centroid = np.mean(wrist_coords, axis=0)

    dist1 = np.linalg.norm(tip1 - wrist_centroid)
    dist2 = np.linalg.norm(tip2 - wrist_centroid)
    dist3 = np.linalg.norm(tip3 - wrist_centroid)
    dist4 = np.linalg.norm(tip4 - wrist_centroid)

    gripper1_tip = tip1 if dist1 > dist2 else tip2
    gripper2_tip = tip3 if dist3 > dist4 else tip4

    return gripper1_tip, gripper2_tip

    

def proj_2d(K, xyz):
    """
    Project 3D points to 2D points using camera intrinsic matrix K. And draw the 2D points in the 2D frame
    """
    # Project the points
    xyz = xyz/xyz[:, 2].unsqueeze(1)
    xyz = torch.matmul(K.float(), xyz.t()).t()
    # Normalize the points
    xyz = xyz[:, :2] 
    
    # Convert to numpy
    points_2d = xyz.cpu().numpy().astype(int)
    
    # Create a blank image
    image = torch.zeros((512, 640, 3), dtype=torch.uint8).cpu().numpy()
    
    # Draw points on the image
    for point in points_2d:
        cv2.circle(image, (point[0], point[1]), radius=3, color=(0, 255, 0), thickness=-1)
    
    # Save or display the image
    cv2.imwrite('projected_points.png', image)
    # cv2.imshow('Projected Points', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


class Pose(object):
    def activate_alpha(self, alpha):
        # Scale alpha to range from -90 to 90 degrees
        return torch.pi * torch.sigmoid(alpha) - torch.pi / 2

    def activate_theta(self, theta):
        # Scale theta to range from -80 to 80 degrees
        return (160.0 / 180.0) * torch.pi * torch.sigmoid(theta) - (80.0 / 180.0) * torch.pi

    def __init__(self, rot, trans, alpha, theta_l, theta_r, init_optimizer = True):
        quat = matrix_to_quaternion(rot)
        quat = F.normalize(quat, p=2, dim=0)
        if init_optimizer:
            self.rot = torch.nn.Parameter(quat, requires_grad=True)
            self.trans = torch.nn.Parameter(trans, requires_grad=True)
            self.alpha = torch.nn.Parameter(alpha, requires_grad=True)
            self.theta_l = torch.nn.Parameter(theta_l, requires_grad=True)
            self.theta_r = torch.nn.Parameter(theta_r, requires_grad=True)
            self.optimizer = torch.optim.Adam([
                {'params': [self.rot], 'lr': 0.001, 'name': 'rot'},
                {'params': [self.trans], 'lr': 0.0001, 'name': 'trans'},
                {'params': [self.alpha], 'lr': 0.005, 'name': 'alpha'},
                {'params': [self.theta_l], 'lr': 0.01, 'name': 'theta_l'},
                {'params': [self.theta_r], 'lr': 0.01, 'name': 'theta_r'}
            ], lr=0.0, eps=1e-15)
        else:
            self.rot = quat.detach().clone()
            self.trans = trans.detach().clone()
            self.alpha = alpha.detach().clone()
            self.theta_l = theta_l.detach().clone()
            self.theta_r = theta_r.detach().clone()


    def get_pose(self):
        rotation_matrix = build_rotation(self.rot)
        return rotation_matrix[0], self.trans, self.alpha, self.theta_l, self.theta_r

    def step_optimizer(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
def array2tensor(array, device="cuda", dtype=torch.float32):
    return torch.tensor(array, dtype=dtype, device=device)
class LPIPS(object):
    """
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    """
    def __init__(self, device="cuda"):
        self.model = lpips.LPIPS(net='alex').to(device)

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        error =  self.model.forward(y_pred, y_true)
        return torch.mean(error)
    
lpips = LPIPS()
def cal_lpips(a, b, device="cuda", batch=2):
    """Compute lpips.
    a, b: [batch, H, W, 3]"""
    if not torch.is_tensor(a):
        a = array2tensor(a, device)
    if not torch.is_tensor(b):
        b = array2tensor(b, device)

    lpips_all = []
    for a_split, b_split in zip(a.split(split_size=batch, dim=0), b.split(split_size=batch, dim=0)):
        out = lpips(a_split, b_split)
        lpips_all.append(out)
    lpips_all = torch.stack(lpips_all)
    lpips_mean = lpips_all.mean()
    return lpips_mean

def rotation_matrix_x(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_angle, -sin_angle],
        [0, sin_angle, cos_angle]
    ])
    return torch.tensor(rotation_matrix, dtype=torch.float32).cuda()
def rotation_matrix_z(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    return torch.tensor(rotation_matrix, dtype=torch.float32).cuda()

def joint_texture_learning(instrument: Instrument, pose: Pose, viewpoint_cam, opt, pipe, 
                        background, global_iteration, semantic_silhouette0, depth0, testing):
    first_iter = 1
    
    # semantic_silhouette0 = None
    min_loss = 10000
    for iteration in range(first_iter, 2):
        # Render
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        rot, trans, alpha, theta_l, theta_r = pose.get_pose()
        
        # if global_iteration > 5000:
        #     instrument.forward_kinematics(rot, trans, alpha, theta_l, theta_r, gs_grad=True, pose_grad = True)
        # else:
        if random.random() > 0:
            instrument.forward_kinematics(rot, trans, alpha, theta_l, theta_r, gs_grad=True, pose_grad = False)
        else:
            instrument.forward_kinematics(rot @ rotation_matrix_x(180), trans, -alpha, theta_r, theta_l, gs_grad=True, pose_grad = False)
        transformation = torch.eye(4, device="cuda")
        transformation[:3, :3] = rot
        transformation[:3, 3] = trans 
        override_camera_center = torch.inverse(transformation)[:3, 3] * 1000
        render_pkg = render(viewpoint_cam, instrument, pipe, bg, 
                    override_camera_center=override_camera_center, render_semantics_silhouette=False)
        render_pkg2 = render(viewpoint_cam, instrument, pipe, bg, 
                    override_camera_center=override_camera_center, render_semantics_silhouette=True)
        semantic_silhouette = render_pkg2["render"].permute(2, 0, 1)
        depth = render_pkg2["depth"]
        image, opacity, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["opacity"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_semantic_map = viewpoint_cam.original_semantic_mask.cuda()
        gt_semantic_map = gt_semantic_map.permute(2, 0, 1)  

        gt_image[:, (gt_semantic_map.sum(dim=0) == 0)] = 1
        image = image.permute(2, 0, 1)
        if semantic_silhouette0 is not None:
            error_mask = (gt_semantic_map.sum(dim=0, keepdim=True) > 0) != (semantic_silhouette0.sum(dim=0, keepdim=True) > 0.5)
            error_mask = error_mask.repeat(3, 1, 1)
            loss_rgb = l1_loss(image * (~error_mask) , gt_image * (~error_mask))
        else:
            loss_rgb = l1_loss(image, gt_image)
        semantic_silhouette_sup = semantic_silhouette0 if semantic_silhouette0 is not None else semantic_silhouette
        loss_silhouette = l1_loss(semantic_silhouette, semantic_silhouette_sup)
        if depth0 is not None:
            depth_loss = l1_loss(depth, depth0)
            loss = loss_rgb + loss_silhouette #+ 0.1 * depth_loss
        else:
            loss = loss_rgb + loss_silhouette
        # scale = instrument.get_scaling
        # scale_ratio = scale.max(dim=-1)[0] / scale.min(dim=-1)[0]
        # regularize the scale
        # scale_loss = torch.mean(torch.relu(scale - 2))
        # scale_ratio_loss = torch.mean(torch.relu(scale_ratio - 2))
        # loss = loss +  0.1 * scale_ratio_loss
        if not testing:
            loss.backward()
            instrument.step_optimizer()
            instrument.zero_grad(set_to_none=True)
        else:
            loss = torch.tensor([0.0], device="cuda")
        # instrument.update_alpha()
        # instrument.prepare_scaling_rot()
        with torch.no_grad():
            psnr_val = psnr(image, gt_image)
    
    semantic_silhouette0 = semantic_silhouette.detach()
    depth0 = depth.detach()
    return instrument, image, gt_image, psnr_val, semantic_silhouette0, depth0
# instrument.forward_kinematics(rot @ rotation_matrix_x(180), trans, -alpha, theta_r, theta_l, gs_grad=True, pose_grad = False)
# render_pkg2 = render(viewpoint_cam, instrument, pipe, bg, 
#                     override_camera_center=override_camera_center, render_semantics_silhouette=True)
# semantic_silhouette = render_pkg2["render"].permute(2, 0, 1)
# save_image(semantic_silhouette,'test1.png')
def infering(gs_type, dataset, opt, pipe, exp_name):
    first_iter = 0
    assert exp_name is not None, "Please provide a path to save the experiment results."
    exp_path = os.path.join(dataset.exp_dir, exp_name)
    memory_pool_path = os.path.join(exp_path, 'memory_pool.pth')
    memory_pool = torch.load(memory_pool_path)

    scene = InstrumentScene(dataset, gaussianModel[gs_type], dataset.sh_degree,
                             pretrain_path = exp_path, pretrain_params='trained_3dgs_param')
    instrument = scene.instrument
    instrument.training_setup(opt)

    white_background = True
    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    viewpoint_stack = None
    first_iter += 1

    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
    viewpoint_cam = viewpoint_stack[0]
    # viewpoint_stack = viewpoint_stack[:25]
    rot = torch.from_numpy(viewpoint_cam.R).cuda()
    trans = torch.from_numpy(viewpoint_cam.T + 0.000).cuda()
    alpha = torch.tensor([0.0], dtype=torch.float32, device="cuda")
    theta_l = torch.tensor([15. / 180. * torch.pi], dtype=torch.float32, device="cuda")
    theta_r = torch.tensor([15. / 180. * torch.pi], dtype=torch.float32, device="cuda")
    pose = Pose(rot, trans, alpha, theta_l, theta_r)

    # online mode to run the render and compare (global iterations) 
    # memory_pool = {}
    rot0, trans0, alpha0, theta_l0, theta_r0 = rot.detach().clone(), trans.detach().clone(), alpha.detach().clone(), theta_l.detach().clone(), theta_r.detach().clone()
    # perform tracking
    
    img_num = len(list(memory_pool.keys()))
    # texture learning
    progress_bar = tqdm(range(first_iter, opt.texture_learning_iterations), desc="Tracking progress")
    named_iteration = 0
    name_iteration = False
    semantic_silhouette0, depth0 = None, None
    original_geometry_dict = {}
    


    assert os.path.exists(f"{scene.source_path}/testing_iterations.txt")

    with open(f"{scene.source_path}/testing_iterations.txt", "r") as f:
            testing_iterations = [int(line.strip()) for line in f]

    render_rgb_path = os.path.join(exp_path, 'renders_rgb')
    render_semantics_path = os.path.join(exp_path, 'renders_semantics')
    render_depth_path = os.path.join(exp_path, 'renders_depth')
    keypoint_map_path = os.path.join(exp_path, 'keypoint_maps')
    os.makedirs(render_rgb_path, exist_ok=True)
    os.makedirs(render_semantics_path, exist_ok=True)
    os.makedirs(render_depth_path, exist_ok=True)
    os.makedirs(keypoint_map_path, exist_ok=True)


    ssims = []
    psnrs = []
    lpipss = []
    rmses = []
    # Training
    for i, iteration in enumerate(list(memory_pool.keys())):
        if iteration  in testing_iterations:
            continue
    # Testing
    # for i, iteration in enumerate(testing_iterations):
        # Every 1000 its we increase the levels of SH up to a maximum degree
        viewpoint_cam = viewpoint_stack[iteration]
        image_name = viewpoint_cam.image_name
        gt_semantic_map = viewpoint_cam.original_semantic_mask.cuda()
        gt_image = viewpoint_cam.original_image.cuda()
        gt_semantic_map = gt_semantic_map.permute(2, 0, 1)
        

        pose_info, dice = memory_pool[iteration]['pose_info'], memory_pool[iteration]['dice']
        wrist_dice, shaft_dice, gripper_dice = dice  
        testing = False if iteration not in testing_iterations else True

        joint_state = pose_info.get_pose()
        pose = Pose(*joint_state)
        per_view_dict = {}
        with torch.no_grad():
            rot, trans, alpha, theta_l, theta_r = pose_info.get_pose()
            instrument.forward_kinematics(rot, trans, alpha, theta_l, theta_r, gs_grad=True, pose_grad = False)
            
            transformation = torch.eye(4, device="cuda")
            transformation[:3, :3] = rot
            transformation[:3, 3] = trans 
            override_camera_center = transformation[:3, 3] * 1000
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            render_pkg = render(viewpoint_cam, instrument, pipe, bg, 
                    override_camera_center=override_camera_center, render_semantics_silhouette=False)
            image, opacity, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["opacity"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            image = image.permute(2, 0, 1)
            depth = render_pkg["depth"]
            opacity = opacity
            opacity_mask = opacity < 0.5
            image[:,opacity_mask] = 1
            mask = gt_semantic_map.sum(dim=0)>0
            gt_shaft_mask = gt_semantic_map[0] > 0.5
            # dilate the shaft mask
            # gt_shaft_mask = F.max_pool2d(gt_shaft_mask[None, None].float(), 3, stride=1, padding=1)[0, 0] > 0.5
            # gt_image[:,gt_shaft_mask ==0] = 1
            # image[:, gt_shaft_mask == 0] = 1
            # gt_image[:,mask == 0] = 1

            # gt_image[..., -35:,:] = 1
            psnrs.append(psnr(image[None], gt_image[None]))
            ssims.append(ssim(image[None], gt_image[None]))
            lpipss.append(cal_lpips(image[None], gt_image[None]))
            # rmses.append(rmse(depth, gt_depth, mask))
        
        # per_view_dict[task_name].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
        #                                                     "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
        #                                                     "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
        #                                                     "RMSES": {name: lp for lp, name in zip(torch.tensor(rmses).tolist(), image_names)}})]

            
            # save_image(semantic_silhouette, f"semantics_{task_name}/{i:05d}.png")
            # # Save the rendered image
            
            save_image(image, render_rgb_path + f"/{iteration:05d}.png")

            # # Save the depth map
            depth = np.clip(depth.cpu().squeeze().numpy().astype(np.uint8), 0, 255)
            cv2.imwrite(render_depth_path + f"/{iteration:05d}.png", depth)

            # Project keypoints to 2D rendered RGB images and save the keypoint map
            # Collect keypoints from instrument parts (after forward_kinematics they are in view/camera space)
            part_keypoints = {}
            # instrument.forward_kinematics(rot @ rotation_matrix_x(180), trans, -alpha, theta_r, theta_l)
            for part_name, part in instrument.part_dict.items():
                kp = part.render_params.get("keypoints", None)
                if kp is None:
                    continue
                kp_np = kp.detach().cpu().numpy().reshape(-1, 3)
                part_keypoints[part_name] = kp_np

            # Prepare intrinsics
            K = viewpoint_cam.K
            if isinstance(K, torch.Tensor):
                K = K.detach().cpu().numpy()
            K = np.array(K).reshape(3, 3)

            # Image size (image is tensor [3, H, W])
            H, W = image.shape[1], image.shape[2]

            # Colors in BGR for cv2
            color_map = {
                "shaft": (255, 0, 0),
                "wrist": (0, 255, 0),
                "l_gripper": (0, 0, 255),
                "r_gripper": (0, 255, 255)
            }

            # Create blank keypoint map and overlay on rendered RGB
          
            # Convert rendered image to uint8 BGR for overlay
            try:
                rendered_rgb = (image.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            except Exception:
                rendered_rgb = np.ones((H, W, 3), dtype=np.uint8) * 255

            overlay = rendered_rgb.copy()

            def project_pts(pts3):
                # pts3: (N,3)
                homog = (K @ pts3.T).T  # (N,3)
                z = homog[:, 2:3].copy()
                # avoid divide by zero
                z[z == 0] = 1e-6
                uv = homog[:, :2] / z
                return uv

            # # Draw keypoints
            # for part_name, pts3 in part_keypoints.items():
            #     if part_name == 'wrist':
            #         # draw the first and last keypoints
            #         pts3 = pts3[[0, 5], :]
            #     uv = project_pts(pts3)
            #     color = color_map.get(part_name, (0, 128, 255))
            #     for p in uv:
            #         x, y = int(round(p[0])), int(round(p[1]))
            #         if 0 <= x < W and 0 <= y < H:
            #             cv2.circle(overlay, (x, y), radius=3, color=color, thickness=-1)
            #             # optional small border for visibility
            #             # cv2.circle(overlay, (x, y), radius=6, color=(0, 0, 0), thickness=1)

            # Save keypoint map and overlay
            # os.makedirs(keypoint_map_path, exist_ok=True)

            # overlay_file = os.path.join(keypoint_map_path, f"{iteration:05d}.png")

            # cv2.imwrite(overlay_file, overlay)
            # with torch.no_grad():
            #     novel_view_folder = 'Results/' + task_name + f"/text_novel_views"
            #     if not os.path.exists(novel_view_folder):
            #         os.makedirs(novel_view_folder)
            #     for k,novel_rot_angle in enumerate([-40, -30, -20, -10, 10, 20, 30, 40]):
                    
            #         rot_novel = rot.clone() @ rotation_matrix_x(novel_rot_angle)
            #         transformation = torch.eye(4, device="cuda")
            #         transformation[:3, :3] = rot_novel
            #         transformation[:3, 3] = trans 
            #         override_camera_center = transformation[:3, 3] * 1000
            #         novel_rot_angle = torch.tensor(novel_rot_angle , dtype=torch.float32, device="cuda")
            #         rot_novel = instrument.calculate_lateral_rotation(novel_rot_angle/180.*torch.pi, rot, trans, alpha)
            #         instrument.forward_kinematics(rot_novel, trans.clone(), alpha.clone(), theta_l.clone(), theta_r.clone(), gs_grad=False, pose_grad = False, with_mesh=True)
            #         render_img, depth = instrument.render_trimesh(viewpoint_cam.K)
            #         render_pkg2 = render(viewpoint_cam, instrument, pipe, bg, 
            #                         override_camera_center=override_camera_center,active_sh_degree= 0, render_semantics_silhouette=False)
            #         semantic_silhouette = render_pkg2["render"].permute(2, 0, 1)
            #         save_image(semantic_silhouette, novel_view_folder + '/' + f'{iteration}_novel_view{int(novel_rot_angle)}_nosh.png')
            #         render_pkg2 = render(viewpoint_cam, instrument, pipe, bg, 
            #                         override_camera_center=override_camera_center,active_sh_degree= None, render_semantics_silhouette=False)
            #         semantic_silhouette = render_pkg2["render"].permute(2, 0, 1)
            #         save_image(semantic_silhouette, novel_view_folder + '/' + f'{iteration}_novel_view{int(novel_rot_angle)}.png')
    print("SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

    # Generate per-image metrics dictionary and save as JSON
    metrics = {}
    def to_number(x):
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(x)

    for idx, iteration in enumerate(testing_iterations):
        # try to get a human-friendly image name, fallback to iteration id
        try:
            image_name = viewpoint_stack[iteration].image_name
        except Exception:
            image_name = str(iteration)
        ssim_val = to_number(ssims[idx]) if idx < len(ssims) else None
        psnr_val = to_number(psnrs[idx]) if idx < len(psnrs) else None
        lpips_val = to_number(lpipss[idx]) if idx < len(lpipss) else None
        metrics[image_name] = {"SSIM": ssim_val, "PSNR": psnr_val, "LPIPS": lpips_val}

    
    metrics_path = os.path.join(exp_path, "image_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved per-image metrics to {metrics_path}")
            # Save the semantic silhouette
            # save_image(semantic_silhouette, f"semantics_{task_name}/{i:05d}.png")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gs_type', type=str, default="gs_instrument")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--exp_name", type = str, default="exp1")


    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.gs_type = args.gs_type


    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])


    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    infering(
        args.gs_type,
        lp.extract(args), op.extract(args), pp.extract(args),
        args.exp_name
    )

    print("\nInfering complete.")