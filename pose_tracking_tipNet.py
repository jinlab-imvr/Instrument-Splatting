import os
import torch
from random import randint
from utils.loss_utils import l1_loss, instrument_dice, l1_loss_normalization
from renderer.gaussian_renderer import render_with_instrument_v3 as render
import sys
from scene import  InstrumentScene
from utils.instrument import Instrument
from games import (
    optimizationParamTypeCallbacks,
    gaussianModel
)
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
from games.mesh_splatting.scene.network import KeyPointNet
import random
import matplotlib.pyplot as plt
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import cv2
from collections import deque
from scipy.ndimage import label
from utils.loss_utils import dice as Dice

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ASPAN_CONFIG = os.path.join(REPO_ROOT, 'submodules', 'ml_aspanformer', 'configs', 'aspan', 'outdoor', 'aspan_test.py')
ASPAN_WEIGHTS = os.path.join(REPO_ROOT, 'submodules', 'ml_aspanformer', 'weights', 'outdoor.ckpt')

remove_endovis_2017 = [173,177,149, 150,151,152, 153, 163, 162]
remove_endovis_2017 = [149, 177, 173, 195, 84, 50, 18, 35, 23, 34, 20, 14, 66, 188, 185, 187, 221, 192, 78, 194, 178, 193, 136, 49, 103, 45, 222, 147, 83, 25, 40, 74, 55, 183, 19, 202, 100, 79, 162, 163, 24, 29, 85, 58, 130, 80, 68, 86, 61, 81, 44, 189, 69, 121]
remove_endovis_2017 = [18,19,173,177,149, 150,151,152, 153, 163, 162, 66, 101]
remove_list_17 = [7,8, 9,10,12,13,17,21,32,47,48,51,73,89,96,98,105,115,116,118,119,123,141,144,159,160,161,164,
               165,166,169,176,180,205,210]+\
                [16, 30,26,33,56,57,70,72,75,82,93,95,104,117,127,134,137,148,151,170,175,190,200,204,212,213,216,218]+\
                    [15,42, 43,52,53,174] +[1, 2, 3,22,46,128]
remove_list_18 = [58, 59, 60, 61, 62, 63, 64, 65,66, 67, 120, 183,99,100,101,102,103,104,105,106,107,108,109,110,111]
remove_list = remove_list_18
# remove_endovins_2018 = [49]
def pixel_to_camera_coords(pixel_coords, depth_map, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (pixel_coords[:, 0] - cx) * depth_map[pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)] / fx
    y = (pixel_coords[:, 1] - cy) * depth_map[pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)] / fy
    z = depth_map[pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)]
    return np.vstack((x, y, z)).T

def get_pose_from_PnP(keypoints_3d, keypoints_2d, camview):
    """
    Get the pose from PnP algorithm
    keypoints_2d: Nx2
    keypoints_3d: Nx3
    K: 3x3
    """
    K = camview.K

    keypoints_2d = keypoints_2d.cpu().numpy()
    keypoints_3d = keypoints_3d.cpu().numpy()
    K = K
    keypoints_2d = keypoints_2d.astype(np.float32)
    keypoints_3d = keypoints_3d.astype(np.float32)
    # Solve PnP
    rvec, tvec,_ = solve_pnp_ransac(keypoints_3d, keypoints_2d, K.astype(np.float32)) # w2c 
    
    rvec_c2w, tvec_c2w = rvec, tvec #convert_w2c_to_c2w(rvec, tvec)
    rvec_matrix = cv2.Rodrigues(rvec_c2w.numpy())[0]

    rotation_matrix = torch.from_numpy(rvec_matrix).float().cuda()
    trans_cam = torch.from_numpy(tvec_c2w.numpy()).float().squeeze().cuda()
    quat_pred = matrix_to_quaternion(rotation_matrix)
    alpha = torch.tensor([0.0], dtype=torch.float32, device="cuda")
    theta_l = torch.tensor([5. / 180. * torch.pi], dtype=torch.float32, device="cuda")
    theta_r = torch.tensor([5. / 180. * torch.pi], dtype=torch.float32, device="cuda")
    return quat_pred, trans_cam, torch.tensor([alpha[0], theta_l[0], theta_r[0]], dtype=torch.float32, device="cuda")

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
        if rot.dim() > 1 and rot.shape[0] == 3:
            quat = matrix_to_quaternion(rot)
            quat = F.normalize(quat, p=2, dim=0)
        else:
            quat = rot
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

def calculate_matches(viewpoint_cam, matcher, lst_viewpoint_cam, pose, iteration, render_depth):
    img = viewpoint_cam.original_image
    img = torch.mean(img, dim = 0, keepdim=True)
    lst_img = lst_viewpoint_cam.original_image
    lst_img = torch.mean(lst_img, dim = 0, keepdim=True)
    lst_semantic_mask = lst_viewpoint_cam.original_semantic_mask
    lst_wrist_mask = (lst_semantic_mask[:, :, 1] > 0.5).cpu().numpy()
    lst_shaft_mask = (lst_semantic_mask[:, :, 0] > 0.5).cpu().numpy()

    data = {'image0': lst_img[None].cuda().float(),
    'image1': img[None].cuda().float()}
    with torch.no_grad():
        matcher(data, online_resize=True)
    corr0, corr1 = data['mkpts0_f'].cpu().numpy(), data['mkpts1_f'].cpu().numpy()
    semantic_mask = viewpoint_cam.original_semantic_mask
    wrist_mask = (semantic_mask[:, :, 1] > 0.5).cpu().numpy()
    shaft_mask = (semantic_mask[:, :, 0] > 0.5).cpu().numpy()
    # gripper_mask = (semantic_mask[:, :, 2] > 0.5).cpu().numpy()
    # Filter matches based on masks
    wrist_mask = lst_wrist_mask[corr0[:, 1].astype(int), corr0[:, 0].astype(int)] * wrist_mask[corr1[:, 1].astype(int), corr1[:, 0].astype(int)]
    wrist_corr0 = corr0[wrist_mask]
    wrist_corr1 = corr1[wrist_mask]
    
    shaft_mask = lst_shaft_mask[corr0[:, 1].astype(int), corr0[:, 0].astype(int)] * shaft_mask[corr1[:, 1].astype(int), corr1[:, 0].astype(int)]
    shaft_corr0 = corr0[shaft_mask]
    shaft_corr1 = corr1[shaft_mask]

    if len(wrist_corr0) > 4 or len(shaft_corr0) > 4:
        corr0 = wrist_corr0 if len(wrist_corr0) > 4 else shaft_corr0
        corr1 = wrist_corr1 if len(wrist_corr0) > 4 else shaft_corr1
        with torch.no_grad():
            depth_map = render_depth.detach().cpu().numpy()
            K = viewpoint_cam.K
            rot_prev, translation_prev, alpha_prev, theta_l_prev, theta_r_prev = pose.get_pose()
            del pose
            corr0_3d = pixel_to_camera_coords(corr0, depth_map/1000., K)
            corr0_3d = rot_prev.detach().cpu().numpy().T @ corr0_3d.T - rot_prev.detach().cpu().numpy().T @ translation_prev.detach().cpu().numpy()[:, None]
            _, rvec, tvec, inliers = cv2.solvePnPRansac(corr0_3d.T, corr1, K, None)
            R, _ = cv2.Rodrigues(rvec)
            t = tvec[:, 0]
            lst_img_np = (lst_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_np = (img.cpu().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
            display = demo_utils.draw_match(lst_img_np, img_np, corr0, corr1)
            if os.path.exists("Matches") == False:
                os.mkdir("Matches")
            cv2.imwrite(f'Matches/matches{iteration}.png', display)
            
            pose = Pose(torch.from_numpy(R).cuda(), torch.from_numpy(t).cuda(), alpha_prev.clone(), theta_l_prev.clone(), theta_r_prev.clone())

    lst_img = img
    lst_wrist_mask = (semantic_mask[:, :, 1] > 0.5).cpu().numpy()
    lst_shaft_mask = (semantic_mask[:, :, 0] > 0.5).cpu().numpy()


def chamfer_loss_loose_reg(pred_points, target_points, radius = None):
    """
    Calculate the Chamfer Distance between two point sets.
    
    Args:
        pred_points (torch.Tensor): Predicted points, shape [B, N, 2] (Batch, Points, Coordinates).
        target_points (torch.Tensor): Target points, shape [B, M, 2] (Batch, Points, Coordinates).
    
    Returns:
        torch.Tensor: Chamfer Distance (scalar).
    """
    # Compute pairwise distances between all points in the two sets
    pred_points = pred_points.unsqueeze(2)  # [B, N, 1, 2]
    target_points = target_points.unsqueeze(1)  # [B, 1, M, 2]
    distances = torch.norm(pred_points - target_points, dim=-1)  # [B, N, M]

    min_dist_pred_to_target, _ = distances.min(dim=2)  # [B, N]
    min_dist_target_to_pred, _ = distances.min(dim=1)  # [B, M]

    r = radius # 40 if iteration < 1000 else 20s
    # Average the distances
    if r == None:
        loss = min_dist_pred_to_target.mean(dim=1) + min_dist_target_to_pred.mean(dim=1)
    else:
        mask_pred = min_dist_pred_to_target > r
        mask_tgt = min_dist_target_to_pred > r
        kept_pred = torch.where(mask_pred, min_dist_pred_to_target, torch.zeros_like(min_dist_pred_to_target))
        kept_tgt = torch.where(mask_tgt, min_dist_target_to_pred, torch.zeros_like(min_dist_target_to_pred))
        loss = torch.sum(kept_pred)/(torch.sum(mask_pred)+1e-5) + torch.sum(kept_tgt)/(torch.sum(mask_tgt)+1e-5)
    return loss



def get_gripper_tips_from_ToolTipNet(mask, ckpt_path=None):
    if ckpt_path is None:
        ckpt_path = os.path.join(REPO_ROOT, 'checkpoints', 'checkpoint_ToolTipNet.pth')
    mask = 1.0 * (mask > 0.5)
    gripper_mask = mask[2] > 0.5
    keypoint_net = KeyPointNet(n_kp=2, height=512, width=640, lim=[-1., 1., -1., 1.], use_gpu=True)
    keypoint_net.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model_state_dict'])
    keypoint_net = keypoint_net.cuda()
    keypoint_net.eval()
    with torch.no_grad():
        mask_tensor = torch.tensor(mask, dtype=torch.float32).cuda()
        mask_tensor = mask_tensor.flip(0)
        pred_keypoints, soft_max = keypoint_net(mask_tensor[None], 1.*gripper_mask[None, None])
        return pred_keypoints[0]
def track(instrument: Instrument, pose: Pose, lst_pose: Pose, viewpoint_cam, opt, pipe, background, global_iteration):
    """
    Perform render and compare.
    """
    first_iter = 1
    prev_theta_l = None
    prev_theta_r = None
    prev_rot = None
    if lst_pose is not None:
        prev_rot, _,_, prev_theta_l, prev_theta_r = lst_pose.get_pose()
        prev_theta_l = prev_theta_l.detach().clone()
        prev_theta_r = prev_theta_r.detach().clone()
        prev_rot = prev_rot.detach().clone()
    
    tracking_iterations = opt.tracking_iterations + 1
    progress_bar = tqdm(range(first_iter, tracking_iterations), desc="Tracking progress")
    
    min_loss = 10000

    gt_image = viewpoint_cam.original_image.cuda()
    gt_semantic_map = viewpoint_cam.original_semantic_mask.cuda()
    gt_semantic_map = gt_semantic_map.permute(2, 0, 1)

    gt_gripper_mask = gt_semantic_map[2] > 0.5
    gt_wrist_mask = gt_semantic_map[1] > 0.5
    tipnet = True
    if tipnet:
        if gt_gripper_mask.sum() > 10:
            tips_coord = get_gripper_tips_from_ToolTipNet(gt_semantic_map)
        else:
            tips_coord = None
    else:
        if gt_gripper_mask.sum() > 10:
            tips1_coord, tips2_coord = get_gripper_tips(instrument, 
                            (gt_gripper_mask.detach().cpu()).numpy(), 
                            (gt_wrist_mask.detach().cpu()).numpy(), with_erosion = False )
            tips1_coord_erosion, tips2_coord_erosion = get_gripper_tips(instrument, 
                            (gt_gripper_mask.detach().cpu()).numpy(), 
                            (gt_wrist_mask.detach().cpu()).numpy(), with_erosion=True)
            # choose the pairs with the smallest distance between two tips
            dist_erosion = np.linalg.norm(tips1_coord_erosion - tips2_coord_erosion)
            dist = np.linalg.norm(tips1_coord - tips2_coord)
            if dist_erosion < dist:
                tips1_coord = tips1_coord_erosion
                tips2_coord = tips2_coord_erosion
            # error when tips1_coord or tips2_coord is too close to the gt_wrist_mask
            if len(np.argwhere(gt_wrist_mask.detach().cpu().numpy())) == 0:
                tips1_coord = None
                tips2_coord = None
            else:   
                min_dist_tips1 = np.min(np.linalg.norm(np.argwhere(gt_wrist_mask.detach().cpu().numpy()) - tips1_coord, axis=1))
                min_dist_tips2 = np.min(np.linalg.norm(np.argwhere(gt_wrist_mask.detach().cpu().numpy()) - tips2_coord, axis=1))
                if min_dist_tips1 < 0.2 * min_dist_tips2 or min_dist_tips2 < 0.2 * min_dist_tips1:
                    tips1_coord = None
                    tips2_coord = None
        else:
            tips1_coord = None
            tips2_coord = None
    for iteration in range(first_iter, tracking_iterations):
        instrument.update_learning_rate_tracking(iteration)
        # Render
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        rot, trans, alpha, theta_l, theta_r = pose.get_pose()
        instrument.forward_kinematics(rot, trans, alpha, theta_l, theta_r)
        structure_loss = torch.relu(- (theta_l + theta_r))
        if prev_theta_l is not None and prev_theta_r is not None:
            # can only be applied on sequential frames with high FPS
            # cannot be applied on EndoVis dataset
            structure_loss = structure_loss + 1e-2 * torch.relu(torch.abs(theta_l - prev_theta_l) - 10./ 180. * torch.pi)
            structure_loss = structure_loss + 1e-2 * torch.relu(torch.abs(theta_r - prev_theta_r) - 10./ 180. * torch.pi)
            # rot matrix to axis angle
            # delta_angle_axis = rotation_matrix_to_angle_axis(torch.inverse(rot) @ prev_rot) # 3x3
            # delta_rot_x = delta_angle_axis[0]

        
        transformation = torch.eye(4, device="cuda")
        transformation[:3, :3] = rot
        transformation[:3, 3] = trans # unit: m
        override_camera_center = torch.inverse(transformation)[:3, 3] * 1000
        render_pkg = render(viewpoint_cam, instrument, pipe, bg, 
                    override_camera_center=override_camera_center, render_semantics_silhouette=True)

        image, opacity, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["opacity"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg["depth"]
        render_depth = torch.clamp(depth, 1.0, 250)
        # Loss

        image = image.permute(2, 0, 1)
        wrist_mask = image[1] > 0.5
        shaft_mask = image[0] > 0.5
        gripper_mask = image[2] > 0.5

        if tips_coord is not None:
            l_gripper_keypoint = instrument.part_dict["l_gripper"].render_params["keypoints"] # 1x3
            r_gripper_keypoint = instrument.part_dict["r_gripper"].render_params["keypoints"]  # 1x3
            K = torch.from_numpy(viewpoint_cam.K).to(l_gripper_keypoint.device).float()
            l_gripper_keypoint_2d = K @ l_gripper_keypoint.T
            r_gripper_keypoint_2d = K @ r_gripper_keypoint.T
            l_gripper_keypoint_2d = l_gripper_keypoint_2d[:2] / l_gripper_keypoint_2d[2]
            r_gripper_keypoint_2d = r_gripper_keypoint_2d[:2] / r_gripper_keypoint_2d[2]
            # keypoint loss
            gripper_keypoints_2d = torch.stack([l_gripper_keypoint_2d[...,0], r_gripper_keypoint_2d[...,0]], dim=0)
            r = 40 if iteration < 400 else 20
            tool_tip_loss = chamfer_loss_loose_reg(gripper_keypoints_2d.unsqueeze(0), tips_coord.unsqueeze(0), radius=r)
        image = torch.cat([image[0:1] * shaft_mask, image[1:2] * wrist_mask, image[2:3] * gripper_mask], dim=0)
        loss_silhouette = l1_loss(image, gt_semantic_map)

        if False:
            combined_gt = gt_semantic_map.clone()
            new_ch1 = image[1] + image[2]
            # 用 where 替代布尔索引的原地修改，保持梯度路径稳定
            new_ch1 = torch.where(new_ch1 > 1.0, new_ch1 * 0.5, new_ch1)
            new_ch2 = torch.zeros_like(image[2])

            # 直接构造新的 combined_render，避免对 clone 的原地改写
            combined_render = torch.stack([image[0], new_ch1, new_ch2], dim=0)

            combined_gt[1] = ((gt_semantic_map[1] + gt_semantic_map[2]) > 0.5).float()
            combined_gt[2] = torch.zeros_like(combined_gt[2])

            loss_silhouette = l1_loss(combined_render, combined_gt)
        else:
            loss_silhouette = l1_loss(image, gt_semantic_map)
        if global_iteration < 6:
            loss = loss_silhouette + structure_loss + 1e-4 * tool_tip_loss
        else:
            try:
                loss = loss_silhouette + structure_loss + 1e-4 * tool_tip_loss
            except UnboundLocalError:
                loss = loss_silhouette + structure_loss
                print("Warning:Tool-tip loss not applied due to missing tips_coord")
        # if True:
        #     loss = tool_tip_loss
        loss.backward()
        pose.step_optimizer()
        pose.zero_grad()
        if min_loss > loss_silhouette.item():
            min_loss = loss_silhouette.item()
            loss_queue = deque(maxlen=300)
        loss_queue.append(loss_silhouette.item())
        if len(loss_queue) == 300:
            if min_loss <= loss_queue[-1]:
                break
        image = (image > 0).detach()
        with torch.no_grad():
            # diff = torch.abs(image - gt_semantic_map)
            if iteration % 30 == 0 or iteration == 1:
                # save_image(diff, f"difference/difference{iteration}.png")
                if os.path.exists("difference") == False:
                    os.mkdir("difference")
                vis_image = 0.7*gt_image + 0.3 * gt_semantic_map
                # vis_image = 0.7*gt_image + 0.3 * image
                # vis_image_np = (vis_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                # cv2.circle(vis_image_np, ( int(l_gripper_keypoint_2d[1]), int(l_gripper_keypoint_2d[0]) ),radius=5, color=(255, 0, 0), thickness=-1)
                # save_image(vis_image, "difference/render_semantics.png")
                render_semantics = image * 1.0
                save_image(0.7*gt_image + 0.3 * image, f"difference/render_semantics{iteration}.png")

            ema_loss_for_log = loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Tracking Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
    with torch.no_grad():
        rot, trans, alpha, theta_l, theta_r = pose.get_pose()
        del pose
    pose_new = Pose(rot.clone(), trans.clone(), alpha.clone(), theta_l.clone(), theta_r.clone())
    return pose_new, render_semantics.detach(), render_depth.detach(),\
            gt_image.detach(), gt_semantic_map.detach(), tips_coord

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


def visualize_depth_map(depth_map, output_path):
    """
    Visualize the depth map using matplotlib and save as an image.

    Parameters:
    depth_map (numpy.ndarray): The depth map to visualize.
    output_path (str): The path to save the output image.
    """
    # Set the background (0 value regions) to NaN for better visualization
    depth_map = np.where(depth_map == 0, np.nan, depth_map)

    # Create a figure and axis
    fig, ax = plt.subplots()
    cax = ax.imshow(depth_map, cmap='viridis', interpolation='nearest')
    # Use a colormap to visualize the depth map
    ax.axis('off')

    # Save the figure with original resolution
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def training(gs_type, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
              checkpoint, debug_from, save_xyz, task_name, pretrain_params, max_frames):
    first_iter = 0
    scene = InstrumentScene(
        dataset,
        gaussianModel[gs_type],
        dataset.sh_degree,
        pretrain_params=pretrain_params,
        pretrain_path=dataset.pretrain_path,
    )
    instrument = scene.instrument
    instrument.training_setup(opt)
    matcher_config = get_cfg_defaults()
    matcher_config.merge_from_file(ASPAN_CONFIG)
    _config = lower_config(matcher_config)
    matcher = ASpanFormer(config=_config['aspan'])
    state_dict = torch.load(ASPAN_WEIGHTS, map_location='cpu')['state_dict']
    matcher.load_state_dict(state_dict, strict=False)
    matcher.cuda(), matcher.eval()
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        instrument.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # iter_start = torch.cuda.Event(enable_timing=True)
    # iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    first_iter += 1

    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
    if max_frames is not None:
        viewpoint_stack = viewpoint_stack[:max_frames]
    viewpoint_cam = viewpoint_stack[0]
    # viewpoint_stack = viewpoint_stack[:max_id]
    rot = torch.from_numpy(viewpoint_cam.R).cuda()
    trans = torch.from_numpy(viewpoint_cam.T + 0.000).cuda()
    alpha = torch.tensor([0.0], dtype=torch.float32, device="cuda")
    theta_l = torch.tensor([5. / 180. * torch.pi], dtype=torch.float32, device="cuda")
    theta_r = torch.tensor([5. / 180. * torch.pi], dtype=torch.float32, device="cuda")
    pose = Pose(rot, trans, alpha, theta_l, theta_r)

    # online mode to run the render and compare (global iterations) 
    memory_pool = {}
    rot0, trans0, alpha0, theta_l0, theta_r0 = rot.detach().clone(), trans.detach().clone(), alpha.detach().clone(), theta_l.detach().clone(), theta_r.detach().clone()
    # perform tracking
    result_dir = os.path.join(dataset.exp_dir, task_name)
    os.makedirs(result_dir, exist_ok=True)
    memory_pool_name = os.path.join(result_dir, 'memory_pool.pth')
    key_frame_pool = {} # save keyframe pose and image for re-localization
    if os.path.exists(memory_pool_name) == False:
        prev_pose = None
        for iteration, viewpoint_cam in enumerate(viewpoint_stack):
            img_name = viewpoint_cam.image_name
            img_idx = int(img_name.split('frame')[-1])
            if img_idx in remove_list:
                continue
            pnp_init = True
            if iteration == 0:
                lst_img = viewpoint_cam.original_image
                lst_img = torch.mean(lst_img, dim = 0, keepdim=True)
                lst_wrist_mask = viewpoint_cam.original_semantic_mask[:, :, 1].cpu().numpy() > 0.5
                lst_shaft_mask = viewpoint_cam.original_semantic_mask[:, :, 0].cpu().numpy() > 0.5
                # lst_gripper_mask = viewpoint_cam.original_semantic_mask[:, :, 2].cpu().numpy() > 0.5
                lst_img0 = lst_img
                lst_wrist_mask0 = lst_wrist_mask
                lst_shaft_mask0 = lst_shaft_mask

            elif pnp_init:
                img = viewpoint_cam.original_image
                img = torch.mean(img, dim = 0, keepdim=True)
                data = {'image0': lst_img[None].cuda().float(),
                'image1': img[None].cuda().float()}
                with torch.no_grad():
                    matcher(data, online_resize=True)
                corr0, corr1 = data['mkpts0_f'].cpu().numpy(), data['mkpts1_f'].cpu().numpy()
                semantic_mask = viewpoint_cam.original_semantic_mask
                wrist_mask = (semantic_mask[:, :, 1] > 0.5).cpu().numpy()
                shaft_mask = (semantic_mask[:, :, 0] > 0.5).cpu().numpy()
                # gripper_mask = (semantic_mask[:, :, 2] > 0.5).cpu().numpy()
                # Filter matches based on masks
                wrist_mask = lst_wrist_mask[corr0[:, 1].astype(int), corr0[:, 0].astype(int)] * wrist_mask[corr1[:, 1].astype(int), corr1[:, 0].astype(int)]
                wrist_corr0 = corr0[wrist_mask]
                wrist_corr1 = corr1[wrist_mask]
                
                shaft_mask = lst_shaft_mask[corr0[:, 1].astype(int), corr0[:, 0].astype(int)] * shaft_mask[corr1[:, 1].astype(int), corr1[:, 0].astype(int)]
                shaft_corr0 = corr0[shaft_mask]
                shaft_corr1 = corr1[shaft_mask]

                if len(wrist_corr0) > 4 or len(shaft_corr0) > 4:
                    corr0 = wrist_corr0 if len(wrist_corr0) > 4 else shaft_corr0
                    corr1 = wrist_corr1 if len(wrist_corr0) > 4 else shaft_corr1
                    with torch.no_grad():
                        depth_map = render_depth.detach().cpu().numpy()
                        K = viewpoint_cam.K
                        rot_prev, translation_prev, alpha_prev, theta_l_prev, theta_r_prev = pose.get_pose()
                        del pose
                        corr0_3d = pixel_to_camera_coords(corr0, depth_map/1000., K)
                        corr0_3d = rot_prev.detach().cpu().numpy().T @ corr0_3d.T - rot_prev.detach().cpu().numpy().T @ translation_prev.detach().cpu().numpy()[:, None]
                        _, rvec, tvec, inliers = cv2.solvePnPRansac(corr0_3d.T, corr1, K, None)
                        R, _ = cv2.Rodrigues(rvec)
                        t = tvec[:, 0]
                        lst_img_np = (lst_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        img_np = (img.cpu().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                        display = demo_utils.draw_match(lst_img_np, img_np, corr0, corr1)
                        if os.path.exists("Matches") == False:
                            os.mkdir("Matches")
                        cv2.imwrite(f'Matches/matches{iteration}.png', display)
                        
                        pose = Pose(torch.from_numpy(R).cuda(), torch.from_numpy(t).cuda(), alpha_prev.clone(), theta_l_prev.clone(), theta_r_prev.clone())
                lst_img = img
                lst_wrist_mask = (semantic_mask[:, :, 1] > 0.5).cpu().numpy()
                lst_shaft_mask = (semantic_mask[:, :, 0] > 0.5).cpu().numpy()
            
            pose, render_semantics, render_depth, gt_image, gt_semantic_map,tips_coord \
                = track(instrument, pose, prev_pose, viewpoint_cam, opt, pipe, background, iteration)
            prev_rot, prev_trans, prev_alpha, prev_theta_l, prev_theta_r = pose.get_pose()
            prev_pose = Pose(prev_rot.clone(), prev_trans.clone(), prev_alpha.clone(), prev_theta_l.clone(), prev_theta_r.clone(), init_optimizer=False)
            save_folder_name = os.path.join(result_dir, "tracking_visualization")
            if os.path.exists(save_folder_name) == False:
                os.makedirs(save_folder_name)

            wrist_dice, shaft_dice, gripper_dice = instrument_dice(render_semantics, gt_semantic_map)
            # add to keyframe pool if good tracking 
            if wrist_dice > 0.86 and shaft_dice > 0.94:
                key_frame_pool[iteration] = {'pose_info': pose, 
                                             'viewpoint_cam': viewpoint_cam,
                                             'render_depth': render_depth}
            if wrist_dice < 0.01 and shaft_dice < 0.01 or wrist_dice < 0.7:
                print("Tracking failed: no overlapped regions")
                del pose
                reinit_from_keyframe = True
                if reinit_from_keyframe and len(key_frame_pool) > 0:
                    # find the most similar keyframe
                    max_kf_idx = -1
                    max_kf_matches = -1
                    img = viewpoint_cam.original_image
                    img = torch.mean(img, dim = 0, keepdim=True)
                    max_kf_idx = -1
                    max_kf_matches = -1
                    for kf_idx in key_frame_pool.keys():
                        kf_viewpoint_cam = key_frame_pool[kf_idx]['viewpoint_cam']
                        kf_img = kf_viewpoint_cam.original_image
                        kf_img = torch.mean(kf_img, dim = 0, keepdim=True)
                        data_kf = {'image0': kf_img[None].cuda().float(),
                        'image1': img[None].cuda().float()}
                        with torch.no_grad():
                            matcher(data_kf, online_resize=True)
                        kf_corr0, corr1 = data_kf['mkpts0_f'].cpu().numpy(), data_kf['mkpts1_f'].cpu().numpy()
                        semantic_mask = viewpoint_cam.original_semantic_mask
                        wrist_mask = (semantic_mask[:, :, 1] > 0.5).cpu().numpy()
                        shaft_mask = (semantic_mask[:, :, 0] > 0.5).cpu().numpy()
                        kf_semantic_mask = kf_viewpoint_cam.original_semantic_mask
                        kf_wrist_mask = (kf_semantic_mask[:, :, 1] > 0.5).cpu().numpy()
                        kf_shaft_mask = (kf_semantic_mask[:, :, 0] > 0.5).cpu().numpy()

                        wrist_mask = kf_wrist_mask[kf_corr0[:, 1].astype(int), kf_corr0[:, 0].astype(int)] \
                                     * wrist_mask[corr1[:, 1].astype(int), corr1[:, 0].astype(int)]
                        kf_wrist_corr0 = kf_corr0[wrist_mask]
                        wrist_corr1 = corr1[wrist_mask]

                        # shaft_mask = kf_shaft_mask[kf_corr1[:, 1].astype(int), kf_corr1[:, 0].astype(int)] * shaft_mask[kf_corr0[:, 1].astype(int), kf_corr0[:, 0].astype(int)]
                        # shaft_corr0 = kf_corr0[shaft_mask]
                        # shaft_corr1 = kf_corr1[shaft_mask]
                        
                        num_matches = len(kf_wrist_corr0)
                        if num_matches > max_kf_matches:
                            max_kf_matches = num_matches
                            max_kf_idx = kf_idx
                            max_wrist_corr1 = wrist_corr1
                            max_wrist_corr0 = kf_wrist_corr0
                    if max_kf_idx != -1 and max_kf_matches >= 5:
                        with torch.no_grad():
                            kf_viewpoint_cam = key_frame_pool[max_kf_idx]['viewpoint_cam']
                            K = kf_viewpoint_cam.K
                            kf_render_depth = key_frame_pool[max_kf_idx]['render_depth']
                            depth_map = kf_render_depth.detach().cpu().numpy()
                            pose_kf = key_frame_pool[max_kf_idx]['pose_info']
                            rot_prev, translation_prev, alpha_prev, theta_l_prev, theta_r_prev = pose_kf.get_pose()
                            del pose_kf
                            corr0_3d = pixel_to_camera_coords(max_wrist_corr0, depth_map/1000., K)
                            corr0_3d = rot_prev.detach().cpu().numpy().T @ corr0_3d.T - rot_prev.detach().cpu().numpy().T @ translation_prev.detach().cpu().numpy()[:, None]
                            _, rvec, tvec, inliers = cv2.solvePnPRansac(corr0_3d.T, max_wrist_corr1, K, None)
                            R, _ = cv2.Rodrigues(rvec)
                            t = tvec[:, 0]
                            lst_img_np = (lst_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                            img_np = (img.cpu().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                            display = demo_utils.draw_match(lst_img_np, img_np, max_wrist_corr0, max_wrist_corr1)
                            if os.path.exists("Matches") == False:
                                os.mkdir("Matches")
                            cv2.imwrite(f'Matches/matches{iteration}.png', display)
                            pose = Pose(torch.from_numpy(R).cuda(), torch.from_numpy(t).cuda(), alpha_prev.clone(), theta_l_prev.clone(), theta_r_prev.clone())
                            print(f"Re-localizing from keyframe {max_kf_idx} with {max_kf_matches} matches")
                        # pose = key_frame_pool[max_kf_idx]['pose_info']
                    else:
                        # rot0, trans0, alpha0, theta_l0, theta_r0 = pose.get_pose()
                        last_max_idx = max(key_frame_pool.keys())
                        print(f"Re-localization failed, use max keyframe {max_kf_idx}")
                        pose = key_frame_pool[max_kf_idx]['pose_info']
                else:
                    print("Re-localization failed, no keyframe available")
                    raise ValueError("Tracking failed and no re-localization possible")
               
                    # rot0, trans0, alpha0, theta_l0, theta_r0 = pose.get_pose()
                    # pose = Pose(rot0.clone(), trans0.clone(), alpha0.clone(), theta_l0.clone(), theta_r0.clone())
                pose, render_semantics, render_depth, gt_image, gt_semantic_map, tips_coord\
                      = track(instrument, pose, prev_pose, viewpoint_cam, opt, pipe, background, iteration)
                prev_pose = None
                wrist_dice, shaft_dice, gripper_dice = instrument_dice(render_semantics, gt_semantic_map)
            instrument_states = pose.get_pose()
            pose_info = Pose(*instrument_states, init_optimizer=False)
            if wrist_dice < 0.8  or shaft_dice < 0.9 or shaft_dice == 1 or wrist_dice == 1:
                pose = Pose(rot0.clone(), trans0.clone(), alpha0.clone(), theta_l0.clone(), theta_r0.clone())
                lst_img = viewpoint_stack[0].original_image
                lst_img = torch.mean(lst_img, dim = 0, keepdim=True)
                lst_wrist_mask = viewpoint_stack[0].original_semantic_mask[:, :, 1].cpu().numpy() > 0.5
                lst_shaft_mask = viewpoint_stack[0].original_semantic_mask[:, :, 0].cpu().numpy() > 0.5
            # resize vis to half of the resolution
            gt_semantic_map = F.interpolate(gt_semantic_map.unsqueeze(0), scale_factor=0.5, mode="nearest").squeeze(0)
            render_semantics = F.interpolate(render_semantics.unsqueeze(0), scale_factor=0.5, mode="nearest").squeeze(0)
            # draw tip1 and tip2 on the gt_image
            gt_image_np = (gt_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            if tips_coord is not None:
                tips1_coord = tips_coord[0].cpu().numpy()
                tips2_coord = tips_coord[1].cpu().numpy()
                cv2.circle(gt_image_np, (int(tips1_coord[0]), int(tips1_coord[1])), radius=5, color=(255, 0, 0), thickness=-1)
                cv2.circle(gt_image_np, (int(tips2_coord[0]), int(tips2_coord[1])), radius=5, color=(0, 0, 255), thickness=-1)
            gt_image = torch.from_numpy(gt_image_np).permute(2, 0, 1).float() / 255.0
            gt_image = F.interpolate(gt_image.unsqueeze(0).to(gt_semantic_map), scale_factor=0.5, mode="bilinear", align_corners=True).squeeze(0)
            vis = torch.cat([gt_image*0.7 + gt_semantic_map*0.3, gt_image*0.7 + render_semantics*0.3], dim=2)
            
            save_image(vis, save_folder_name+f"/tracking_{iteration}.png")
            print(f"Tracking iteration {iteration} wrist_dice: {wrist_dice} shaft_dice: {shaft_dice}")
            memory_pool[iteration] = {'pose_info': pose_info, 'dice': (wrist_dice, shaft_dice, gripper_dice)}
        # save the memory pool
        torch.save(memory_pool, memory_pool_name)

    # quantify the dice of each part
    # wrist_dices = torch.tensor(wrist_dices)
    # shaft_dices = torch.tensor(shaft_dices)
    # gripper_dices = torch.tensor(gripper_dices)
    # print("Wrist Dice: Mean =", wrist_dices.mean().item(), "Std Dev =", wrist_dices.std().item())
    # print("Shaft Dice: Mean =", shaft_dices.mean().item(), "Std Dev =", shaft_dices.std().item())
    # print("Gripper Dice: Mean =", gripper_dices.mean().item(), "Std Dev =", gripper_dices.std().item())

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument( '--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--gs_type', type=str, default="gs_mesh")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--meshes", nargs="+", type=str, default=[])
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--task_name", type=str, default='surgpose7_tracking')
    parser.add_argument(
        "--pretrain_params",
        type=str,
        default="pretrain_param_3dgs",
        choices=("pretrain_param_3dgs", "pretrain_3dgs_dense_param"),
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--save_xyz", action='store_true')

    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.num_splats = args.num_splats
    lp.meshes = args.meshes
    lp.gs_type = args.gs_type

    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        args.gs_type,
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.save_xyz, args.task_name,
        args.pretrain_params, args.max_frames
    )

    print("\nTraining complete.")
