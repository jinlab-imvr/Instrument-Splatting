import os
import torch
from utils.loss_utils import l1_loss, instrument_dice
from renderer.gaussian_renderer import render_with_instrument_v3 as render
import sys
from instrument_splatting.scene import InstrumentScene
from utils.instrument import Instrument
from instrument_splatting import (
    optimizationParamTypeCallbacks,
    gaussianModel
)
import torch.nn.functional as F
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.graphics_utils import build_rotation, matrix_to_quaternion
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from torchvision.utils import save_image
from submodules.ml_aspanformer.src.ASpanFormer.aspanformer import ASpanFormer 
from submodules.ml_aspanformer.src.config.default import get_cfg_defaults
from submodules.ml_aspanformer.src.utils.misc import lower_config
# from submodules.ml_aspanformer.demo import demo_utils
import numpy as np

import cv2
from collections import deque
from scipy.ndimage import label

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
        loss = kept_pred.mean(dim=1) + kept_tgt.mean(dim=1)
    return loss[0]

def pixel_to_camera_coords(pixel_coords, depth_map, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (pixel_coords[:, 0] - cx) * depth_map[pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)] / fx
    y = (pixel_coords[:, 1] - cy) * depth_map[pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)] / fy
    z = depth_map[pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)]
    return np.vstack((x, y, z)).T

def get_gripper_tips(instrument: Instrument, gripper_segmentation_mask, 
                     wrist_segmentation_mask, with_erosion = True, repeatTimes = 1):
    """
    Get the tips of the instrument from the segmentation mask
    """
    # Label connected components
    labeled_mask, num_features = label(gripper_segmentation_mask)
    # Find the two largest components
    component_sizes = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
    largest_components = sorted(range(1, num_features + 1), key=lambda i: component_sizes[i - 1], reverse=True)[:2]
    # Get the coordinates of the two largest components
    if num_features > 1:
        gripper_coords1 = np.argwhere(labeled_mask == largest_components[0])
        gripper_coords2 = np.argwhere(labeled_mask == largest_components[1])
        if len(gripper_coords2) < 0.3 * len(gripper_coords1):
            num_features = 1
        else:
            # Compute the centroids of the two components
            centroid1 = np.mean(gripper_coords1, axis=0)
            centroid2 = np.mean(gripper_coords2, axis=0)

            # Center the coordinates by subtracting the centroids
            centered_coords1 = gripper_coords1 - centroid1
            centered_coords2 = gripper_coords2 - centroid2

            svd_centered_coords1 = centered_coords1
            svd_centered_coords2 = centered_coords2

    if num_features ==1:
        gripper_coords = np.argwhere(labeled_mask == largest_components[0])
        centroid = np.mean(gripper_coords, axis=0)
        centered_coords = gripper_coords - centroid
        # erode the labeled_mask
        if with_erosion:
            for k in range(1,7):
                eroded_labeled_mask = labeled_mask == largest_components[0]
                eroded_labeled_mask = cv2.erode(eroded_labeled_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=k)
                labeled_mask2, num_features2 = label(eroded_labeled_mask)
                num_features = num_features2
                component_sizes2 = [(labeled_mask2 == i).sum() for i in range(1, num_features2 + 1)]
                if num_features2 == 1:
                    continue
                elif num_features2 < 1:
                    break
                largest_components2 = sorted(range(1, num_features2 + 1), key=lambda i: component_sizes2[i - 1], reverse=True)[:2]
                gripper_coords1 = np.argwhere(labeled_mask2 == largest_components2[0])
                gripper_coords2 = np.argwhere(labeled_mask2 == largest_components2[1])
                if num_features2 > 1 and len(gripper_coords2) > 0.1 * len(gripper_coords1):
                    
                    # Compute the centroids of the two components
                    centroid1 = np.mean(gripper_coords1, axis=0)
                    centroid2 = np.mean(gripper_coords2, axis=0)             
                    # Center the coordinates by subtracting the centroids
                    centered_coords1 = gripper_coords1 - centroid1
                    centered_coords2 = gripper_coords2 - centroid2
                    svd_centered_coords1 = centered_coords1
                    svd_centered_coords2 = centered_coords2
                    break

        # repeatTimes = 2
        if num_features == 1 or len(gripper_coords2) < 0.1 * len(gripper_coords1): 
            gripper_coords = np.argwhere(labeled_mask==largest_components[0])            
            for i in range(repeatTimes):
                
                centroid = np.mean(gripper_coords, axis=0)
                centered_coords = gripper_coords - centroid
                _, _, vh = np.linalg.svd(centered_coords)
                principal_axis = vh[0]
                
                # Use the wrist mask to define the positive direction of the principal axis
                wrist_coords = np.argwhere(wrist_segmentation_mask > 0)
                wrist_centroid = np.mean(wrist_coords, axis=0)
                direction_vector = centroid - wrist_centroid
                # Ensure the principal axis points from wrist to gripper
                if np.dot(principal_axis, direction_vector) < 0:
                    principal_axis = -principal_axis

                # get the orthogonal vector of principal axis
                orthogonal_vector = np.array([[0, -1],[1, 0]]) @ principal_axis[:,None]
                orthogonal_vector = orthogonal_vector[:,0]
                # Project the centered coordinates onto the orthogonal vector
                projections = np.dot(centered_coords, orthogonal_vector)

                # projected points with coord < 0 belongs to left gripper
                gripper_coords1 = gripper_coords[projections < 0]
                gripper_coords2 = gripper_coords[projections >= 0]
                gripper_coords = gripper_coords1 if len(gripper_coords1) > len(gripper_coords2) else gripper_coords2
                # Compute the centroids of the two components
                centroid1 = np.mean(gripper_coords1, axis=0)
                centroid2 = np.mean(gripper_coords2, axis=0)

                # Center the coordinates by subtracting the centroids
                centered_coords1 = gripper_coords1 - centroid1
                centered_coords2 = gripper_coords2 - centroid2
                svd_centered_coords1 = centered_coords1
                svd_centered_coords2 = centered_coords2

    # Perform SVD on both components

    _, _, vh1 = np.linalg.svd(svd_centered_coords1)
    _, _, vh2 = np.linalg.svd(svd_centered_coords2)

    # Project the centered coordinates onto the principal axis
    principal_axis1 = vh1[0]
    principal_axis2 = vh2[0]
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

    # # Create a blank image
    # image = np.zeros((512, 640, 3), dtype=np.uint8)
    # # Draw the gripper on the image
    # for coord in gripper_coords2:
    #     cv2.circle(image, (coord[1], coord[0]), 1, (0, 255, 0), -1)  # Draw a small circle at each coordinate
    # # Display the image
    # cv2.line(image, (tip1[1], tip1[0]), (tip2[1], tip2[0]), (255, 0, 0), 2)
    # cv2.imwrite('test.png', image)

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


def track(instrument: Instrument, pose: Pose, viewpoint_cam, opt, pipe, background, global_iteration, exp_path):
    """
    Perform render and compare.
    """
    first_iter = 1
    
    tracking_iterations = opt.tracking_iterations + 1 if global_iteration == 0 else 2000
    progress_bar = tqdm(range(first_iter, tracking_iterations), desc="Tracking progress")
    
    min_loss = 10000

    gt_image = viewpoint_cam.original_image.cuda()
    gt_semantic_map = viewpoint_cam.original_semantic_mask.cuda()
    gt_semantic_map = gt_semantic_map.permute(2, 0, 1)

    gt_gripper_mask = gt_semantic_map[2] > 0.5
    gt_wrist_mask = gt_semantic_map[1] > 0.5
    if gt_gripper_mask.sum() > 10:
        tips1_coord, tips2_coord = get_gripper_tips(instrument, 
                        (gt_gripper_mask.detach().cpu()).numpy(), 
                        (gt_wrist_mask.detach().cpu()).numpy(), with_erosion = False, repeatTimes = 1)
        tips1_coord_erosion, tips2_coord_erosion = get_gripper_tips(instrument, 
                        (gt_gripper_mask.detach().cpu()).numpy(), 
                        (gt_wrist_mask.detach().cpu()).numpy(), with_erosion=True, repeatTimes = 1)
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


        tips1_loss = 0
        tips2_loss = 0
        if tips1_coord is not None:
            if isinstance(tips1_coord, torch.Tensor) == False:
                tips1_coord = torch.from_numpy(tips1_coord).cuda()
                tips2_coord = torch.from_numpy(tips2_coord).cuda()
            l_gripper_keypoint = instrument.part_dict["l_gripper"].render_params["keypoints"] # 1x3
            r_gripper_keypoint = instrument.part_dict["r_gripper"].render_params["keypoints"]  # 1x3
            K = torch.from_numpy(viewpoint_cam.K).to(l_gripper_keypoint.device).float()
            l_gripper_keypoint_2d = K @ l_gripper_keypoint.T
            r_gripper_keypoint_2d = K @ r_gripper_keypoint.T
            l_gripper_keypoint_2d = l_gripper_keypoint_2d[:2] / l_gripper_keypoint_2d[2]
            r_gripper_keypoint_2d = r_gripper_keypoint_2d[:2] / r_gripper_keypoint_2d[2]
            # keypoint loss
            
            tips1_loss = torch.norm(l_gripper_keypoint_2d[:,0] - torch.flip(tips2_coord, dims=[0]))
            if tips1_loss < torch.norm(l_gripper_keypoint_2d[:,0] - torch.flip(tips1_coord, dims=[0])):
                tips2_loss = torch.norm(r_gripper_keypoint_2d[:,0] - torch.flip(tips1_coord, dims=[0]))
            else:
                tips2_loss = torch.norm(r_gripper_keypoint_2d[:,0] - torch.flip(tips2_coord, dims=[0]))
                tips1_loss = torch.norm(l_gripper_keypoint_2d[:,0] - torch.flip(tips1_coord, dims=[0]))
            r = 40 if iteration < 1000 else 20
            tips1_loss = torch.relu(tips1_loss - r)
            tips2_loss = torch.relu(tips2_loss - r)
            # gripper_keypoints_2d = torch.stack([l_gripper_keypoint_2d[...,0], r_gripper_keypoint_2d[...,0]], dim=0)
            # tips_coord = torch.stack([torch.flip(tips1_coord, dims=[0]), torch.flip(tips2_coord, dims=[0])], dim=0)
            # tool_tip_loss = chamfer_loss_loose_reg(gripper_keypoints_2d.unsqueeze(0), tips_coord.unsqueeze(0), radius=r)
        image = torch.cat([image[0:1] * shaft_mask, image[1:2] * wrist_mask, image[2:3] * gripper_mask], dim=0)
        loss_silhouette = l1_loss(image, gt_semantic_map)

        if global_iteration < 6:
            loss = loss_silhouette + structure_loss + 5e-5 * tips1_loss + 5e-5 * tips2_loss
        else:
            loss = loss_silhouette + structure_loss + 5e-5 * tips1_loss + 5e-5 * tips2_loss
        loss.backward()
        pose.step_optimizer()
        pose.zero_grad()
        if min_loss > loss_silhouette.item():
            min_loss = loss_silhouette.item()
            loss_queue = deque(maxlen=200)
        loss_queue.append(loss_silhouette.item())
        if len(loss_queue) == 150:
            if min_loss <= loss_queue[-1]:
                break
        image = 1.0 * (image > 0).detach()
        with torch.no_grad():
            # Debug: visualize the render&compare progress
            # if iteration % 30 == 0 or iteration == 1:
            #     # save_image(diff, f"difference/difference{iteration}.png")
            #     if os.path.exists(exp_path + "/difference") == False:
            #         os.mkdir(exp_path + "/difference")
            #     vis_image = 0.7*gt_image + 0.3 * gt_semantic_map
            #     # vis_image = 0.7*gt_image + 0.3 * image
            #     # vis_image_np = (vis_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            #     # cv2.circle(vis_image_np, ( int(l_gripper_keypoint_2d[1]), int(l_gripper_keypoint_2d[0]) ),radius=5, color=(255, 0, 0), thickness=-1)
            #     save_image(vis_image, exp_path + "/difference/render_semantics.png")
            #     render_semantics = image * 1.0
            #     save_image(0.7 * gt_image + 0.3 * image, exp_path + f"/difference/render_semantics{iteration}.png")

            ema_loss_for_log = loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Tracking Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
    with torch.no_grad():
        rot, trans, alpha, theta_l, theta_r = pose.get_pose()
        del pose
    pose_new = Pose(rot.clone(), trans.clone(), alpha.clone(), theta_l.clone(), theta_r.clone())
    return pose_new, image.detach(), render_depth.detach(),\
            gt_image.detach(), gt_semantic_map.detach(), tips1_coord, tips2_coord



def training(gs_type, dataset, opt, pipe, checkpoint, exp_name):
    first_iter = 0
    assert exp_name is not None, "Please provide a path to save the experiment results."
    exp_path = os.path.join(dataset.exp_dir, exp_name)
    if os.path.exists(exp_path) == False:
        os.makedirs(exp_path)
    scene = InstrumentScene(dataset, gaussianModel[gs_type], dataset.sh_degree,
                             pretrain_path = dataset.pretrain_path, pretrain_params='pretrain_param_3dgs')
    instrument = scene.instrument
    instrument.training_setup(opt)
    matcher_config = get_cfg_defaults()
    matcher_config.merge_from_file('submodules/ml_aspanformer/configs/aspan/outdoor/aspan_test.py')
    _config = lower_config(matcher_config)
    matcher = ASpanFormer(config=_config['aspan'])
    state_dict = torch.load('submodules/ml_aspanformer/weights/outdoor.ckpt', map_location='cpu')['state_dict']
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
    viewpoint_cam = viewpoint_stack[0]
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

    for iteration, viewpoint_cam in enumerate(viewpoint_stack):
        img_name = viewpoint_cam.image_name
        img_idx = int(img_name.split('frame')[-1])
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

                    ## Debug: save the matching results
                    # lst_img_np = (lst_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    # img_np = (img.cpu().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                    # display = demo_utils.draw_match(lst_img_np, img_np, corr0, corr1)
                    # if os.path.exists("Matches") == False:
                    #     os.mkdir("Matches")
                    # cv2.imwrite(f'Matches/matches{iteration}.png', display)
                    
                    pose = Pose(torch.from_numpy(R).cuda(), torch.from_numpy(t).cuda(), alpha_prev.clone(), theta_l_prev.clone(), theta_r_prev.clone())
            lst_img = img
            lst_wrist_mask = (semantic_mask[:, :, 1] > 0.5).cpu().numpy()
            lst_shaft_mask = (semantic_mask[:, :, 0] > 0.5).cpu().numpy()
        pose, render_semantics, render_depth, gt_image, gt_semantic_map,tips1_coord, tips2_coord = track(instrument, pose, viewpoint_cam, opt, pipe, background, iteration, exp_path)
        tips1_coord:np.ndarray
        tips2_coord:np.ndarray
        wrist_dice, shaft_dice, gripper_dice = instrument_dice(render_semantics > 0.5, gt_semantic_map)
        if wrist_dice < 0.01 and shaft_dice < 0.01:
            print("Tracking failed: no overlapped regions")
            del pose
            pose = Pose(rot0.clone(), trans0.clone(), alpha0.clone(), theta_l0.clone(), theta_r0.clone())
            pose, render_semantics, render_depth, gt_image, gt_semantic_map,tips1_coord, tips2_coord = track(instrument, pose, viewpoint_cam, opt, pipe, background, iteration, exp_path)
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
        if tips1_coord is not None:
            cv2.circle(gt_image_np, (int(tips1_coord[1]), int(tips1_coord[0])), radius=5, color=(255, 0, 0), thickness=-1)
            cv2.circle(gt_image_np, (int(tips2_coord[1]), int(tips2_coord[0])), radius=5, color=(0, 0, 255), thickness=-1)
        gt_image = torch.from_numpy(gt_image_np).permute(2, 0, 1).float() / 255.0
        gt_image = F.interpolate(gt_image.unsqueeze(0).to(gt_semantic_map), scale_factor=0.5, mode="bilinear", align_corners=True).squeeze(0)
        vis = torch.cat([gt_image*0.7 + gt_semantic_map*0.3, gt_image*0.7 + render_semantics*0.3], dim=2)
        if os.path.exists(exp_path + '/track_visualization') == False:
            os.makedirs(exp_path + '/track_visualization')
        save_image(vis, exp_path + '/track_visualization' + f"/tracking_{iteration}.png")
        print(f"Tracking iteration {iteration} wrist_dice: {wrist_dice} shaft_dice: {shaft_dice}")
        memory_pool[iteration] = {'pose_info': pose_info, 'dice': (wrist_dice, shaft_dice,gripper_dice)}
    # save the memory pool
    torch.save(memory_pool, os.path.join(exp_path, "memory_pool.pth"))
    # print the average Dice score for each part 
    total_wrist_dice = 0
    total_shaft_dice = 0
    total_gripper_dice = 0
    for iter_idx in memory_pool:
        wrist_dice, shaft_dice, gripper_dice = memory_pool[iter_idx]['dice']
        total_wrist_dice += wrist_dice
        total_shaft_dice += shaft_dice
        total_gripper_dice += gripper_dice
    num_iters = len(memory_pool)
    print(f"Average wrist dice: {total_wrist_dice/num_iters}, Average shaft dice: {total_shaft_dice/num_iters}, Average gripper dice: {total_gripper_dice/num_iters}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gs_type', type=str, default="gs_instrument")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--exp_name", type = str, default="exp1")

    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.gs_type = args.gs_type

    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    training(
        args.gs_type,
        lp.extract(args), op.extract(args), pp.extract(args),
        args.start_checkpoint, args.exp_name
    )

    print("\nTraining complete.")