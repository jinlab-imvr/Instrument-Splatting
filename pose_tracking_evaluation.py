import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l1_loss_normalization, instrument_dice

from renderer.gaussian_renderer import render_with_instrument_v3 as render
import sys
from instrument_splatting.scene import Scene, InstrumentScene
from utils.instrument import Instrument
from instrument_splatting import (
    optimizationParamTypeCallbacks,
    gaussianModel
)
from utils.camera_utils import Camera
import torch.autograd as autograd
import torch.nn.functional as F
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.graphics_utils import build_rotation, matrix_to_quaternion
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from torchvision.utils import save_image
from submodules.ml_aspanformer.src.ASpanFormer.aspanformer import ASpanFormer 
from submodules.ml_aspanformer.src.config.default import get_cfg_defaults
from submodules.ml_aspanformer.src.utils.misc import lower_config
from submodules.ml_aspanformer.demo import demo_utils
import numpy as np
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

def training(gs_type, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, save_xyz):
    first_iter = 0
    scene = InstrumentScene(dataset, gaussianModel[gs_type], dataset.sh_degree, pretrain_params='pretrain_param_3dgs')
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
    white_background = True
    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # iter_start = torch.cuda.Event(enable_timing=True)
    # iter_end = torch.cuda.Event(enable_timing=True)
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
    # endovis17
    memory_pool_list = ['memory_pool_3dgs_endovis2017', 'memory_pool_3dgs_endovis2017_keypoints',
                        'memory_pool_mesh_endovis2017']
    memory_pool_list = ['memory_pool_3dgs_wo_soft_endovins17','memory_pool_3dgs_wo_soft_endovins17']
    # memory_pool_list = ['memory_pool_3dgs_surgpose_4','memory_pool_3dgs_keypoints_surgpose_4']
    # memory_pool_list = ['memory_pool_3dgs_keypoint_surgpose_1','memory_pool_3dgs_keypoint_surgpose_1']
    # memory_pool_list = [ 'memory_pool_3dgs_endovis2017_keypoints', 'memory_pool_3dgs_endovis2017_keypoints']
    # surgpose_4
    memory_pool_list = ['memory_pool_3dgs_endovins18',"memory_pool_3dgs_keypoint_endovins18"]
    memory_pool_list = ['memory_pool_3dgs_surgpose_5','memory_pool_3dgs_keypoint_surgpose_5']
    
    # load the memory pool
    # memory_pool_list = [torch.load(mp+'.pth')  for mp in memory_pool_list]
    
    # img_num = len(list(memory_pool_list[0].keys()))
    # texture learning
    progress_bar = tqdm(range(first_iter, opt.texture_learning_iterations), desc="Tracking progress")
    named_iteration = 0
    name_iteration = False
    semantic_silhouette0, depth0 = None, None
    original_geometry_dict = {}
    
    # iterations = list(memory_pool_list[0].keys())
    # final_memory_pool = {}
    task_name = 'surgpose6_pose_half'
    mp = 'Results/' + task_name + '/memory_pool_updated.pth'#'/memory_pool_updated.pth'
    pose_evaluation_dir = 'Results/' + task_name + '/pose_evaluation'
    if not os.path.exists(pose_evaluation_dir):
        os.makedirs(pose_evaluation_dir)
    final_memory_pool = torch.load(mp)
    iterations = list(final_memory_pool.keys())
    psnr_stacks = []
    psnr_dict = {}
    wrist_dices = []
    shaft_dices = []
    gripper_dices = []
    updated_wrist_dices = []
    updated_shaft_dices = []
    updated_gripper_dices = []
    for iteration in iterations:
        final_memory_pool[iteration]['pose_info']
        pose_info, dice = final_memory_pool[iteration]['pose_info'], final_memory_pool[iteration]['dice']
        wrist_dice, shaft_dice, gripper_dice = dice 
        wrist_dices.append(wrist_dice)
        shaft_dices.append(shaft_dice)
        gripper_dices.append(gripper_dice)

        viewpoint_cam = viewpoint_stack[iteration]
        instrument.forward_kinematics(*pose_info.get_pose(), gs_grad=False, pose_grad = False, with_mesh=True)
        render_img, depth = instrument.render_trimesh(viewpoint_cam.K)

        transformation = torch.eye(4, device="cuda")
        gt_semantic_map = viewpoint_cam.original_semantic_mask.cuda()
        gt_semantic_map = gt_semantic_map.permute(2, 0, 1)
        with torch.no_grad():
            transformation = torch.eye(4, device="cuda")
            transformation[:3, :3] = rot
            transformation[:3, 3] = trans 
            override_camera_center = torch.inverse(transformation)[:3, 3] * 1000
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            render_pkg = render(viewpoint_cam, instrument, pipe, bg*0, 
                override_camera_center=override_camera_center, render_semantics_silhouette=True)
            semantic_silhouette = render_pkg["render"].permute(2, 0, 1)
            wrist_mask = semantic_silhouette[1] > 0.5
            shaft_mask = semantic_silhouette[0] > 0.5
            gripper_mask = semantic_silhouette[2] > 0.5
            semantic_silhouette = torch.cat([semantic_silhouette[0:1] * shaft_mask, semantic_silhouette[1:2] * wrist_mask, semantic_silhouette[2:3] * gripper_mask], dim=0)
            silhouette = torch.sum(semantic_silhouette, dim=0)
            gt_semantic_map = viewpoint_cam.original_semantic_mask.cuda()
            gt_semantic_map = gt_semantic_map.permute(2, 0, 1)
            gt_silhouette = torch.sum(gt_semantic_map, dim=0)
            semantic_silhouette = (semantic_silhouette > 0.0).detach().float()
            wrist_dice, shaft_dice, gripper_dice = instrument_dice(semantic_silhouette, gt_semantic_map)
            final_memory_pool[iteration]['updated_dice'] = (wrist_dice, shaft_dice, gripper_dice)
            updated_wrist_dices.append(wrist_dice)
            updated_shaft_dices.append(shaft_dice)
            updated_gripper_dices.append(gripper_dice)

        depth_map = depth
        depth_mask = depth_map > 0
        
        gt_image = viewpoint_cam.original_image.cpu().numpy()
        vis_image = gt_image.copy()
        render_img = render_img.transpose(2, 0, 1)
        vis_image[:, depth_mask]  = render_img[:, depth_mask]/255.
        if not os.path.exists('Results/' + task_name + '/pose_visualization'):
            os.makedirs('Results/' + task_name + '/pose_visualization')
        if not os.path.exists('Results/' + task_name + '/depth_visualization'):
            os.makedirs('Results/' + task_name + '/depth_visualization')
        if not os.path.exists('Results/' + task_name + '/depth'):
            os.makedirs('Results/' + task_name + '/depth')
        if not os.path.exists('Results/' + task_name + '/semantic_silhouette'):
            os.makedirs('Results/' + task_name + '/semantic_silhouette')
           
        
        save_image(torch.from_numpy(vis_image), 'Results/' + task_name + '/pose_visualization/frame{:03d}.png'.format(iteration))
        # k = iteration+200+60 if iteration+200+60 < 300 else iteration+100+60
        cv2.imwrite( 'Results/' + task_name + '/depth/frame{:03d}.png'.format(iteration), (depth_map*1000).astype(np.uint8))
        visualize_depth_map(depth_map,  'Results/' + task_name + '/depth_visualization/frame{:03d}.png'.format(iteration))
        save_image(0.7*torch.from_numpy(gt_image).to(semantic_silhouette) + 0.3 * semantic_silhouette, 'Results/' + task_name + '/semantic_silhouette/frame{:03d}.png'.format(iteration))
        
    # quantify the dice of each part
    wrist_dices = torch.tensor(wrist_dices)
    shaft_dices = torch.tensor(shaft_dices)
    gripper_dices = torch.tensor(gripper_dices)
    updated_wrist_dices = torch.tensor(updated_wrist_dices)
    updated_shaft_dices = torch.tensor(updated_shaft_dices)
    updated_gripper_dices = torch.tensor(updated_gripper_dices)
    print("Wrist Dice: Mean =", wrist_dices.mean().item(), "Std Dev =", wrist_dices.std().item())
    print("Shaft Dice: Mean =", shaft_dices.mean().item(), "Std Dev =", shaft_dices.std().item())
    print("Gripper Dice: Mean =", gripper_dices.mean().item(), "Std Dev =", gripper_dices.std().item())
    print("Updated Wrist Dice: Mean =", updated_wrist_dices.mean().item(), "Std Dev =", updated_wrist_dices.std().item())
    print("Updated Shaft Dice: Mean =", updated_shaft_dices.mean().item(), "Std Dev =", updated_shaft_dices.std().item())
    print("Updated Gripper Dice: Mean =", updated_gripper_dices.mean().item(), "Std Dev =", updated_gripper_dices.std().item())

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--gs_type', type=str, default="gs_instrument")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--meshes", nargs="+", type=str, default=[])
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
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
        args.start_checkpoint, args.debug_from, args.save_xyz
    )

    print("\nTraining complete.")