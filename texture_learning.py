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
import torch.nn.functional as F
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import build_rotation, matrix_to_quaternion
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from torchvision.utils import save_image
from submodules.ml_aspanformer.src.ASpanFormer.aspanformer import ASpanFormer 
from submodules.ml_aspanformer.src.config.default import get_cfg_defaults
from submodules.ml_aspanformer.src.utils.misc import lower_config
import numpy as np
import random
import cv2
from collections import deque
from scipy.ndimage import label
from utils.loss_utils import dice as Dice

remove_cases = []
def pixel_to_camera_coords(pixel_coords, depth_map, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (pixel_coords[:, 0] - cx) * depth_map[pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)] / fx
    y = (pixel_coords[:, 1] - cy) * depth_map[pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)] / fy
    z = depth_map[pixel_coords[:, 1].astype(int), pixel_coords[:, 0].astype(int)]
    return np.vstack((x, y, z)).T


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
    error_id = []
    for i, point in enumerate(points_2d):
        try:
            cv2.circle(image, (point[0], point[1]), radius=3, color=(0, 255, 0), thickness=-1)
        except:
            print("Error drawing circle:", point)
            error_id.append(i)

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
                {'params': [self.alpha], 'lr': 0.02, 'name': 'alpha'},
                {'params': [self.theta_l], 'lr': 0.1, 'name': 'theta_l'},
                {'params': [self.theta_r], 'lr': 0.1, 'name': 'theta_r'}
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

def texture_learning(instrument: Instrument, pose: Pose, viewpoint_cam, opt, pipe, 
                        background, global_iteration, semantic_silhouette0, depth0, testing, lateral_inversion = False):
    first_iter = 1
    # semantic_silhouette0 = None
    min_loss = 10000
    end_iteration = 2
    min_loss_rgb = 10000
    # loss_rgb_iters = deque(maxlen=50)
    
    psnr_max_dict = {'psnr':0, 'pose': None}
    psnrs = []
    losses = []
    for iteration in range(first_iter, end_iteration):
        # Render
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        rot, trans, alpha, theta_l, theta_r = pose.get_pose()
        
        if lateral_inversion:
            instrument.forward_kinematics(rot @ rotation_matrix_x(180), trans, -alpha, theta_r, theta_l, gs_grad=True, pose_grad = True)
            transformation = torch.eye(4, device="cuda")
            transformation[:3, :3] = rot @ rotation_matrix_x(180)
            transformation[:3, 3] = trans 
        else:
            instrument.forward_kinematics(rot, trans, alpha, theta_l, theta_r, gs_grad=True, pose_grad = True)
            transformation = torch.eye(4, device="cuda")
            transformation[:3, :3] = rot
            transformation[:3, 3] = trans 
        override_camera_center = transformation[:3, 3] * 1000
        render_pkg  = render(viewpoint_cam, instrument, pipe, bg, 
                    override_camera_center=override_camera_center, active_sh_degree=None, render_semantics_silhouette=False)
        render_pkg2 = render(viewpoint_cam, instrument, pipe, bg-1, 
                    override_camera_center=override_camera_center, render_semantics_silhouette=True)
        semantic_silhouette = render_pkg2["render"].permute(2, 0, 1)
        depth = render_pkg2["depth"]
        image, opacity, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["opacity"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        image_name = viewpoint_cam.image_name
        # generate a dictionary to store the loss and psnr for each iteration
        
        gt_semantic_map = viewpoint_cam.original_semantic_mask.cuda()
        gt_semantic_map = gt_semantic_map.permute(2, 0, 1)  

        image = image.permute(2, 0, 1)
        psnr_val = psnr(image, gt_image)
        if psnr_val.mean().item() > psnr_max_dict['psnr']:
                psnr_max_dict['psnr'] = psnr_val.mean().item()
                psnr_max_dict['pose'] = pose.get_pose()
                if iteration == 1:
                    psnr_max_dict['init_psnr'] = psnr_val.mean().item()
                    psnr_max_dict['init_pose'] = pose.get_pose()
        if semantic_silhouette0 is not None:
            error_mask = (gt_semantic_map.sum(dim=0, keepdim=True) > 0) != (semantic_silhouette0.sum(dim=0, keepdim=True) > 0.5)
            error_mask = error_mask.repeat(3, 1, 1)
            gripper_error_mask = (gt_semantic_map[2] > 0.5) != (semantic_silhouette0[2] > 0.5)
            gripper_error_mask = gripper_error_mask.repeat(3, 1, 1)
            
            loss_rgb = l1_loss(image * (~gripper_error_mask) , gt_image * (~gripper_error_mask))
        else:
            loss_rgb = l1_loss(image, gt_image)
       
        semantic_silhouette_sup = semantic_silhouette0 if semantic_silhouette0 is not None else semantic_silhouette
        loss_silhouette = l1_loss(semantic_silhouette, semantic_silhouette_sup)


        loss = loss_rgb + loss_silhouette 
        
        if not testing:
            (loss).backward()
            instrument.step_optimizer()
            instrument.zero_grad(set_to_none=True)
            psnrs.append(psnr_val.mean().item())
            losses.append(loss.item())
        else:
            loss = torch.tensor([0.0], device="cuda")

    
    semantic_silhouette0 = semantic_silhouette.detach()
    depth0 = depth.detach()

    # if image_name in log_dict.keys():
    #         if 'psnrs' not in log_dict[image_name]:
    #             log_dict[image_name]['psnrs'] = []
    #         if 'losses' not in log_dict[image_name]:
    #             log_dict[image_name]['losses'] = []
    #         log_dict[image_name]['psnrs'].append(psnrs)
    #         log_dict[image_name]['losses'].append(losses)
    #         # save the log_dict to a npy file
    #         np.save(f'Results/{task_name}/texture_learning_pose_log.npy', log_dict)
    with torch.no_grad():
        rot, trans, alpha, theta_l, theta_r = psnr_max_dict['pose']
        # del psnr_max_dict['pose']
        # del pose
        psnr_max_dict['pose'] = None
        psnr_val = psnr_max_dict['psnr']
        pose_new = Pose(rot.clone(), trans.clone(), alpha.clone(), theta_l.clone(), theta_r.clone())
    
    return instrument, image, gt_image, psnr_val, semantic_silhouette0, depth0, pose_new
# instrument.forward_kinematics(rot @ rotation_matrix_x(180), trans, -alpha, theta_r, theta_l, gs_grad=True, pose_grad = False)
# render_pkg2 = render(viewpoint_cam, instrument, pipe, bg, 
#                     override_camera_center=override_camera_center, render_semantics_silhouette=False)
# semantic_silhouette = render_pkg2["render"].permute(2, 0, 1)
# save_image(semantic_silhouette,'test1.png')
def training(gs_type, dataset, opt, pipe, testing_iterations, checkpoint_iterations, exp_name):
    first_iter = 0
    assert exp_name is not None, "Please provide a path to save the experiment results."
    exp_path = os.path.join(dataset.exp_dir, exp_name)
    # if os.path.exists(exp_path) == False:
    #     os.makedirs(exp_path)
    scene = InstrumentScene(dataset, gaussianModel[gs_type], dataset.sh_degree,
                             pretrain_path = dataset.pretrain_path, pretrain_params='pretrain_param_3dgs')
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
    # load memory pool
    memory_pool_path = os.path.join(exp_path, f'memory_pool.pth')
    memory_pool = torch.load(memory_pool_path) 
    
    # texture learning
    progress_bar = tqdm(range(first_iter, opt.texture_learning_iterations), desc="Texture Learning")
    named_iteration = 0
    name_iteration = False
    semantic_silhouette0, depth0 = None, None
    original_geometry_dict = {}
    
    iterations = list(memory_pool.keys())
    final_memory_pool = {}

    for iteration in iterations:
        final_memory_pool[iteration] = memory_pool[iteration]
    psnr_dict = {}
    available_iterations = []
    for iteration in iterations:
        pose_info, dice = final_memory_pool[iteration]['pose_info'], final_memory_pool[iteration]['dice']
        wrist_dice, shaft_dice, gripper_dice = dice
        if iteration not in remove_cases  and wrist_dice > 0.8 and shaft_dice > 0.8:
            available_iterations.append(iteration)
    testing_iterations = random.sample(available_iterations, 12)
    if os.path.exists(f"{scene.source_path}/testing_iterations.txt") == False:
        with open(f"{scene.source_path}/testing_iterations.txt", "w") as f:
            for item in testing_iterations:
                f.write("%s\n" % item)
    else:
        testing_iterations = []
        with open(f"{scene.source_path}/testing_iterations.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                testing_iterations.append(int(line.strip()))
    print("====Testing iterations==: ", testing_iterations)

    
    if dataset.infer:
        available_iterations = [it for it in available_iterations if it in testing_iterations]
    else:
        available_iterations = [it for it in available_iterations if it not in testing_iterations]

    total_iterations = opt.texture_learning_iterations if dataset.infer \
            else len(available_iterations)

    for i in range(total_iterations):
        if not name_iteration:
            iteration = random.choice(available_iterations)
            if iteration in remove_cases:
                i = i -1
                continue
        else:
            iteration = named_iteration
        if dataset.infer:
            iteration = available_iterations[i]
        

        if i % 400 == 0 and i < 10000:
            instrument.oneupSHdegree()
        viewpoint_cam = viewpoint_stack[iteration]

        pose_info, dice = final_memory_pool[iteration]['pose_info'], final_memory_pool[iteration]['dice']
        wrist_dice, shaft_dice, gripper_dice = dice  
        testing = False if iteration not in testing_iterations else True
        
        
        joint_state = pose_info.get_pose()
        pose = Pose(*joint_state)
        if (i+1) % 2000 == 0 and i < 10000:
            instrument.update_learning_rate_texture_learning(i)
        if iteration not in original_geometry_dict.keys():
            semantic_silhouette0, depth0 = None, None
            instrument, image, gt_image, psnr_val, semantic_silhouette0, depth0, _ = texture_learning(instrument, pose,
                                viewpoint_cam, opt, pipe, background, i,semantic_silhouette0, depth0,testing)
            original_geometry_dict[iteration] = (semantic_silhouette0, depth0)
        else:
            semantic_silhouette0, depth0 = original_geometry_dict[iteration]
            
            instrument, image, gt_image, psnr_val, _, _, _ = texture_learning(instrument, pose,
                        viewpoint_cam, opt, pipe, background, i, semantic_silhouette0, depth0, testing)
            
            
        updated_joint_state = pose.get_pose()
        final_memory_pool[iteration]['pose_info'] = Pose(*updated_joint_state, init_optimizer=False)
        
        if i > 3000 and i % 10 == 0:
            
            if os.path.exists(os.path.join(exp_path, "texture_learning_visulization")) == False:
                os.mkdir(os.path.join(exp_path, "texture_learning_visulization"))
            # if testing and os.path.exists(os.path.join(exp_path, "texture_learning_visulization_test")) == False:
            #     os.mkdir(os.path.join(exp_path, "texture_learning_visulization_test"))
            vis = torch.cat([gt_image, image], dim=2)
            vis = F.interpolate(vis.unsqueeze(0), scale_factor=0.5, mode="bilinear", align_corners=True).squeeze(0)
            if testing:
                save_image(vis, os.path.join(exp_path, "texture_learning_visulization_test", f"texture_learning_{iteration}.png"))
            else:
                save_image(vis, os.path.join(exp_path, "texture_learning_visulization", f"texture_learning_{iteration}.png"))

        with torch.no_grad():
            if i % 10 == 0 or i == 1:
                if os.path.exists(os.path.join(exp_path, "text_learning")) == False:
                    os.mkdir(os.path.join(exp_path, "text_learning"))
                if testing:
                    if os.path.exists(os.path.join(exp_path, "text_learning_test")) == False:
                        os.mkdir(os.path.join(exp_path, "text_learning_test"))
                    save_image(image, os.path.join(exp_path, "text_learning_test", f"img{iteration}.png"))
                else:
                    save_image(image, os.path.join(exp_path, "text_learning", f"img{iteration}.png"))
            if isinstance(psnr_val, float):
                ema_loss_for_log = psnr_val
            else:
                ema_loss_for_log = psnr_val.mean().item()
            psnr_dict[iteration] = ema_loss_for_log
            if i % 10 == 0:
                progress_bar.set_postfix({"Texture Learning PSNR": f"{ema_loss_for_log:.{3}f}"})
                progress_bar.update(10)
            if (i in checkpoint_iterations):
                # calculate the mean error
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                for part_name, part in instrument.part_dict.items():
                    gs_model = part.gaussian_model
                    checkpoint = {
                        'opacity': gs_model._opacity,
                        'alpha': gs_model._alpha,
                        "xyz": gs_model._xyz,
                        'vertices': gs_model.vertices,
                        'rotation': gs_model._rotation,
                        'scale': gs_model._scale,
                        'scaling': gs_model._scaling,
                        'features_dc': gs_model._features_dc,
                        'features_rest': gs_model._features_rest
                    }
                    # torch.save(checkpoint, scene.source_path + "/" + str(part_name)+"_trained_3dgs_onethird_param" + str(i) + ".pth")
                    torch.save(checkpoint, os.path.join(exp_path, str(part_name)+"_trained_3dgs_param" + str(i) + ".pth"))
                    
     
            
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gs_type', type=str, default="gs_instrument")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--exp_name", type = str, default="exp1")
    parser.add_argument("--infer", default=False, action="store_true")

    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.gs_type = args.gs_type
    lp.infer = args.infer

    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])


    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    training(
        args.gs_type,
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.checkpoint_iterations,args.exp_name
    )

    print("\nTraining complete.")