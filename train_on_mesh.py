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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from renderer.gaussian_renderer import render_with_instrument_v2 as render
import sys
from scene import Scene, InstrumentScene
from games import (
    optimizationParamTypeCallbacks,
    gaussianModel
)
import numpy as np

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, load_viewcam_from_npz
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from utils.camera_utils import Camera
import cv2
from torchvision.utils import save_image
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def proj_2d(K, xyz, xyz_d, image = None):
    """
    Project 3D points to 2D points using camera intrinsic matrix K. And draw the 2D points in the 2D frame
    """
    # Project the points
    xyz = xyz/xyz[:, 2].unsqueeze(1)
    xyz_d = xyz_d/xyz_d[:, 2].unsqueeze(1)
    xyz = torch.matmul(K.float(), xyz.t()).t()
    xyz_d = torch.matmul(K.float(), xyz_d.t()).t()
    # Normalize the points
    xyz = xyz[:, :2] / xyz[:, 2].unsqueeze(1)
    xyz_d = xyz_d[:, :2] / xyz_d[:, 2].unsqueeze(1)
    
    # Convert to numpy
    points_2d = xyz.cpu().numpy().astype(int)
    points_2d_d = xyz_d.cpu().numpy().astype(int)
    
    # Create a blank image
    if image is None:
        image = 255*torch.ones((512, 640, 3), dtype=torch.uint8).cpu().numpy()
    
    # Draw points on the image
    for point in points_2d:
        cv2.circle(image, (point[0], point[1]), radius=1, color=(0, 255, 0), thickness=-1)
    # for point in points_2d_d:
    #     cv2.circle(image, (point[0], point[1]), radius=1, color=(0, 0, 255), thickness=-1)
    # Save or display the image
    cv2.imwrite('projected_points.png', image)
    # cv2.imshow('Projected Points', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

use_triangle_dict = {'shaft': False, 'wrist': True, 'l_gripper': True, 'r_gripper': True}
scale_coeff_dict = {'shaft': 1.0, 'wrist': 1000.0, 'l_gripper': 1000.0, 'r_gripper': 1000.0}  

def training(gs_type, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, save_xyz):
    first_iter = 0
    # gaussians = gaussianModel[gs_type](dataset.sh_degree)
    scene = InstrumentScene(dataset, gaussianModel[gs_type], dataset.sh_degree)
    instrument = scene.instrument
    # setup training for each part_gaussian
    # gaussians.training_setup(opt)
    instrument.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        instrument.restore(model_params, opt)
    white_backgorund = True
    bg_color = [1, 1, 1] if white_backgorund else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # Pick a random Camera
    part_name = 'shaft'
    npy_file_root = './instrument_mesh/' + str(part_name) + "_render"
    # npy_file_root = '/mnt/iMVR/shuojue/code/gaussian-mesh-splatting/instrument_mesh/shaft_render3'
    npz_files = sorted([f for f in os.listdir(npy_file_root) if f.endswith('.npz')])[1:]
    viewpoint_stack = []
    
    obj2world = getattr(instrument, f"{part_name}2world").to("cuda")
    for i, npz_file in enumerate(npz_files):
        npz_path = os.path.join(npy_file_root, npz_file)
        T0 = obj2world
        img, depth, mask, w2c, cam_K = load_viewcam_from_npz(npz_path, device='cuda')
        img = img.permute(2, 0, 1)
        img = img[:3]
        w2c = w2c @ obj2world
        viewpoint = Camera(colmap_id=i, R = w2c[:3,:3], T=w2c[:3,3]*1000, FoVx=None, FoVy=None, K=cam_K, 
                        image=img, world_view_transform = None, zfar=500, znear=1, depth=depth,
                        gt_alpha_mask=None, image_name=npz_file, uid=i, data_device="cuda")
        viewpoint_stack.append(viewpoint)
    gs_model = instrument.part_dict[part_name].gaussian_model
    flag = True
    render_err_dict = {}
    depth_err_dict = {}
    for iteration in range(first_iter, opt.iterations + 1):
        # os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        # if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
        #     torch.save(instrument.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")

        iter_start.record()

        instrument.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
            # instrument.oneupSHdegree()

        # Randomly select a viewpoint
        # viewpoint_stack = viewpoint_stack[:100]
        if flag:
            idx = randint(0, len(viewpoint_stack) - 1)
            viewpoint_cam = viewpoint_stack[idx]
        else:
            viewpoint_cam = viewpoint_stack[idx]
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        render_pkg = render(viewpoint_cam, gs_model,
                             pipe, bg, override_camera_center=None, render_depth_silhouette=False, scale_coef = 1.0)
        image, depth, opacity, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"], render_pkg["opacity"],\
              render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
        render_pkg2 = render(viewpoint_cam, gs_model,
                             pipe, bg, override_camera_center=None, render_depth_silhouette=True, scale_coef = 1.0)
        image2 = render_pkg2["render"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth = viewpoint_cam.original_depth.cuda() * 1000
        gt_silhouette = (gt_depth > 0).unsqueeze(0)*1.0
        mask = gt_depth > 0
        gt_image = gt_image * mask.unsqueeze(0)
        # set the mask of the gt_image to 1
        gt_image = gt_image + (1 - mask.unsqueeze(0)*1)
        image = image.permute(2, 0 ,1)
        image2 = image2.permute(2, 0 ,1)
        loss_silhouette = l1_loss(image2[1:2], gt_silhouette)
        loss_depth = l1_loss(depth, gt_depth)
        depth_err = torch.abs(depth - gt_depth)
        render_err = torch.abs(image[1:2] - gt_silhouette)
        rgb_loss = l1_loss(image, gt_image)
        loss =  rgb_loss # + loss_depth 
        scale = gs_model.get_scaling
        # penalize large scale with the threshold of 1
        # loss += 0.01 * torch.sum(torch.relu(scale - 0.5))
        loss.backward()
        # K = viewpoint_cam.K
        # xyz = gs_model.get_xyz.detach()
        # xyz_d = xyz #+ 0.001*gs_model.normals.detach()
        # w2c = viewpoint_cam.world_view_transform
        # xyz = torch.concat([xyz*1000, torch.ones((xyz.shape[0],1), device = xyz.device)], dim = 1) @ w2c
        # xyz_d = torch.concat([xyz_d*1000, torch.ones((xyz_d.shape[0],1), device = xyz_d.device)], dim = 1) @ w2c
        # vis_image = gt_silhouette.clone().permute(1,2,0).detach().cpu().numpy()
        # vis_image = (vis_image * 255).astype(np.uint8)
        # vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        # proj_2d(K, xyz[...,:3].detach(),xyz_d[...,:3].detach(), vis_image.astype(np.uint8))
        iter_end.record()
        if iteration > 15000:
            render_err_dict[idx] = render_err.mean().item()
            depth_err_dict[idx] = depth_err.mean().item()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log =  loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
            #                 testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            cameras_extent = 200 # mm
            # if iteration < opt.densify_until_iter and iteration % 500 == 0:
            #     # Keep track of max radii in image-space for pruning
            #     gs_model.max_radii2D[visibility_filter] = torch.max(gs_model.max_radii2D[visibility_filter],
            #                                                             radii[visibility_filter])
            #     gs_model.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gs_model.densify_and_prune(opt.densify_grad_threshold, 0.005, cameras_extent,
            #                                     size_threshold)

                # if iteration % opt.opacity_reset_interval == 0 or (
                #         dataset.white_background and iteration == opt.densify_from_iter):
                #     gs_model.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                
                gs_model.optimizer.step()
                gs_model.optimizer.zero_grad(set_to_none=True)
                # instrument.zero_grad(set_to_none=True)

            # TODO: Implement densification for the other types of instrument
            if (iteration in checkpoint_iterations):
                # calculate the mean error
                render_err = torch.tensor(list(render_err_dict.values()))
                depth_err = torch.tensor(list(depth_err_dict.values()))
                print(f"Mean render error: {render_err.mean().item()}")
                print(f"Mean depth error: {depth_err.mean().item()}")
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
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
                torch.save(checkpoint, scene.source_path + "/" + str(part_name)+"_pretrain_surfel_param" + str(iteration) + ".pth")
 
        if hasattr(gs_model, 'update_alpha'):
            gs_model.update_alpha()
        if hasattr(gs_model, 'prepare_scaling_rot'):
            gs_model.prepare_scaling_rot(use_triangle_dict[part_name])
    


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--gs_type', type=str, default="instrument_mesh")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--meshes", nargs="+", type=str, default=[])
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15000, 29000])
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
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        args.gs_type,
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.save_xyz
    )

    # All done
    print("\nTraining complete.")
