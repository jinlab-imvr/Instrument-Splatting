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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from instrument_splatting.scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import trimesh
from utils.instrument import Instrument

def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    viewpoint_camera.camera_center = viewpoint_camera.camera_center
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    _xyz = pc.get_xyz

    means3D = _xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def get_depth_and_silhouette(pts_3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)
    
    return depth_silhouette

def render_with_instrument_v3(viewpoint_camera, instrument: Instrument, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
                           override_color = None, override_camera_center = None,active_sh_degree = None,
                           render_semantics_silhouette = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # gaussian_dict = instrument.get_gaussian_models()
    # gaussian_list = list(gaussian_dict.values())

 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    render_params = instrument.get_render_params()
    _xyz = render_params['xyz'] * 1000

    if active_sh_degree is None:
        active_sh_degree = instrument.active_sh_degree
    

    screenspace_points = torch.zeros_like(_xyz, dtype=_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    if override_camera_center is not None:
        camera_center = override_camera_center
    else:
        camera_center = viewpoint_camera.camera_center
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos= torch.tensor([0,0,0],dtype=torch.float32,device='cuda'), #viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = _xyz
    means2D = screenspace_points
    opacity = render_params['opacity']
    features = render_params['feature']

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = instrument.get_covariance(scaling_modifier)
    else:
        scales = render_params['scale']
        rotations = render_params['rotation']


    if render_semantics_silhouette:
        # depth_silhouette = get_depth_and_silhouette(means3D, viewpoint_camera.world_view_transform.T)
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        if isinstance(instrument, Instrument):
            semantics = instrument.get_semantics
        else:
            semantics = torch.ones_like(instrument.get_xyz, dtype=torch.float32, device='cuda') 
            
        rendered_image, radii, depth, alpha = rasterizer(
            means3D = means3D,
            means2D = means2D,
            colors_precomp = semantics,
            opacities = opacity,
            scales = scales ,
            rotations = rotations,
            cov3D_precomp = None
            )
        
        rendered_image = rendered_image.permute(1, 2, 0)
        depth = depth.squeeze(0)
        # spotlight light source model
        # rendered_image = viewpoint_camera.spotlight_render(rendered_image, depth.detach())
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth":depth,
                "opacity": alpha.squeeze(0)}
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    
    if override_color is None:
        if True: # pipe.convert_SHs_python:
            shs_view = features.transpose(1, 2).view(-1, 3, (instrument.shaft.gaussian_model.max_sh_degree+1)**2)
            dir_pp = (_xyz - camera_center.repeat(features.shape[0], 1).cuda())
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = instrument.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, alpha = rasterizer(
            means3D = means3D,
            means2D = means2D,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales ,
            rotations = rotations,
            cov3D_precomp = None
            )
        
    rendered_image = rendered_image.permute(1, 2, 0)
    depth = depth.squeeze(0)
    # spotlight light source model
    # rendered_image = viewpoint_camera.spotlight_render(rendered_image, depth.detach())
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "opacity": alpha.squeeze(0)}

def render_with_instrument(viewpoint_camera, instrument: Instrument, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
                           override_color = None, override_camera_center = None,
                           render_semantics_silhouette = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # gaussian_dict = instrument.get_gaussian_models()
    # gaussian_list = list(gaussian_dict.values())

 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(instrument.get_xyz, dtype=instrument.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    if override_camera_center is not None:
        camera_center = override_camera_center
    else:
        camera_center = viewpoint_camera.camera_center
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=instrument.active_sh_degree,
        campos= torch.tensor([0,0,0],dtype=torch.float32,device='cuda'), #viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    _xyz = instrument.get_xyz * 1000

    means3D = _xyz
    means2D = screenspace_points
    opacity = instrument.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = instrument.get_covariance(scaling_modifier)
    else:
        scales = instrument.get_scaling
        rotations = instrument.get_rotation


    if render_semantics_silhouette:
        # depth_silhouette = get_depth_and_silhouette(means3D, viewpoint_camera.world_view_transform.T)
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        if isinstance(instrument, Instrument):
            semantics = instrument.get_semantics
        else:
            semantics = torch.ones_like(instrument.get_xyz, dtype=torch.float32, device='cuda') 
            
        return means3D
        rendered_image, radii, depth, alpha = rasterizer(
            means3D = means3D,
            means2D = means2D,
            colors_precomp = semantics,
            opacities = opacity,
            scales = scales ,
            rotations = rotations,
            cov3D_precomp = None
            )
        
        rendered_image = rendered_image.permute(1, 2, 0)
        depth = depth.squeeze(0)
        # spotlight light source model
        # rendered_image = viewpoint_camera.spotlight_render(rendered_image, depth.detach())
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth":depth,
                "opacity": alpha.squeeze(0)}
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if True: #pipe.convert_SHs_python:
            shs_view = instrument.get_features.transpose(1, 2).view(-1, 3, (instrument.shaft.gaussian_model.max_sh_degree+1)**2)
            dir_pp = (instrument.get_xyz.cuda() - camera_center.repeat(instrument.get_features.shape[0], 1).cuda())
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(instrument.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = instrument.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if isinstance(instrument, Instrument):
        semantics = instrument.get_semantics
    else:
        semantics = torch.ones_like(instrument.get_xyz, dtype=torch.float32, device='cuda') 
        semantics[:,0] = 0
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        semantics = semantics)
    
    rendered_image = rendered_image.permute(1, 2, 0)
    depth = depth.squeeze(0)
    # spotlight light source model
    # rendered_image = viewpoint_camera.spotlight_render(rendered_image, depth.detach())
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "opacity": alpha.squeeze(0)}


def render_with_instrument_v2(viewpoint_camera, instrument: Instrument, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
                           override_color = None, override_camera_center = None,
                           render_depth_silhouette = False, activate_sh_degree= None, scale_coef = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # gaussian_dict = instrument.get_gaussian_models()
    # gaussian_list = list(gaussian_dict.values())

 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(instrument.get_xyz, dtype=instrument.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    if activate_sh_degree is None:
        active_sh_degree = instrument.active_sh_degree
    else:
        active_sh_degree = activate_sh_degree
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    if override_camera_center is not None:
        viewpoint_camera.camera_center = override_camera_center
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos= torch.tensor([0,0,0],dtype=torch.float32,device='cuda'), #viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    _xyz = instrument.get_xyz * 1000

    means3D = _xyz
    means2D = screenspace_points
    opacity = instrument.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = instrument.get_covariance(scaling_modifier)
    else:
        # scales = instrument.get_scaling
        scales = instrument.get_scaling
        rotations = instrument.get_rotation


    if render_depth_silhouette:
        depth_silhouette = get_depth_and_silhouette(means3D, viewpoint_camera.world_view_transform.T)
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        if isinstance(instrument, Instrument):
            semantics = instrument.get_semantics
        else:
            semantics = torch.ones_like(instrument.get_xyz, dtype=torch.float32, device='cuda') 
            semantics[:,0] = 0
        # rendered_image, radii, depth, alpha, rendered_semantics = rasterizer(
        #     means3D = means3D,
        #     means2D = means2D,
        #     colors_precomp = depth_silhouette,
        #     opacities = opacity,
        #     scales = scales ,
        #     rotations = rotations,
        #     cov3D_precomp = None,
        #     semantics = semantics)
        rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        colors_precomp = depth_silhouette,
        opacities = opacity,
        scales = scales * scale_coef,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

        rendered_image = rendered_image.permute(1, 2, 0)
        depth = depth.squeeze(0)
        # spotlight light source model
        # rendered_image = viewpoint_camera.spotlight_render(rendered_image, depth.detach())
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth":depth,
                "opacity": alpha.squeeze(0),}
                # "semantics": rendered_semantics}
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = instrument.get_features.transpose(1, 2).view(-1, 3, (instrument.shaft.gaussian_model.max_sh_degree+1)**2)
            dir_pp = (instrument.get_xyz.cuda() - viewpoint_camera.camera_center.repeat(instrument.get_features.shape[0], 1).cuda())
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = instrument.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if isinstance(instrument, Instrument):
        semantics = instrument.get_semantics
    else:
        semantics = torch.ones_like(instrument.get_xyz, dtype=torch.float32, device='cuda') 
        semantics[:,0] = 0
    # rendered_image, radii, depth, alpha, rendered_semantics = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales* 10,
    #     rotations = rotations,
    #     cov3D_precomp = None,
    #     semantics = semantics)
    
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales * scale_coef,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = rendered_image.permute(1, 2, 0)
    depth = depth.squeeze(0)
    # spotlight light source model
    # rendered_image = viewpoint_camera.spotlight_render(rendered_image, depth.detach())
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "opacity": alpha.squeeze(0),}
            # "semantics": rendered_semantics}

