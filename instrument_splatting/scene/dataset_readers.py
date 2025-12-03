#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use

import os
import numpy as np
import trimesh
import torch

from utils.sh_utils import SH2RGB
from utils.instrument import Part, Instrument

from typing import NamedTuple
from PIL import Image
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from instrument_splatting.scene.gaussian_model import BasicPointCloud
from utils.instrument import Instrument


softmax = torch.nn.Softmax(dim=2)

semantic_mask_mapping = {
	0: np.array([0,0,0]),
	10: np.array([1,0,0]), # shaft
	20: np.array([0,1,0]), # wrist
	30: np.array([0,0,1]), #gripper
}
class CameraInfo(NamedTuple):
    # add the attribute K to store the intrinsic matrix
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    K: np.array
    image: np.array
    semantic_mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    keypoints_2d: np.array

class SceneInfo(NamedTuple):
    point_cloud: NamedTuple
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

class InstrumentSceneInfo(NamedTuple):
    instrument: Instrument
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_root: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        intrinsic_matrix = np.array(contents["intrinsic_matrix"])

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension) if '.png' not in frame["file_path"] else os.path.join(path, frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            if "transform_matrix" in frame:
                w2c = np.array(frame["transform_matrix"])
            else:
                w2c = np.eye(4)
            
            if "points_2d" in frame:
                points_2d = frame["points_2d"]
                keypoints_2d = np.array([points_2d[key] for key in sorted(points_2d.keys())])
            else:
                keypoints_2d = None
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            # w2c = np.linalg.inv(c2w)
            R = w2c[:3,:3]  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if 'color/' in image_path:
                semantic_mask_path = image_path.replace('/color/', '/l_mask/')
            else:
                semantic_mask_path = image_path.replace('/left_frames_raw/', '/l_mask/')
            
            semantic_mask = Image.open(semantic_mask_path)
            semantic_mask = np.array(semantic_mask)
            if semantic_mask.ndim == 2:
                semantic_mask_rgb = np.zeros((*semantic_mask.shape, 3), dtype=np.float32)
                for key, value in semantic_mask_mapping.items():
                    semantic_mask_rgb[semantic_mask == key] = value
                semantic_mask = semantic_mask_rgb
            else:
                semantic_mask = semantic_mask[:,:,::-1] / 255.0
            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(intrinsic_matrix[1,1], image.size[1])
            FovY = fovy 
            FovX = focal2fov(intrinsic_matrix[0,0], image.size[0])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, K = intrinsic_matrix , image=image,
                            semantic_mask = semantic_mask, keypoints_2d = keypoints_2d,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices

def readInstrumentInfo(path, white_background, extension=".png"):
    cad_path = './instrument_mesh'

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension)
    print("Reading Test Transforms")    
    test_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension)
    print("Reading Mesh object")
    # mesh_scene = trimesh.load(f'{path}/mesh.obj', force='mesh')
    shaft_mesh = trimesh.load_mesh(f'{cad_path}/transformed_shaft.obj', force='mesh')
    wrist_mesh = trimesh.load_mesh(f'{cad_path}/transformed_wrist.obj', force='mesh')
    gripper_mesh_left = trimesh.load_mesh(f'{cad_path}/transformed_gripper_left.obj', force='mesh')
    gripper_mesh_right = trimesh.load_mesh(f'{cad_path}/transformed_gripper_right.obj', force='mesh')

    if isinstance(shaft_mesh, trimesh.Scene):
        shaft_mesh = trimesh.util.concatenate([k for k in shaft_mesh.geometry.values()])
    if isinstance(wrist_mesh, trimesh.Scene):
        wrist_mesh = trimesh.util.concatenate([k for k in wrist_mesh.geometry.values()])
    if isinstance(gripper_mesh_left, trimesh.Scene):
        gripper_mesh_left = trimesh.util.concatenate([k for k in gripper_mesh_left.geometry.values()])
    if isinstance(gripper_mesh_right, trimesh.Scene):
        gripper_mesh_right = trimesh.util.concatenate([k for k in gripper_mesh_right.geometry.values()])
    
    instrument = Instrument(shaft_mesh, wrist_mesh, gripper_mesh_left, gripper_mesh_right)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    """In the stage II (pose tracking) and III (texture learning), we didn't use mesh to initialize the gaussian splatting model.
       Thus, we comment out the following line. In the stage I (geometry pretraining), we used following commented initialize_scene 
       to create pcd: MeshPointCloud based on ray-tracing points on CAD surface (not provided in the repository).
       If you want to use mesh to initialize the gaussian splatting model, please uncomment it and provide num_splats (number per splat).
    """
    # instrument.initialize_scene(num_splats, path)
    scene_info = InstrumentSceneInfo(instrument = instrument,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_root=path)
    return scene_info


# sceneLoadTypeCallbacks = {
#     "gs_instrument": readInstrumentInfo
# }



