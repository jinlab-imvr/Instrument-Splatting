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
#

import os
import numpy as np
import trimesh
import torch

from games.mesh_splatting.utils.graphics_utils import MeshPointCloud
from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    readCamerasFromTransforms,
    readCamerasFromTransforms_rarp,
    getNerfppNorm,
    SceneInfo,
    InstrumentSceneInfo,
    storePly,
)
from utils.sh_utils import SH2RGB
from utils.instrument import Part, Instrument

softmax = torch.nn.Softmax(dim=2)


def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices

def readInstrumentMeshInfo(path, white_background, eval, num_splats, extension=".png", instance_id=0):
    cad_path = './instrument_mesh'

    print("Reading Training Transforms")
    rarp_dataset = False 
    if not os.path.exists( os.path.join(path, "transforms.json")):
        # if os.path.exists(os.path.join(path, "keypoints.json")):
            rarp_dataset = True 
            print("RARP dataset detected")
    if 'SARRARP' in path:
        rarp_dataset = True 
        print("RARP dataset detected")

    if not rarp_dataset:
        print("Reading View Points")
        train_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension)
        test_cam_infos = train_cam_infos # readCamerasFromTransforms(path, "transforms.json", white_background, extension)
    else:
        print("Reading Keypoints")
        instance_idx = instance_id
        assert instance_idx > 0, "For RARP dataset, please provide instance index (>0)"
        train_cam_infos = readCamerasFromTransforms_rarp(path, instance_idx, white_background, extension)
        test_cam_infos = train_cam_infos # readCamerasFromTransforms(path, "keypoints.json", white_background, extension)
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

 
    instrument.initialize_scene(num_splats, path)
    scene_info = InstrumentSceneInfo(instrument = instrument,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_root=path)
    return scene_info



def readNerfSyntheticMeshInfo(
        path, white_background, eval, num_splats, extension=".png"
):
    print("Reading Training Transforms")
    # train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    # print("Reading Mesh object")
    # mesh_scene = trimesh.load(f'{path}/mesh.obj', force='mesh')
    # vertices = mesh_scene.vertices
    # vertices = transform_vertices_function(
    #     torch.tensor(vertices),
    # )
    # faces = mesh_scene.faces
    # triangles = vertices[torch.tensor(mesh_scene.faces).long()].float()

    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    # nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "points3d.ply")
    # # if not os.path.exists(ply_path):
    # if True:
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts_each_triangle = num_splats
    #     num_pts = num_pts_each_triangle * triangles.shape[0]
    #     print(
    #         f"Generating random point cloud ({num_pts})..."
    #     )

    #     # We create random points inside the bounds traingles
    #     alpha = torch.rand(
    #         triangles.shape[0],
    #         num_pts_each_triangle,
    #         3
    #     )

    #     xyz = torch.matmul(
    #         alpha,
    #         triangles
    #     )
    #     xyz = xyz.reshape(num_pts, 3)

    #     shs = np.random.random((num_pts, 3)) / 255.0

    #     pcd = MeshPointCloud(
    #         alpha=alpha,
    #         points=xyz,
    #         colors=SH2RGB(shs),
    #         normals=np.zeros((num_pts, 3)),
    #         vertices=vertices,
    #         faces=faces,
    #         transform_vertices_function=transform_vertices_function,
    #         triangles=triangles.cuda()
    #     )

    #     storePly(ply_path, pcd.points, SH2RGB(shs) * 255)

    # scene_info = SceneInfo(point_cloud=pcd,
    #                        train_cameras=train_cam_infos,
    #                        test_cameras=test_cam_infos,
    #                        nerf_normalization=nerf_normalization,
    #                        ply_path=ply_path)
    # return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    # "Blender_Mesh": readNerfSyntheticMeshInfo,
    "Instrument_Mesh": readInstrumentMeshInfo
}
