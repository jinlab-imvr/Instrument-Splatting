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
import sys
sys.path.append('../')
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.instrument import Instrument
from utils.part import Part

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
    overall_mask: np.array
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
    if isinstance(cam_info[0], list):
        cam_info = cam_info[0] + cam_info[1]
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

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

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def load_keypoints_grouped(keypoints_json_path):
    """
    Load keypoints.json that contains a top-level "keypoints" list where each entry has
    'frame_index', 'frame_filename', 'x', 'y'. Returns a dict:
        frame_filename (str) -> np.array([[x1,y1],[x2,y2],...], dtype=np.float32)
    The order of points for a given frame_filename is preserved as they appear in the list.
    """
    if not os.path.exists(keypoints_json_path):
        return {}

    with open(keypoints_json_path, "r") as f:
        data = json.load(f)

    kp_list = data.get("keypoints", [])
    grouped = {}
    for kp in kp_list:
        fname = kp.get("frame_filename")
        if fname is None:
            # fall back to zero-padded index if filename missing
            frame_index = kp.get("frame_index", 0)
            fname = f"{frame_index:05d}.png"
        grouped.setdefault(fname, []).append((kp.get("x", 0), kp.get("y", 0)))

    # convert lists to numpy arrays
    for k, v in list(grouped.items()):
        grouped[k] = np.array(v, dtype=np.float32)

    return grouped

def readCamerasFromTransforms_rarp(path, instance_idx, white_background, extension=".png"):
    rarp_fc = 587.54401824
    cam_infos = []
    frame_names = []
    instance_name = os.path.join(path, f"instance{instance_idx}")
    instance_idx = int(instance_idx)
    images_path = os.path.join(path, "frames")
    assert instance_idx > 0, "instance_id should be greater than 0"
    
    mask_overall_path = os.path.join(path, instance_name, "refined_masks_overall") if os.path.exists(os.path.join(path, instance_name, "refined_masks_overall"))\
        else os.path.join(path, instance_name, "masks_overall")
    mask_shaft_path = os.path.join(path, instance_name,  "refined_masks_shaft") if os.path.exists(os.path.join(path, instance_name, "refined_masks_shaft"))\
        else os.path.join(path, instance_name, "masks_shaft")
    mask_wrist_path = os.path.join(path, instance_name,  "refined_masks_wrist") if os.path.exists(os.path.join(path, instance_name, "refined_masks_wrist"))\
        else os.path.join(path, instance_name, "masks_wrist")
    mask_gripper_path = os.path.join(path, instance_name, "refined_masks_gripper") if os.path.exists(os.path.join(path, instance_name, "refined_masks_gripper"))\
        else os.path.join(path, instance_name, "masks_gripper")
    image = Image.open(os.path.join(images_path, sorted(os.listdir(images_path))[0]))
    width = image.size[0]
    height = image.size[1] 
    if os.path.exists(os.path.join(path, "LeftCalibration.ini")):
        # read ini file to get intrinsic matrix
        import configparser
        config = configparser.ConfigParser()
        config.read(os.path.join(path, "LeftCalibration.ini"))
        fx = float(config['StereoLeft']['fc_x'])
        fy = float(config['StereoLeft']['fc_y'])
        cx = float(config['StereoLeft']['cc_x'])
        cy = float(config['StereoLeft']['cc_y'])
        intrinsic_matrix = np.array([[fx, 0, cx],
                                   [0, fy, cy], 
                                   [0, 0, 1]])
        # # debug stop
        # import pdb
        # pdb.set_trace()
    else:
        intrinsic_matrix = np.array([[rarp_fc, 0, width/2.],
                                   [0, rarp_fc, height/2.], 
                                   [0, 0, 1]])
    for idx, frame_name in enumerate(sorted(os.listdir(images_path))):
        cam_name = os.path.join(images_path, frame_name)
        frame_idx = str(int(frame_name.split(".")[0]) - 1).zfill(5)  # frame index starts from 0
        w2c = np.eye(4)
        instance_name_other = os.path.join(path, f"instance{(instance_idx % 2) + 1}")
        if os.path.exists(os.path.join(path, instance_name, "keypoints.json")):
            keypoints_map = load_keypoints_grouped(os.path.join(path, instance_name, "keypoints.json"))
            key_frame_names = keypoints_map.keys()
            
            valid_key_frames = []
            for key_frame_name in list(key_frame_names):
                points_2d = keypoints_map[key_frame_name]
                keypoints_2d = np.array(points_2d)
                if len(keypoints_2d) != 8:
                    
                    if len(keypoints_2d) >= 6:
                        valid_key_frames.append(key_frame_name)
                    if len(keypoints_2d) > 8 :
                        keypoints_2d = keypoints_2d[:8,:]
                        # valid_key_frames.append(key_frame_name)
                    if len(keypoints_2d) < 6:
                        keypoints_2d = None
                else:
                    valid_key_frames.append(key_frame_name)
            if len(valid_key_frames) == 0:
            #     key_frame_name = valid_key_frames[0]
            #     points_2d = keypoints_map[key_frame_name]
            #     keypoints_2d = np.array(points_2d)
            # else:
                raise RuntimeError(f"No valid key frames with at least 6 keypoints found in {os.path.join(path, instance_name, 'keypoints.json')}")
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

        semantic_mask =  np.zeros((image.size[1], image.size[0]), dtype=np.uint8)

        for part_id, part_mask_path in zip([10,20,30], [ mask_shaft_path, mask_wrist_path, mask_gripper_path]):
            part_mask_file = os.path.join(part_mask_path, frame_idx + '.' + frame_name.split(".")[1] )
            if os.path.exists(part_mask_file):
                part_mask = Image.open(part_mask_file)
                part_mask = np.array(part_mask)
                semantic_mask[part_mask > 0] = part_id
        try:
            overall_mask = Image.open(os.path.join(mask_overall_path, frame_idx + '.' + frame_name.split(".")[1] ))
            overall_mask = np.array(overall_mask)
        except:
            overall_mask = None
        if np.sum(semantic_mask) == 0:
            continue
        frame_names.append(frame_name)
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
                        semantic_mask = semantic_mask, overall_mask = overall_mask,  keypoints_2d = keypoints_2d,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    # based on key_frame_name, reorder cam_infos bidirectionally into two lists
    cam_infos_forward= []
    cam_infos_backward = []
    # i have calculated the key_frame_name above
    # print(os.path.join(path, instance_name, "keypoints.json"))
    if key_frame_name in frame_names:
        key_frame_index = frame_names.index(key_frame_name)
        # add frames from key_frame_index to the end
        cam_infos_forward.extend(cam_infos[key_frame_index:])
        # add frames from the beginning to key_frame_index-1 in reverse order
        cam_infos_backward.extend(reversed(cam_infos[:key_frame_index]))
        cam_infos = [cam_infos_forward, cam_infos_backward]
    return cam_infos

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
            if os.path.exists(os.path.join(path, "keypoints.json")):
                keypoints_map = load_keypoints_grouped(os.path.join(path, "keypoints.json"))
                key_frame_names = keypoints_map.keys()
                valid_key_frames = []
                for key_frame_name in list(key_frame_names):
                    points_2d = keypoints_map[key_frame_name]
                    keypoints_2d = np.array(points_2d)
                    if len(keypoints_2d) != 8:
                        
                        if len(keypoints_2d) >= 6:
                            valid_key_frames.append(key_frame_name)
                        if len(keypoints_2d) > 8 :
                            keypoints_2d = keypoints_2d[:8,:]
                            # valid_key_frames.append(key_frame_name)
                        if len(keypoints_2d) < 6:
                            keypoints_2d = None
                    else:
                        valid_key_frames.append(key_frame_name)
                if len(valid_key_frames) > 0:
                    key_frame_name = valid_key_frames[0]
                    points_2d = keypoints_map[key_frame_name]
                    keypoints_2d = np.array(points_2d)
           
            elif "points_2d" in frame:
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
            overall_mask = semantic_mask.sum(axis=2) > 0
           
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, K = intrinsic_matrix , image=image,
                            semantic_mask = semantic_mask, overall_mask = overall_mask, keypoints_2d = keypoints_2d,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}