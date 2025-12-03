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
import random
import json
from utils.system_utils import searchForMaxIteration
from instrument_splatting.scene.gaussian_instrument_model import GaussianInstrumentModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from instrument_splatting.scene.dataset_readers import InstrumentSceneInfo
# from instrument_splatting.scene.flexible_deform_model import FDM

from instrument_splatting.scene.dataset_readers import (
    readInstrumentInfo
)


# for customized scene loading functions
sceneLoadTypeCallbacks = {
    "gs_instrument": readInstrumentInfo
}

class InstrumentScene:

    gaussians : GaussianInstrumentModel

    def __init__(self, args : ModelParams, gaussians_init_func, sh_degree,  
                 shuffle=False, resolution_scales=[1.0], pretrain_path = None, pretrain_params = None, load_iteration = None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.source_path = args.source_path
        self.pretrain_path = 'pretrained_models' if pretrain_path is None else pretrain_path
        self.loaded_iter = None
        # if bg_path is not None:
        #     self.bg_gaussians = FDM()
        #     self.bg_gaussians.load_ply(bg_path+'/point_cloud.ply')
        #     self.bg_gaussians.load_model(os.path.join(self.model_path, bg_path))


        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "transforms.json")):
            if args.gs_type == "gs_instrument":
                print("Found transforms_train.json file, assuming Instrument_Mesh data set!")
                scene_info = sceneLoadTypeCallbacks["gs_instrument"](
                    args.source_path, args.white_background)
            else:
                KeyError("Could not recognize scene type! No transforms_train.json file found.")
        else:
            KeyError("Could not recognize scene type! No transforms_train.json file found.")

   
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.instrument = scene_info.instrument
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        for part_name in self.instrument.part_dict.keys():
            use_triangle_scale = False if part_name in ["shaft", ] else True
            scale_coef = 1.0 if part_name in ["shaft", ] else 1000.0
            part_gaussians = gaussians_init_func(sh_degree, use_triangle_scale, scale_coef)
            part_gaussians: GaussianInstrumentModel

            # load the pretriained GS parameters
            if pretrain_params is None:
                param_file_path = None
                raise ValueError(f"Please provide pretrained parameters for part {part_name}!")
                # param_files_path = sorted([f for f in os.listdir(self.source_path) if f'{part_name}_pretrain_param_3dgs' in f])
            else:
                param_files_paths = sorted([f for f in os.listdir(self.pretrain_path) if f'{part_name}_{pretrain_params}' in f])
                
                if len(param_files_paths)  == 0:
                    # param_files_path = sorted([f for f in os.listdir(self.source_path) if f'{part_name}_pretrain_param_3dgs' in f])
                    raise ValueError(f"Pretrained parameters {pretrain_params} not found for part {part_name}")
                if load_iteration is not None:
                    param_files_paths = [f for f in param_files_paths if f'{load_iteration}' in f]
                    if len(param_files_paths) == 0:
                        raise ValueError(f"Pretrained parameters for iteration {load_iteration} not found for part {part_name}")
                
                param_file = param_files_paths[-1] # load the latest one
                param_file_path = os.path.join(self.pretrain_path, param_file)
                    
            part_gaussians.create_from_pcd(None, self.cameras_extent, gs_param_file=param_file_path)
            print(f"Loaded pretrained GS parameters for part {part_name} from {param_file_path}.")
            self.instrument.gaussian_model_setup(part_name, part_gaussians)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        for part_name in self.instrument.part_dict.keys():
            part_gaussians = self.instrument.part_dict[part_name].gaussian_model    
            part_gaussians.save_ply(os.path.join(point_cloud_path, f"{part_name}_point_cloud.ply"))


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]