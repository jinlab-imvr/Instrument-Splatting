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
import random
import json
from utils.system_utils import searchForMaxIteration
from games.scenes import sceneLoadTypeCallbacks
from utils.instrument import Part, Instrument
import trimesh
from scene.gaussian_model import GaussianModel
from games.mesh_splatting.scene.gaussian_mesh_model_3dgs import GaussianMeshModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.dataset_readers import InstrumentSceneInfo
# from scene.flexible_deform_model import FDM

# class Scene:

#     gaussians : GaussianModel

#     def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, 
#                  shuffle=True, resolution_scales=[1.0], bg_gaussians=None, bg_pcd_ply_path=None):
#         """b
#         :param path: Path to colmap scene main folder.
#         """
#         self.model_path = args.model_path
#         self.loaded_iter = None
#         self.gaussians = gaussians
#         if bg_gaussians is not None:
#             self.bg_gaussians = bg_gaussians
#             self.bg_gaussians.load_ply(bg_pcd_ply_path)
#             self.bg_gaussians.load_model(os.path.join(self.model_path, "bg_gaussians.json"))
#         if load_iteration:
#             if load_iteration == -1:
#                 self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
#             else:
#                 self.loaded_iter = load_iteration
#             print("Loading trained model at iteration {}".format(self.loaded_iter))

#         self.train_cameras = {}
#         self.test_cameras = {}

#         if os.path.exists(os.path.join(args.source_path, "sparse")):
#             if args.gs_type == "gs_multi_mesh":
#                 scene_info = sceneLoadTypeCallbacks["Colmap_Mesh"](
#                     args.source_path, args.images, args.eval, args.num_splats, args.meshes
#                 )
#             else:
#                 scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
#         elif os.path.exists(os.path.join(args.source_path, "transforms.json")) or os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
#             if args.gs_type == "gs_mesh":
#                 print("Found transforms_train.json file, assuming Instrument_Mesh data set!")
#                 scene_info = sceneLoadTypeCallbacks["Instrument_Mesh"](
#                     args.source_path, args.white_background, args.eval, args.num_splats[0], instance
#                 )
#             elif args.gs_type == "gs_flame":
#                 print("Found transforms_train.json file, assuming Flame Blender data set!")
#                 scene_info = sceneLoadTypeCallbacks["Blender_FLAME"](args.source_path, args.white_background, args.eval)
#             else:
#                 print("Found transforms_train.json file, assuming Blender data set!")
#                 scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
#         else:
#             assert False, "Could not recognize scene type!"

#         if not self.loaded_iter:
#             if args.gs_type == "gs_multi_mesh":
#                 for i, ply_path in enumerate(scene_info.ply_path):
#                     with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, f"input_{i}.ply") , 'wb') as dest_file:
#                         dest_file.write(src_file.read())
#             else:
#                 with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
#                     dest_file.write(src_file.read())
#             # json_cams = []
#             # camlist = []
#             # if scene_info.test_cameras:
#             #     camlist.extend(scene_info.test_cameras)
#             # if scene_info.train_cameras:
#             #     camlist.extend(scene_info.train_cameras)
#             # for id, cam in enumerate(camlist):
#             #     json_cams.append(camera_to_JSON(id, cam))
#             # with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
#             #     json.dump(json_cams, file)

#         if shuffle:
#             random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
#             random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

#         self.cameras_extent = scene_info.nerf_normalization["radius"]

#         for resolution_scale in resolution_scales:
#             print("Loading Training Cameras")
#             self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
#             print("Loading Test Cameras")
#             self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

#         if self.loaded_iter:
#             self.gaussians.load_ply(os.path.join(self.model_path,
#                                                            "point_cloud",
#                                                            "iteration_" + str(self.loaded_iter),
#                                                            "point_cloud.ply"))
#             self.gaussians.point_cloud = scene_info.point_cloud
#             if args.gs_type == "gs_mesh":
#                 self.gaussians.triangles = scene_info.point_cloud.triangles
#         else:
#             self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

#     def save(self, iteration):
#         point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
#         self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

#     def getTrainCameras(self, scale=1.0):
#         return self.train_cameras[scale]

#     def getTestCameras(self, scale=1.0):
#         return self.test_cameras[scale]
    
# class MultiGaussianModel:
#     def __init__(self, gaussians : list, gaussian_names : list):
#         self.gaussian_dict = dict(zip(gaussian_names, gaussians))

#     def training_setup(self, opt):
#         for gaussian in self.gaussian_dict.values():
#             gaussian: GaussianMeshModel
#             gaussian.training_setup(opt)
    
#     def restore(self, model_params, opt):
#         for gaussian in self.gaussian_dict.values():
#             gaussian: GaussianMeshModel
#             gaussian.restore(model_params, opt)
    
        
class InstrumentScene:

    gaussians : GaussianMeshModel
    # def __init__(self, args : ModelParams, gaussians_init_func, sh_degree, load_iteration=None, 
    #              shuffle=False, resolution_scales=[1.0], pretrain_params = None, task_name = None, instance_id=1,
    #               bg_path=None, task_path = None):

    # def __init__(self, args : ModelParams, gaussians_init_func, sh_degree,  
    #              shuffle=False, resolution_scales=[1.0], pretrain_path = None, instance_id=1,pretrain_params = None, load_iteration = None):
    #     """b
    #     :param path: Path to colmap scene main folder.
    #     """
    #     self.model_path = args.model_path
    #     self.source_path = args.source_path
    #     self.pretrain_path = 'pretrained_models' if pretrain_path is None else pretrain_path
    #     self.loaded_iter = None
    #     # if bg_path is not None:
    #     #     self.bg_gaussians = FDM()
    #     #     self.bg_gaussians.load_ply(bg_path+'/point_cloud.ply')
    #     #     self.bg_gaussians.load_model(os.path.join(self.model_path, bg_path))


    #     self.train_cameras = {}
    #     self.test_cameras = {}

    #     if os.path.exists(os.path.join(args.source_path, "sparse")):
    #         if args.gs_type == "gs_multi_mesh":
    #             scene_info = sceneLoadTypeCallbacks["Colmap_Mesh"](
    #                 args.source_path, args.images, args.eval, args.num_splats, args.meshes
    #             )
    #         else:
    #             scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
    #     elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
    #         if args.gs_type == "gs_mesh" or args.gs_type == "instrument_mesh":
    #             print("Found transforms_train.json file, assuming Instrument_Mesh data set!")
              
    #             scene_info = sceneLoadTypeCallbacks["Instrument_Mesh"](
    #                 args.source_path, args.white_background, args.eval, args.num_splats[0], instance_id = instance_id
    #             )
    #         else:
    #             KeyError("Could not recognize scene type! No transforms_train.json file found.")
    #     else:
    #         print("Could not recognize scene type! No transforms_train.json file found.")
    #         if args.gs_type == "gs_mesh" or args.gs_type == "instrument_mesh":
    #             print("Found transforms_train.json file, assuming Instrument_Mesh data set!")
              
    #             scene_info = sceneLoadTypeCallbacks["Instrument_Mesh"](
    #                 args.source_path, args.white_background, args.eval, args.num_splats[0], instance_id = instance_id
    #             )

   
    #     json_cams = []
    #     camlist = []
    #     if scene_info.test_cameras:
    #         camlist.extend(scene_info.test_cameras)
    #     if scene_info.train_cameras:
    #         camlist.extend(scene_info.train_cameras)
    #     for id, cam in enumerate(camlist):
    #         json_cams.append(camera_to_JSON(id, cam))
    #     with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
    #         json.dump(json_cams, file)

    #     if shuffle:
    #         random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
    #         random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

    #     self.cameras_extent = scene_info.nerf_normalization["radius"]
    #     self.instrument = scene_info.instrument
    #     for resolution_scale in resolution_scales:
    #         print("Loading Training Cameras")
    #         self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
    #         print("Loading Test Cameras")
    #         self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
    #     for part_name in self.instrument.part_dict.keys():
    #         use_triangle_scale = False if part_name in ["shaft", ] else True
    #         scale_coef = 1.0 if part_name in ["shaft", ] else 1000.0
    #         part_gaussians = gaussians_init_func(sh_degree, use_triangle_scale, scale_coef)
    #         part_gaussians: GaussianMeshModel

    #         # load the pretriained GS parameters
    #         if pretrain_params is None:
    #             param_file_path = None
    #             raise ValueError(f"Please provide pretrained parameters for part {part_name}!")
    #             # param_files_path = sorted([f for f in os.listdir(self.source_path) if f'{part_name}_pretrain_param_3dgs' in f])
    #         else:
    #             param_files_paths = sorted([f for f in os.listdir(self.pretrain_path) if f'{part_name}_{pretrain_params}' in f])
                
    #             if len(param_files_paths)  == 0:
    #                 # param_files_path = sorted([f for f in os.listdir(self.source_path) if f'{part_name}_pretrain_param_3dgs' in f])
    #                 raise ValueError(f"Pretrained parameters {pretrain_params} not found for part {part_name}")
    #             if load_iteration is not None:
    #                 param_files_paths = [f for f in param_files_paths if f'{load_iteration}' in f]
    #                 if len(param_files_paths) == 0:
    #                     raise ValueError(f"Pretrained parameters for iteration {load_iteration} not found for part {part_name}")
                
    #             param_file = param_files_paths[-1] # load the latest one
    #             param_file_path = os.path.join(self.pretrain_path, param_file)
                    
    #         part_gaussians.create_from_pcd(None, self.cameras_extent, gs_param_file=param_file_path)
    #         print(f"Loaded pretrained GS parameters for part {part_name} from {param_file_path}.")
    #         self.instrument.gaussian_model_setup(part_name, part_gaussians)

    def __init__(self, args : ModelParams, gaussians_init_func, sh_degree, load_iteration=None, 
                 shuffle=False, resolution_scales=[1.0], pretrain_params = None, pretrain_path = None, task_name = None, instance_id=1,
                  bg_path=None, task_path = None):
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
        # if load_iteration:
        #     if load_iteration == -1:
        #         self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        #     else:
        #         self.loaded_iter = load_iteration
        #     print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.gs_type == "gs_multi_mesh":
                scene_info = sceneLoadTypeCallbacks["Colmap_Mesh"](
                    args.source_path, args.images, args.eval, args.num_splats, args.meshes
                )
            else:
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            if args.gs_type == "gs_mesh" or args.gs_type == "instrument_mesh":
                print("Found transforms_train.json file, assuming Instrument_Mesh data set!")
              
                scene_info = sceneLoadTypeCallbacks["Instrument_Mesh"](
                    args.source_path, args.white_background, args.eval, args.num_splats[0], instance_id = instance_id
                )
            else:
                KeyError("Could not recognize scene type! No transforms_train.json file found.")
        else:
            print("Could not recognize scene type! No transforms_train.json file found.")
            if args.gs_type == "gs_mesh" or args.gs_type == "instrument_mesh":
                print("Found transforms_train.json file, assuming Instrument_Mesh data set!")
              
                scene_info = sceneLoadTypeCallbacks["Instrument_Mesh"](
                    args.source_path, args.white_background, args.eval, args.num_splats[0], instance_id = instance_id
                )

        if not self.loaded_iter:
            if args.gs_type == "gs_multi_mesh":
                for i, ply_path in enumerate(scene_info.ply_path):
                    with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, f"input_{i}.ply") , 'wb') as dest_file:
                        dest_file.write(src_file.read())
            else:
                if hasattr(scene_info, "ply_root"):
                    scene_info: InstrumentSceneInfo
                    path = scene_info.ply_root
                    for part_name in scene_info.instrument.part_dict.keys():
                        ply_path = os.path.join(path, f"{part_name}_points3d.ply")
                        with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, f"{part_name}_input.ply") , 'wb') as dest_file:
                            dest_file.write(src_file.read())
                # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                #     dest_file.write(src_file.read())
            # json_cams = []
            # camlist = []
            # if scene_info.test_cameras:
            #     camlist.extend(scene_info.test_cameras)
            # if scene_info.train_cameras:
            #     camlist.extend(scene_info.train_cameras)
            # for id, cam in enumerate(camlist):
            #     json_cams.append(camera_to_JSON(id, cam))
            # with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            #     json.dump(json_cams, file)

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
            part_gaussians: GaussianMeshModel
            if self.loaded_iter:
                part_gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            f"{part_name}_point_cloud.ply"))
                part_gaussians.point_cloud = self.instrument.part_dict[part_name].pcd
                if args.gs_type == "gs_mesh":
                    part_gaussians.triangles = self.instrument.part_dict[part_name].pcd.triangles
            else:
                # load the pretriained GS parameters
                if pretrain_params is None:
                    param_file_path = None
                    # param_files_path = sorted([f for f in os.listdir(self.source_path) if f'{part_name}_pretrain_param_3dgs' in f])
                else:
                    if task_name is None:
                        param_files_path = sorted([f for f in os.listdir(self.pretrain_path) if f'{part_name}_{pretrain_params}' in f])
                    else:
                        if task_path is not None:
                            self.pretrain_path = task_path + '/' + task_name + '/'
                        else:
                            self.pretrain_path = 'Results/'+ task_name + '/'
                        param_files_path = sorted([f for f in os.listdir(self.pretrain_path) if f'{part_name}_trained_3dgs_param' in f])
                    if len(param_files_path)  == 0:
                        # param_files_path = sorted([f for f in os.listdir(self.source_path) if f'{part_name}_pretrain_param_3dgs' in f])
                        if len(param_files_path)  == 0:
                            raise ValueError(f"Pretrained parameters {pretrain_params} not found for part {part_name}")
                    if load_iteration is not None:
                        # find the param file with the closest iteration number
                        param_file = [f for f in param_files_path if f'{load_iteration}.pth' in f][0]
                    else:
                        param_file = param_files_path[-1]
                    param_file_path = os.path.join(self.pretrain_path, param_file)
                # import pdb; pdb.set_trace()
                part_gaussians.create_from_pcd(self.instrument.part_dict[part_name].pcd, self.cameras_extent, gs_param_file=param_file_path)
                # use_triangle_scales = False if part_name in ["shaft"] else True
                # part_gaussians.prepare_scaling_rot(use_triangle_scales)
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
    

class InstrumentGaussianWrapper:

    gaussians : GaussianMeshModel

    def __init__(self, gaussians_init_func, sh_degree, pretrain_params, pretrain_path = None, results_path = None, load_iteration = None):
        """b
        :param path: Path to colmap scene main folder.
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pretrain_path = pretrain_path if pretrain_path is not None else os.path.join(repo_root, 'pretrained_models')
        self.loaded_iter = None
        self.cad_path = os.path.join(repo_root, 'instrument_mesh')

        self.train_cameras = {}
        self.test_cameras = {}
        shaft_mesh = trimesh.load_mesh(f'{self.cad_path}/transformed_shaft.obj', force='mesh')
        wrist_mesh = trimesh.load_mesh(f'{self.cad_path}/transformed_wrist.obj', force='mesh')
        gripper_mesh_left = trimesh.load_mesh(f'{self.cad_path}/transformed_gripper_left.obj', force='mesh')
        gripper_mesh_right = trimesh.load_mesh(f'{self.cad_path}/transformed_gripper_right.obj', force='mesh')

        if isinstance(shaft_mesh, trimesh.Scene):
            shaft_mesh = trimesh.util.concatenate([k for k in shaft_mesh.geometry.values()])
        if isinstance(wrist_mesh, trimesh.Scene):
            wrist_mesh = trimesh.util.concatenate([k for k in wrist_mesh.geometry.values()])
        if isinstance(gripper_mesh_left, trimesh.Scene):
            gripper_mesh_left = trimesh.util.concatenate([k for k in gripper_mesh_left.geometry.values()])
        if isinstance(gripper_mesh_right, trimesh.Scene):
            gripper_mesh_right = trimesh.util.concatenate([k for k in gripper_mesh_right.geometry.values()])
    
        self.instrument = Instrument(shaft_mesh, wrist_mesh, gripper_mesh_left, gripper_mesh_right)  
        # self.instrument.initialize_scene(num_splats, path)
        for part_name in self.instrument.part_dict.keys():
            use_triangle_scale = False if part_name in ["shaft", ] else True
            scale_coef = 1.0 if part_name in ["shaft", ] else 1000.0
            part_gaussians = gaussians_init_func(sh_degree, use_triangle_scale, scale_coef)
            part_gaussians: GaussianMeshModel
            if results_path is None:
                param_files_path = sorted([f for f in os.listdir(self.pretrain_path) if f'{part_name}_{pretrain_params}' in f])
            else:
                self.pretrain_path = results_path
                param_files_path = sorted([f for f in os.listdir(self.pretrain_path) if f'{part_name}_trained_3dgs_param' in f])
            if load_iteration is not None:
                # find the param file with the closest iteration number
                param_file = [f for f in param_files_path if f'{load_iteration}.pth' in f][0]
            else:
                param_file = param_files_path[-1]
            
            param_file_path = os.path.join(self.pretrain_path, param_file)
            # self.instrument.part_dict[part_name].pcd       
            cameras_extent = 5.817785098116701e-09     
            part_gaussians.create_from_pcd(None, cameras_extent, gs_param_file=param_file_path)
            # use_triangle_scales = False if part_name in ["shaft"] else True
            # part_gaussians.prepare_scaling_rot(use_triangle_scales)
            self.instrument.gaussian_model_setup(part_name, part_gaussians)
        

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        for part_name in self.instrument.part_dict.keys():
            part_gaussians = self.instrument.part_dict[part_name].gaussian_model    
            part_gaussians.save_ply(os.path.join(point_cloud_path, f"{part_name}_point_cloud.ply"))
