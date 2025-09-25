import trimesh
import  numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
# from instrument_splatting.utils.graphics_utils import MeshPointCloud
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
from utils.part import Part
import pyrender
import cv2
# from instrument_splatting.scene.dataset_readers import InstrumentSceneInfo
# from instrument_splatting.scene.gaussian_mesh_model_3dgs import GaussianMeshModel


def random_point(mesh, num_points=1000):
    vertices = mesh.vertices
    selected_indices = np.random.choice(vertices.shape[0], num_points, replace=False)
    selected_points = vertices[selected_indices]
    return selected_points

class gaussian_attr(object):
    def __init__(self, alpha: torch.Tensor, vertices: torch.Tensor, rot: torch.Tensor, scale: torch.Tensor, opacity: torch.Tensor, shs_coef: torch.Tensor):
        self.alpha = alpha
        self.vertices = vertices
        self.rot = rot
        self.scale = scale
        self.opacity = opacity
        self.shs_coef = shs_coef
    
    def quaternion_to_matrix(self, q):
        """
        Convert a quaternion to a rotation matrix.
        The quaternion should be in the form [w, x, y, z] where w is the real part.
        """
        w, x, y, z = q
        return torch.tensor([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ], dtype=torch.float32)

    def matrix_to_quaternion(self, m):
        """
        Convert a rotation matrix to a quaternion.
        The quaternion will be in the form [w, x, y, z] where w is the real part.
        """
        
        trace = m[0, 0] + m[1, 1] + m[2, 2]

        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = 2.0 * torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

        return torch.tensor([w, x, y, z], dtype=torch.float32) 

    def apply_transform(self, transformation: torch.Tensor):
        xyz = torch.mm(transformation[:3,:3] @ self.xyz.t()) + transformation[:3,3].unsqueeze(1)
        self.xyz = xyz.t()
        # Convert quaternion to rotation matrix
        rot_matrix = self.quaternion_to_matrix(self.rot)
        # Apply the transformation to the rotation matrix
        transformed_rot_matrix = torch.mm(transformation[:3, :3], rot_matrix)
        # Convert the transformed rotation matrix back to quaternion
        self.rot = self.matrix_to_quaternion(transformed_rot_matrix)
        # Apply the scale
        self.scale = torch.mm(transformation[:3, :3], self.scale.t()).t()
        

class Instrument(object):
    def __init__(self, shaft_mesh, wrist_mesh, gripper_mesh_left, gripper_mesh_right):
        self._preComputeTransform()

        self.shaft = Part('shaft', shaft_mesh, self.bias2shaft)
        self.wrist = Part('wrist',wrist_mesh, self.bias2wrist)
        self.l_gripper = Part('l_gripper', gripper_mesh_left, self.bias2l_gripper)
        self.r_gripper = Part('r_gripper', gripper_mesh_right, self.bias2r_gripper)
        self.new_mesh_dict = {}

        self.part_dict = {'shaft': self.shaft, 'wrist': self.wrist, 'l_gripper': self.l_gripper, 'r_gripper': self.r_gripper}

    def gaussian_model_setup(self, part_name, gaussian_model):
        self.part_dict[part_name].gaussian_setup(gaussian_model)

    def _rodrigues_rotation_matrix(self, axis, theta):
        """
        Calculate the rotation matrix using Rodrigues' rotation formula.
        
        Parameters:
        axis (torch tensor): The axis of rotation (must be a unit vector).
        theta (float): The angle of rotation in radians.
        
        Returns:
        torch tensor: The rotation matrix.
        """
        axis = axis / torch.norm(axis.float())
        a = torch.cos(theta / 2.0)
        b, c, d = -axis * torch.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        row1 = torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)])
        row2 = torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)])
        row3 = torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc])
        
        return torch.stack([row1, row2, row3])[...,0]

    def _preComputeTransform(self):
        # convert TriMesh coordinate system to Blender coordinate system
        bias2world = torch.eye(4)
        bias2world[:3, :3] = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
        self.bias2world = bias2world
        flip_wrist = torch.eye(4)
        flip_wrist[:3, :3] = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).T
        shaft2world = torch.eye(4)
        wrist2world = torch.eye(4)
        l_gripper2world = torch.eye(4)
        r_gripper2world = torch.eye(4)

        shaft2world[:3, 3] = torch.tensor([-0.2159, 0, 0])  # unit: m
        l_gripper2world[:3, 3] = torch.tensor([0.009, 0, 0])  # unit: m
        r_gripper2world[:3, 3] = torch.tensor([0.009, 0, 0])  # unit: m

        self.shaft2world = shaft2world
        self.wrist2world = wrist2world
        self.l_gripper2world = l_gripper2world
        self.r_gripper2world = r_gripper2world
        
        self.bias2shaft = torch.inverse(shaft2world) @ bias2world
        self.bias2l_gripper = torch.inverse(l_gripper2world) @ bias2world
        self.bias2r_gripper = torch.inverse(r_gripper2world) @ bias2world
        self.bias2wrist = torch.inverse(wrist2world) @ flip_wrist @ bias2world


    def _apply_transformation(self, part_name, transformation, gs_grad, pose_grad):
        # changing the mesh: now only for debug
        if self.with_mesh:
            self.new_mesh_dict[part_name] = self.part_dict[part_name].apply_transformation_mesh(transformation)
        ## changing the params of gs model
        # self.part_dict[part_name].apply_transformation_gaussian_(transformation) 
        
        if pose_grad == False:
            transformation = transformation.detach()
        render_params = self.part_dict[part_name].apply_transformation_params(transformation, gs_grad)
        return render_params
    
    def initialize_scene(self, num_pts_per_triangle, ply_root):
        """
        Initialize the point cloud for mesh scene.
        
        Parameters:
        num_pts_per_triangle (int): The number of points to sample per triangle.
        """
        for part_name, part in self.part_dict.items():
            surf_point_file =  os.path.join(ply_root, f'{part_name}_ray_tracing.npz')
            # part.create_meshpoint_cloud(num_pts_per_triangle,  os.path.join(ply_root, f'{part_name}_points3d.ply'))
            # continue
            if os.path.exists(surf_point_file):
                npz = np.load(surf_point_file)
                surf_face_idxes = torch.from_numpy(npz['surf_face_idxes'])
                surf_alphas = torch.from_numpy(npz['surf_alpha'])
                surf_vertices = torch.from_numpy(npz['surf_vertices'])
                
                part.create_meshpoint_cloud_with_ray_tracing(surf_face_idxes, surf_alphas, surf_vertices,  os.path.join(ply_root, f'{part_name}_points3d.ply'))
            else:
                part.create_meshpoint_cloud(num_pts_per_triangle,  os.path.join(ply_root, f'{part_name}_points3d.ply'))

    def training_setup(self, opt):
        for part_name, part in self.part_dict.items():
            part.gaussian_model.training_setup(opt)
    
    def restore(self, model_params, opt):
        for part_name, part in self.part_dict.items():
            part.gaussian_model.restore(model_params, opt)
    def step_optimizer(self):
        for i,(part_name, part) in enumerate(self.part_dict.items()):
             # if i > 0: continue
            part.gaussian_model.optimizer.step()
    def zero_grad(self, set_to_none = False):
        for i,(part_name, part) in enumerate(self.part_dict.items()):
             # if i > 0: continue
            part.gaussian_model.optimizer.zero_grad(set_to_none)
    @property
    def get_xyz(self):
        points = []
        for i,(part_name, part) in enumerate(self.part_dict.items()):
            # if i > 0: continue
            points.append(part.gaussian_model.get_xyz)
        return torch.cat(points, dim=0) 
    
    def get_keypoints(self):
        keypoints = []
        keypoints.append(self.part_dict['shaft'].keypoints)#render_params['keypoints'])
        keypoints.append(self.part_dict['wrist'].keypoints)#render_params['keypoints'])
        keypoints.append(self.part_dict['l_gripper'].keypoints)#render_params['keypoints'])
        keypoints.append(self.part_dict['r_gripper'].keypoints)#render_params['keypoints'])
        return torch.cat(keypoints, dim=0)
    @property
    def get_opacity(self):
        opacities = []
        for i,(part_name, part) in enumerate(self.part_dict.items()):
             # if i > 0: continue
            opacities.append(part.gaussian_model.get_opacity)
        return torch.cat(opacities, dim=0)
    
    def get_covariance(self, scaling_modifier):
        covariances = []
        for part_name, part in self.part_dict.items():
            covariances.append(part.gaussian_model.get_covariance(scaling_modifier))
        return torch.cat(covariances, dim=0)
    @property
    def get_scaling(self):
        scales = []
        for i,(part_name, part) in enumerate(self.part_dict.items()):
             # if i > 0: continue
            if part_name in ['wrist', 'l_gripper', 'r_gripper']:
                # scales.append(1000 * part.gaussian_model.get_scaling)
                scales.append(part.gaussian_model.get_scaling)
            else:
                scales.append(part.gaussian_model.get_scaling)
        return torch.cat(scales, dim=0) 
    @property
    def get_scaling_iso(self):
        scales = []
        for i,(part_name, part) in enumerate(self.part_dict.items()):
             # if i > 0: continue
            if part_name in ['wrist', 'l_gripper', 'r_gripper']:
                scales.append(1000 * part.gaussian_model.get_scaling_iso )
            else:
                scales.append(part.gaussian_model.get_scaling_iso)
        return torch.cat(scales, dim=0) 
    @property
    def get_rotation(self):
        rotations = []
        for i,(part_name, part) in enumerate(self.part_dict.items()):
             # if i > 0: continue
            rotations.append(part.gaussian_model.get_rotation)
        return torch.cat(rotations, dim=0)
    @property
    def get_features(self):
        shs = []
        for part_name, part in self.part_dict.items():
            shs.append(part.gaussian_model.get_features)
        return torch.cat(shs, dim=0)
    
    def update_learning_rate(self, iteration):
        for part_name, part in self.part_dict.items():
            part.gaussian_model.update_learning_rate(iteration)

    def update_learning_rate_tracking(self, iteration):
        # TODO: update the learning rate for each part
        # for part_name, part in self.part_dict.items():
        #     part.gaussian_model.update_learning_rate_tracking(iteration)
        pass

    def update_learning_rate_texture_learning(self, iteration):
        # TODO: update the learning rate for each part
        for part_name, part in self.part_dict.items():
            part.gaussian_model.texture_learning_update_learning_rate(iteration)
        pass

    def oneupSHdegree(self):
        for part_name, part in self.part_dict.items():
            part.gaussian_model.oneupSHdegree()
    @property
    def active_sh_degree(self):
        return self.part_dict['shaft'].gaussian_model.active_sh_degree
    @property
    def get_semantics(self):
        semantics = []
        if not(hasattr(self, 'l_gripper_semantic_mask') and hasattr(self, 'r_gripper_semantic_mask')):
            xyz = self.part_dict['l_gripper'].gaussian_model.get_xyz
            length = self.part_dict['l_gripper'].gaussian_model.vertices[...,0].max() - self.part_dict['l_gripper'].gaussian_model.vertices[...,0].min()
            self.l_gripper_semantic_mask = xyz[..., 0] < self.part_dict['l_gripper'].gaussian_model.vertices[...,0].min() + 0.46 * length
            
            xyz = self.part_dict['r_gripper'].gaussian_model.get_xyz
            self.r_gripper_semantic_mask = xyz[..., 0] < self.part_dict['r_gripper'].gaussian_model.vertices[...,0].min() + 0.46 * length

        for i,(part_name, part) in enumerate(self.part_dict.items()):
             # if i > 0: continue
            semantic = torch.zeros_like(part.gaussian_model.get_xyz)
            semantic[:, min(i,2)] = 1
            if 'gripper' in part_name:
                if 'l' in part_name:
                    semantic[self.l_gripper_semantic_mask, 2] = 0
                    semantic[self.l_gripper_semantic_mask, 1] = 1
                else:
                    semantic[self.r_gripper_semantic_mask, 2] = 0
                    semantic[self.r_gripper_semantic_mask, 1] = 1
                    # semantic[...,2] = 0
            semantics.append(semantic)
        return torch.cat(semantics, dim=0)
    
    def get_gaussian_models(self):
        return {part_name: part.gaussian_model for part_name, part in self.part_dict.items()}
    
    def update_alpha(self): 
        for part_name, part in self.part_dict.items():
            part.gaussian_model.update_alpha()

    def require_grad(self, requires_grad):
        for part_name, part in self.part_dict.items():
            part.gaussian_model.require_grad(requires_grad)
    def prepare_scaling_rot(self):
        for part_name, part in self.part_dict.items():
            if part_name in ['wrist', 'l_gripper', 'r_gripper']:
                part.gaussian_model.prepare_scaling_rot(use_triangle_scale=True)
            else:
                part.gaussian_model.prepare_scaling_rot(use_triangle_scale=False)
    

    def get_new_mesh(self):
        new_mesh_all = []
        for part_name, part in self.part_dict.items():
            part_new_mesh = part.gaussian_model.new_mesh
            new_mesh_all.append(part_new_mesh)
        new_mesh = trimesh.util.concatenate(new_mesh_all)
        return new_mesh
    
    def get_render_params(self):
        _xyz = []
        _opacities = []
        _rotations = []
        _scalings = []
        _features = []

        for part_name, part in self.part_dict.items():
            render_params = part.render_params
            _xyz.append(render_params['xyz'])
            _opacities.append(render_params['opacity'])
            _rotations.append(render_params['rotation'])
            _scalings.append(render_params['scale'])
            _features.append(render_params['features'])
        return {'xyz': torch.cat(_xyz, dim=0), 'opacity': torch.cat(_opacities, dim=0), 
                'rotation': torch.cat(_rotations, dim=0), 'scale': torch.cat(_scalings, dim=0), 
                'feature': torch.cat(_features, dim=0)}
    
    def calculate_lateral_rotation(self, rot_angle, rot_wrist2camera=torch.eye(3), 
                                   trans_wrist2camera=torch.zeros(3), alpha=torch.tensor(0)):
               
       # wrist to camera transformation (initialized)
        wrist2camera_transformation = torch.eye(4).cuda()
        wrist2camera_transformation[:3, :3] = rot_wrist2camera
        wrist2camera_transformation[:3, 3] = trans_wrist2camera

        # wrist transformation
        rot_wrist = self._rodrigues_rotation_matrix(torch.tensor([0, 1, 0], device="cuda"), alpha)
        
        wrist2shaft_transformation = torch.eye(4).cuda()
        wrist2shaft_transformation[:3, :3] = rot_wrist
        wrist2shaft_transformation[:3, 3] = torch.tensor([0.2159, 0, 0], device='cuda')
        shaft2camera_transformation = wrist2camera_transformation @ torch.inverse(wrist2shaft_transformation)

        # generate shaft lateral rotation
        lateral_shaft_transformation = torch.eye(4).cuda()
        lateral_shaft_transformation[:3, :3] = self._rodrigues_rotation_matrix(torch.tensor([1, 0, 0], device='cuda'), rot_angle.unsqueeze(0))
        
        wrist2camera_transformation = (shaft2camera_transformation @ lateral_shaft_transformation) @ wrist2shaft_transformation

        return wrist2camera_transformation[:3,:3]
    
    def forward_kinematics(self, rot_wrist2camera=torch.eye(3), trans_wrist2camera=torch.zeros(3),
                            alpha=torch.tensor(0), theta_l=torch.tensor(0), theta_r=torch.tensor(0),
                            gs_grad = False, pose_grad = True, with_mesh = False):
       
        self.with_mesh = with_mesh
       # wrist to camera transformation (initialized)
        wrist2camera_transformation = torch.eye(4).cuda()
        wrist2camera_transformation[:3, :3] = rot_wrist2camera
        wrist2camera_transformation[:3, 3] = trans_wrist2camera

        # wrist transformation
        rot_wrist = self._rodrigues_rotation_matrix(torch.tensor([0, 1, 0], device="cuda"), alpha)
        self.rot_wrist = rot_wrist
        wrist2shaft_transformation = torch.eye(4).cuda()
        wrist2shaft_transformation[:3, :3] = rot_wrist
        wrist2shaft_transformation[:3, 3] = torch.tensor([0.2159, 0, 0], device='cuda')
        shaft2camera_transformation = wrist2camera_transformation @ torch.inverse(wrist2shaft_transformation)

        # shaft transformation
        shaft_transformation = shaft2camera_transformation
        # wrist transformation
        wrist_transformation = wrist2shaft_transformation
        
        # left gripper transformation
        l_gripper_transformation = torch.eye(4).cuda()
        l_gripper_transformation[:3, :3] = self._rodrigues_rotation_matrix(torch.tensor([0, 0, 1], device='cuda'), theta_l)
        l_gripper_transformation[:3, 3] = torch.tensor([0.009, 0, 0])
        # right gripper transformation
        r_gripper_transformation = torch.eye(4).cuda()
        r_gripper_transformation[:3, :3] = self._rodrigues_rotation_matrix(torch.tensor([0, 0, 1], device='cuda'), -theta_r)
        r_gripper_transformation[:3, 3] = torch.tensor([0.009, 0, 0])
        # apply the transformation
        self._apply_transformation('shaft', shaft_transformation, gs_grad, pose_grad)
        self.shaft_transformation = shaft_transformation
        self.wrist_transform = shaft_transformation @ wrist_transformation
        self._apply_transformation('wrist', self.wrist_transform, gs_grad, pose_grad)
        self.l_gripper_transformation = self.wrist_transform @ l_gripper_transformation
        self._apply_transformation('l_gripper', self.l_gripper_transformation, gs_grad, pose_grad)
        self.r_gripper_transformation = self.wrist_transform @ r_gripper_transformation
        self._apply_transformation('r_gripper', self.r_gripper_transformation, gs_grad, pose_grad)

    def render_trimesh(self,K):
        # render the trimesh
        mesh_list = []
        for part_name, mesh in self.new_mesh_dict.items():
            mesh_list.append(mesh)
        combined_mesh = trimesh.util.concatenate(mesh_list)
        # Define the camera intrinsic matrix K
        K = K

        # Define the camera extrinsic matrix (identity matrix)
        extrinsic = np.eye(4) #@ (self.bias2world.cpu().numpy())
        extrinsic[:, 1:3] *= -1

        # Render the scene
        # Render the scene using pyrender

        # Create a pyrender scene
        pyrender_scene = pyrender.Scene()

        # material = pyrender.MetallicRoughnessMaterial(
        # metallicFactor=0.0,  # Set to 0 for non-metallic
        # roughnessFactor=0.8,  # Increase roughness to reduce shininess
        # baseColorFactor=[0.8, 0.8, 0.8, 1.0],  # Set base color to a less bright value
        # alphaMode='OPAQUE')
        
        # Add the combined mesh to the pyrender scene
        mesh = pyrender.Mesh.from_trimesh(combined_mesh)
        pyrender_scene.add(mesh)

        # Create a camera
        camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear = 0.001, zfar = 0.6)
        pyrender_scene.add(camera, pose=extrinsic)
        # Create a light source
        light = pyrender.PointLight(intensity = 1)
        light_pose = extrinsic.copy()
        light_pose[:, 1:3] *= -1
        light_pose[2, 3] += 0.1
        light_pose[0, 3] += 0.1
        pyrender_scene.add(light, pose=light_pose)

        # Render the scene
        renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=512)
        color, depth = renderer.render(pyrender_scene)

        # Save the image
        # cv2.imwrite('rendered_image.png', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        return color, depth

        

    
# instrument = Instrument(shaft_mesh, wrist_mesh, gripper_mesh_left, gripper_mesh_right)
# instrument.forward_kinematics(alpha=torch.pi/4, theta_l=torch.pi/12, theta_r=-torch.pi/12)
# combined_mesh = trimesh.util.concatenate([instrument.wrist.mesh, instrument.shaft.mesh, instrument.l_gripper.mesh, instrument.r_gripper.mesh])
# combined_mesh.show()
