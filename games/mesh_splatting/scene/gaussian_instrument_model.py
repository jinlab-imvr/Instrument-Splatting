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

import torch
import numpy as np

from torch import nn
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid, rot_to_quat_batch
from utils.sh_utils import RGB2SH
from games.mesh_splatting.utils.graphics_utils import MeshPointCloud




class GaussianMeshModel(GaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.point_cloud = None
        self._shaft_alpha = torch.empty(0)
        self._wrist_alpha = torch.empty(0)
        self._l_gripper_alpha = torch.empty(0)
        self._r_gripper_alpha = torch.empty(0)

        self._shaft_scale = torch.empty(0)
        self._wrist_scale = torch.empty(0)
        self._l_gripper_scale = torch.empty(0)
        self._r_gripper_scale = torch.empty(0)

        self.shaft_alpha = torch.empty(0)
        self.wrist_alpha = torch.empty(0)
        self.l_gripper_alpha = torch.empty(0)
        self.r_gripper_alpha = torch.empty(0)


        self.softmax = torch.nn.Softmax(dim=2)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.update_alpha_func = self.softmax

        self.shaft_vertices = None
        self.wrist_vertices = None
        self.l_gripper_vertices = None
        self.r_gripper_vertices = None

        self.shaft_faces = None
        self.wrist_faces = None
        self.l_gripper_faces = None
        self.r_gripper_faces = None

        self.shaft_triangles = None
        self.wrist_triangles = None
        self.l_gripper_triangles = None
        self.r_gripper_triangles = None

        self.part_names = ['shaft', 'wrist', 'l_gripper', 'r_gripper']

    @property
    def get_xyz(self):
        return torch.concat([self._shaft_xyz, self._wrist_xyz, self._l_gripper_xyz, self._r_gripper_xyz], dim=0)
        
    def create_from_gaussian_parts(self, gs_dict: MeshPointCloud, spatial_lr_scale: float, gs_param_file = None):
        
        self.shaft_triangles = gs_dict['shaft'].triangles
        self.wrist_triangles = gs_dict['wrist'].triangles
        self.l_gripper_triangles = gs_dict['l_gripper'].triangles
        self.r_gripper_triangles = gs_dict['r_gripper'].triangles
        
        self._shaft_alpha = nn.Parameter(gs_dict['shaft'].alpha.detach().requires_grad_(True))
        self._wrist_alpha = nn.Parameter(gs_dict['wrist'].alpha.detach().requires_grad_(True))
        self._l_gripper_alpha = nn.Parameter(gs_dict['l_gripper'].alpha.detach().requires_grad_(True))
        self._r_gripper_alpha = nn.Parameter(gs_dict['r_gripper'].alpha.detach().requires_grad_(True))

        self.shaft_vertices = gs_dict['shaft'].vertices
        self.wrist_vertices = gs_dict['wrist'].vertices
        self.l_gripper_vertices = gs_dict['l_gripper'].vertices
        self.r_gripper_vertices = gs_dict['r_gripper'].vertices

        self.shaft_faces = gs_dict['shaft'].faces
        self.wrist_faces = gs_dict['wrist'].faces
        self.l_gripper_faces = gs_dict['l_gripper'].faces
        self.r_gripper_faces = gs_dict['r_gripper'].faces

        self.update_alpha()
        
        self._shaft_features_dc = nn.Parameter(gs_dict['shaft'].features_dc.detach().requires_grad_(True))
        self._wrist_features_dc = nn.Parameter(gs_dict['wrist'].features_dc.detach().requires_grad_(True))
        self._l_gripper_features_dc = nn.Parameter(gs_dict['l_gripper'].features_dc.detach().requires_grad_(True))
        self._r_gripper_features_dc = nn.Parameter(gs_dict['r_gripper'].features_dc.detach().requires_grad_(True))

        self._shaft_features_rest = nn.Parameter(gs_dict['shaft'].features_rest.detach().requires_grad_(True))
        self._wrist_features_rest = nn.Parameter(gs_dict['wrist'].features_rest.detach().requires_grad_(True))
        self._l_gripper_features_rest = nn.Parameter(gs_dict['l_gripper'].features_rest.detach().requires_grad_(True))
        self._r_gripper_features_rest = nn.Parameter(gs_dict['r_gripper'].features_rest.detach().requires_grad_(True))

        self._shaft_scale = nn.Parameter(gs_dict['shaft']._scale.detach().requires_grad_(True))
        self._wrist_scale = nn.Parameter(gs_dict['wrist']._scale.detach().requires_grad_(True))
        self._l_gripper_scale = nn.Parameter(gs_dict['l_gripper']._scale.detach().requires_grad_(True))
        self._r_gripper_scale = nn.Parameter(gs_dict['r_gripper']._scale.detach().requires_grad_(True))

        self._shaft_opacity = nn.Parameter(gs_dict['shaft'].opacity.detach().requires_grad_(True))
        self._wrist_opacity = nn.Parameter(gs_dict['wrist'].opacity.detach().requires_grad_(True))
        self._l_gripper_opacity = nn.Parameter(gs_dict['l_gripper'].opacity.detach().requires_grad_(True))
        self._r_gripper_opacity = nn.Parameter(gs_dict['r_gripper'].opacity.detach().requires_grad_(True))

        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def build_rotation(self, r):

        q = F.normalize(r, p=2, dim=1)
        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R

    def multiply_quaternions(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Performs batch-wise quaternion multiplication.

        Given two quaternions, this function computes their product. The operation is
        vectorized and can be performed on batches of quaternions.

        Args:
            q: A tensor representing the first quaternion or a batch of quaternions.
            Expected shape is (... , 4), where the last dimension contains quaternion components (w, x, y, z).
            r: A tensor representing the second quaternion or a batch of quaternions with the same shape as q.
        Returns:
            A tensor of the same shape as the input tensors, representing the product of the input quaternions.
        """
        w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]

        w = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
        x = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0
        y = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0
        z = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
        return torch.stack((w, x, y, z), dim=-1)

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

        return torch.stack([w, x, y, z])

    def apply_transform(self, shaft_transformation: torch.Tensor, wrist_transformation: torch.Tensor, l_gripper_transformation: torch.Tensor, r_gripper_transformation: torch.Tensor):
        """
        Apply transformation to the vertices of the mesh
        """
        for transformation in [shaft_transformation, wrist_transformation, l_gripper_transformation, r_gripper_transformation]:
            
            transformation_t = transformation.t()
            
            self._shaft_xyz = torch.matmul(self._shaft_xyz.detach(), transformation_t[:3,:3]) + transformation_t[3,:3].unsqueeze(0)
            quat = self.matrix_to_quaternion(transformation[:3,:3])
            quat = F.normalize(quat, p =2, dim=0)
            self._rotation = self.multiply_quaternions(self.get_rotation.detach(), quat.unsqueeze(0)) #.squeeze(0)
            self.vertices = torch.matmul(self.vertices.detach(), transformation_t[:3,:3]) + transformation_t[3,:3].unsqueeze(0)
            self.triangles = torch.bmm(self.triangles.detach(), transformation_t[:3,:3].unsqueeze(0).repeat(self.triangles.shape[0], 1, 1))\
                                + transformation_t[3,:3].unsqueeze(0)


    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.

        """
        _shaft_xyz = torch.matmul(
            self.shaft_alpha,
            self.shaft_triangles
        )
        assert _shaft_xyz.shape[1] == 1
        self._shaft_xyz = _shaft_xyz.reshape(
                _shaft_xyz.shape[0] * _shaft_xyz.shape[1], 3
            )
        
        _wrist_xyz = torch.matmul(
            self.wrist_alpha,
            self.wrist_triangles
        )
        assert _wrist_xyz.shape[1] == 1
        self._wrist_xyz = _wrist_xyz.reshape(
                _wrist_xyz.shape[0] * _wrist_xyz.shape[1], 3
            )
        
        _l_gripper_xyz = torch.matmul(
            self.l_gripper_alpha,
            self.l_gripper_triangles
        )
        assert _l_gripper_xyz.shape[1] == 1
        self._l_gripper_xyz = _l_gripper_xyz.reshape(
                _l_gripper_xyz.shape[0] * _l_gripper_xyz.shape[1], 3
            )
        
        _r_gripper_xyz = torch.matmul(
            self.r_gripper_alpha,
            self.r_gripper_triangles
        )
        assert _r_gripper_xyz.shape[1] == 1
        self._r_gripper_xyz = _r_gripper_xyz.reshape(
                _r_gripper_xyz.shape[0] * _r_gripper_xyz.shape[1], 3
            )
        

    def prepare_scaling_rot(self, use_triangle_scale = True, eps=1e-8):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from
        centroid to 2nd vertex onto subspace spanned by v0 and v1.
        """

        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)

        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        triangles = self.triangles
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + eps)
        means = torch.mean(triangles, dim=1)
        v1 = triangles[:, 1] - means
        # delta_rad = self._rot
        # cos_delta = torch.cos(delta_rad)
        # sin_delta = torch.sin(delta_rad)

        # v1_rotated = cos_delta * v1 + sin_delta * torch.cross(v0, v1, dim=1) + (1 - cos_delta) * dot(v1, v0) * v0
        # v1 = v1_rotated


        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps
        v1 = v1 / v1_norm
        v2_init = triangles[:, 2] - means
        
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        s0 = eps * torch.ones_like(s1)
        scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
        scales = scales.broadcast_to((*self.alpha.shape[:2], 3))
        if use_triangle_scale:
            self._scaling = torch.log(
                torch.nn.functional.relu(self._scale*scales.flatten(start_dim=0, end_dim=1)) + eps
            )
        else:
            self._scaling = torch.log(
                torch.nn.functional.relu(self._scale) + eps
            )
        
        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        rotation = rotation.transpose(-2, -1)
        self._rotation = rot_to_quat_batch(rotation)

    def update_alpha(self):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1+ alpha2+ alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 +alpha3 >= 0
        """
        
        self.shaft_alpha = torch.relu(self._shaft_alpha) + 1e-8
        self.shaft_alpha = self.shaft_alpha / self.shaft_alpha.sum(dim=-1, keepdim=True)
        self.shaft_triangles = self.shaft_vertices[self.shaft_faces]

        self.wrist_alpha = torch.relu(self._wrist_alpha) + 1e-8
        self.wrist_alpha = self.wrist_alpha / self.wrist_alpha.sum(dim=-1, keepdim=True)
        self.wrist_triangles = self.wrist_vertices[self.wrist_faces]

        self.l_gripper_alpha = torch.relu(self._l_gripper_alpha) + 1e-8
        self.l_gripper_alpha = self.l_gripper_alpha / self.l_gripper_alpha.sum(dim=-1, keepdim=True)
        self.l_gripper_triangles = self.l_gripper_vertices[self.l_gripper_faces]

        self.r_gripper_alpha = torch.relu(self._r_gripper_alpha) + 1e-8
        self.r_gripper_alpha = self.r_gripper_alpha / self.r_gripper_alpha.sum(dim=-1, keepdim=True)
        self.r_gripper_triangles = self.r_gripper_vertices[self.r_gripper_faces]

        self._calc_xyz()

    def training_setup(self, training_args):
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l_params = [
            # {'params': [self.vertices], 'lr': training_args.vertices_lr, "name": "vertices"},
            {'params': [self._shaft_alpha], 'lr': training_args.alpha_lr, "name": "shaft_alpha"},
            {'params': [self._wrist_alpha], 'lr': training_args.alpha_lr, "name": "wrist_alpha"},
            {'params': [self._l_gripper_alpha], 'lr': training_args.alpha_lr, "name": "l_gripper_alpha"},
            {'params': [self._r_gripper_alpha], 'lr': training_args.alpha_lr, "name": "r_gripper_alpha"},

            {'params': [self._shaft_features_dc], 'lr': training_args.feature_lr, "name": "shaft_f_dc"},
            {'params': [self._wrist_features_dc], 'lr': training_args.feature_lr, "name": "wrist_f_dc"},
            {'params': [self._l_gripper_features_dc], 'lr': training_args.feature_lr, "name": "l_gripper_f_dc"},
            {'params': [self._r_gripper_features_dc], 'lr': training_args.feature_lr, "name": "r_gripper_f_dc"},

            {'params': [self._shaft_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "shaft_f_rest"},
            {'params': [self._wrist_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "wrist_f_rest"},
            {'params': [self._l_gripper_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "l_gripper_f_rest"},
            {'params': [self._r_gripper_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "r_gripper_f_rest"},
            
            {'params': [self._shaft_scale], 'lr': training_args.scaling_lr, "name": "shaft_scaling"},
            {'params': [self._wrist_scale], 'lr': training_args.scaling_lr, "name": "wrist_scaling"},
            {'params': [self._l_gripper_scale], 'lr': training_args.scaling_lr, "name": "l_gripper_scaling"},
            {'params': [self._r_gripper_scale], 'lr': training_args.scaling_lr, "name": "r_gripper_scaling"},

            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scale], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._rot], 'lr': training_args.scaling_lr, "name": "rot"}
        ]

        self.optimizer = torch.optim.Adam(l_params, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration) -> None:
        """ Learning rate scheduling per step """
        pass

    def save_ply(self, path):
        self._save_ply(path)

        attrs = self.__dict__
        additional_attrs = [
            '_alpha',
            '_scale',
            #'_rotation',
            'point_cloud',
            'triangles',
            'vertices',
            'faces'
        ]

        save_dict = {}
        for attr_name in additional_attrs:
            save_dict[attr_name] = attrs[attr_name]

        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        torch.save(save_dict, path_model)

    def load_ply(self, path):
        self._load_ply(path)
        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        params = torch.load(path_model)
        alpha = params['_alpha']
        scale = params['_scale']
       # rotate = params['_rotation']
        if 'vertices' in params:
            self.vertices = params['vertices']
        if 'triangles' in params:
            self.triangles = params['triangles']
        if 'faces' in params:
            self.faces = params['faces']
        # point_cloud = params['point_cloud']
        self._alpha = nn.Parameter(alpha)
        self._scale = nn.Parameter(scale)
       # self._rotate = nn.Parameter(rotate)
