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
from instrument_splatting.scene.gaussian_model import GaussianModel
from utils.general_utils import rot_to_quat_batch
from utils.sh_utils import RGB2SH
from instrument_splatting.utils.graphics_utils import MeshPointCloud



class GaussianInstrumentModel(GaussianModel):

    def __init__(self, sh_degree: int, use_triangle_scale, scale_coef):

        super().__init__(sh_degree)
        self.point_cloud = None
        self._alpha = torch.empty(0)
        self._scale = torch.empty(0)
        self.alpha = torch.empty(0)
        self.softmax = torch.nn.Softmax(dim=2)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.update_alpha_func = self.softmax

        self.vertices = None
        self.faces = None
        self.triangles = None

        self.use_triangle_scale = use_triangle_scale
        self.scale_coef = scale_coef

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) * self.scale_coef
    
    @property
    def get_scaling_iso(self):
        scales = self.scaling_activation(self._scaling) * self.scale_coef
        return torch.stack((scales[:,0], scales[:,0], scales[:,0]), dim=1)


    def create_from_pcd(self, pcd: MeshPointCloud, spatial_lr_scale: float, gs_param_file = None):
        """
            Create the Gaussian Splatting model from a given Mesh priors following MeshGS paper. However, in Instrument-Splatting,
            we didn't use mesh to initialize the Gaussian Splatting model. Instead, we use gs_param_file to read GS parameters for Stage II and Stage III. 
            
            :param self: Description
            :param pcd: Description
            :type pcd: MeshPointCloud
            :param spatial_lr_scale: Description
            :type spatial_lr_scale: float
            :param gs_param_file: Description
            :type gs_param_file: str or None
        """
        
        if gs_param_file is not None:
            data = torch.load(gs_param_file)
        else:
            self.point_cloud = pcd
            self.triangles = self.point_cloud.triangles
            self.spatial_lr_scale = spatial_lr_scale
            pcd_alpha_shape = pcd.alpha.shape

            print("Number of faces: ", pcd_alpha_shape[0])
            print("Number of points at initialisation in face: ", pcd_alpha_shape[1])

            alpha_point_cloud = pcd.alpha.float().cuda()
            scale = torch.ones((pcd.points.shape[0], 3)).float().cuda()

            print("Number of points at initialisation : ",
                alpha_point_cloud.shape[0] * alpha_point_cloud.shape[1])

            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0

            rots = torch.zeros((pcd.points.shape[0], 1), device="cuda").float()
            rots[:, 0] = 0.0
            self._rot = rots

            self.vertices = \
            self.point_cloud.vertices.clone().detach().cuda().float()
            self.faces = torch.tensor(self.point_cloud.faces).cuda()
            self._alpha = alpha_point_cloud# check update_alpha

        if gs_param_file is not None:
            
            alpha_point_cloud = data['alpha']
            scale = data['scale']
            scaling = data['scaling']
            opacities = data['opacity']
            xyz = data['xyz']
            features = torch.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0:1] = data['features_dc'].permute(0, 2, 1)
            features[:, :3, 1:] = data['features_rest'].permute(0, 2, 1)
        else:
            opacities = torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda") #* 12

        
        if gs_param_file is None:
            self._update_alpha() # define the self._xyz inside this function
        else:
            self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scale = scale
    
        
        if gs_param_file is None:
            self._prepare_scaling_rot()
        else:
            self._scaling = nn.Parameter(scaling.requires_grad_(True))
            self._rotation = nn.Parameter(data['rotation'].requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
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

        
    def apply_transform_params(self, transformation: torch.Tensor, gs_grad: bool = False):
        """
        Apply transformation to the vertices of the mesh
        """

        transformation_t = transformation.t()
        if gs_grad == False:
            _xyz = self.get_xyz.detach().clone()
            rotations = self.get_rotation.detach().clone()
            _opcaity = self.get_opacity.detach().clone()
            _scale = self.get_scaling.detach().clone()
            _features = self.get_features.detach().clone()
        else:
            _xyz = self.get_xyz
            rotations = self.get_rotation
            _opcaity = self.get_opacity
            _scale = self.get_scaling
            _features = self.get_features

        _xyz = torch.matmul(_xyz, transformation_t[:3,:3]) + transformation_t[3,:3].unsqueeze(0)

        quat = self.matrix_to_quaternion(transformation[:3,:3])
        quat = F.normalize(quat, p =2, dim=0)
        _rotation = self.multiply_quaternions(rotations, quat.unsqueeze(0)) #.squeeze(0)
        _rotation = F.normalize(_rotation, p =2, dim=1)
        return {
            "xyz": _xyz,
            "rotation": _rotation,
            "opacity": _opcaity,
            "scale": _scale,
            'features': _features
        }
        


    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.

        """
        _xyz = torch.matmul(
            self.alpha,
            self.triangles
        )
        assert _xyz.shape[1] == 1
        _xyz = _xyz.reshape(
                _xyz.shape[0] * _xyz.shape[1], 3
            )
        self._xyz = nn.Parameter(_xyz.requires_grad_(True))

    def _prepare_scaling_rot(self, use_triangle_scale = True, eps=1e-8):
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
        if self.use_triangle_scale:
            self._scaling = torch.log(
                torch.nn.functional.relu(self._scale*scales.flatten(start_dim=0, end_dim=1)) + eps
            )
        else:
            self._scaling = torch.log(
                torch.nn.functional.relu(self._scale*0.001) + eps
            )
        
        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)

        rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        rotation = rotation.transpose(-2, -1)
        self._rotation = rot_to_quat_batch(rotation)
        self._scaling = nn.Parameter(self._scaling.requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation.requires_grad_(True))

    # def require_grad(self, requires_grad: bool):
    #     self._alpha.requires_grad = requires_grad
    #     self._features_dc.requires_grad = requires_grad
    #     self._features_rest.requires_grad = requires_grad
    #     self._opacity.requires_grad = requires_grad
    #     self._scale.requires_grad = requires_grad
        # self._rot.requires_grad = requires_grad
    def _update_alpha(self):
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
        self.alpha = torch.relu(self._alpha) + 1e-8
        self.alpha = self.alpha / self.alpha.sum(dim=-1, keepdim=True)
        self.triangles = self.vertices[self.faces]
        self._calc_xyz()

    def training_setup(self, training_args):
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.percent_dense = training_args.percent_dense if hasattr(training_args, 'percent_dense') else 0.01
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        l_params = [
            {'params': [self._xyz], 'lr': training_args.vertices_lr*1e-3, "name": "xyz"},
            # {'params': [self._xyz], 'lr': training_args.vertices_lr, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr , "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l_params, lr=0.0, eps=1e-15)

    
    def update_learning_rate(self, iteration) -> None:
        """ Learning rate scheduling per step """
        pass
    
    def texture_learning_update_learning_rate(self, iteration) -> None:
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "f_rest" or param_group["name"] == "f_dc":
                lr = param_group['lr']
                lr = 0.5 * lr
                param_group['lr'] = lr
                return lr
            
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
