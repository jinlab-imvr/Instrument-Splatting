import trimesh
import  numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from instrument_splatting.utils.graphics_utils import MeshPointCloud
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
import os
# from utils.part import Part
# from instrument_splatting.scene.dataset_readers import InstrumentSceneInfo
from instrument_splatting.scene.gaussian_instrument_model import GaussianInstrumentModel

    
class Part(object):
    def __init__(self, name, mesh: trimesh.Trimesh, world2part=None):
        self.mesh = mesh.copy()
        if isinstance(world2part, torch.Tensor):
            world2part = world2part.numpy()
        self.transformation = world2part if world2part is not None else np.eye(4)
        self.mesh.apply_transform(world2part)
        self.name = name
        world2part = torch.tensor(world2part, device="cuda")
        bias2world = torch.eye(4)
        bias2world[:3, :3] = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
        bias2world = bias2world.float().cuda()
        world2part =  world2part @torch.linalg.inv(bias2world)
        if name is "shaft":
            self.keypoints = torch.tensor([[-0.009026, -1e-6, 0.004178]], device="cuda") @ world2part[:3,:3].t() + world2part[:3,3][None] 
        elif name is "wrist":
            self.keypoints = torch.tensor([[0.002693, -0.000254, 0.002975], [0.009,0,0.002344]], device="cuda") @ world2part[:3,:3].t() + world2part[:3,3][None] 
        # elif "l_gripper" in name:
        #     self.keypoints = torch.tensor([[0.018439, 0.000635, -0.0003056]], device="cuda") @ world2part[:3,:3].t() + world2part[:3,3][None] 
        # elif "r_gripper" in name:
        #     self.keypoints = torch.tensor([[0.018439, -0.000635, -0.0003056]], device="cuda") @ world2part[:3,:3].t() + world2part[:3,3][None] 
        elif "l_gripper" in name:
            self.keypoints = torch.tensor([[0.017935, 0.000635, 0.000748     ]], device="cuda") @ world2part[:3,:3].t() + world2part[:3,3][None] 
        elif "r_gripper" in name:
            self.keypoints = torch.tensor([[0.017935, -0.000635, 0.000748]], device="cuda") @ world2part[:3,:3].t() + world2part[:3,3][None] 

    def get_vertices(self):
        self.vertices = torch.tensor(self.mesh.vertices)
        return self.vertices


    def get_nomals(self, faces=None):
        self.normals = torch.tensor(self.mesh.vertex_normals)
        self.get_vertices()
        if faces is None:
            # when faces is not provided, we assume the faces are the same as the mesh faces
            faces = self.mesh.faces
        self.normals = self.normals[torch.tensor(faces).long()].float()
        return self.normals

    def get_faces_and_triangles(self):    
        faces = self.mesh.faces
        self.faces = faces
        self.get_vertices()
        self.triangles =  self.vertices[torch.tensor(faces).long()].float()
        return self.triangles

    def get_valid_triangle_masks(self, triangles = None, remove_area_threshold=1e-10, max_length_threshold=5e-2):
        # remove those primitives with small-area triangles
        if triangles is None:
            triangles = self.triangles
        else:
            # update the triangles with the given triangles
            self.triangles = triangles
        A = torch.norm(triangles[:,0] - triangles[:,1], dim=1).double()
        B = torch.norm(triangles[:,1] - triangles[:,2], dim=1).double()
        C = torch.norm(triangles[:,2] - triangles[:,0], dim=1).double()
        s = (A + B + C) / 2
        areas = torch.sqrt(s * (s - A) * (s - B) * (s - C))
        remove_mask = areas < remove_area_threshold

        abc = torch.stack([A, B, C], dim=1)
        max_length, max_idxes = torch.max(abc, dim=1)

        densify_mask = max_length > max_length_threshold

        
        return remove_mask, densify_mask, max_length.float(), max_idxes
    

    def  storePly(self, path, xyz, rgb):
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


    def create_meshpoint_cloud_with_ray_tracing(self, surf_face_idxes, surf_alphas, surf_vertices, ply_path):
        
        triangles = self.get_faces_and_triangles()
        
        # remove those primitives with small-area triangles
        # self.triangles = triangles[surf_face_idxes]
        # NOTE: self.vertices include all the vertices of the mesh
        
        faces = self.faces
        self.faces = torch.tensor(faces)
        self.faces = self.faces[surf_face_idxes]
        self.triangles = self.vertices[self.faces]
        # initialize the points with even per-triangle sampling
        num_pts = surf_face_idxes.shape[0]
        nomals = self.get_nomals(self.faces)
        alpha = surf_alphas
        # normalize the alpha
        alpha = alpha / alpha.sum(dim=-1, keepdim=True)
        self.alpha = alpha.unsqueeze(1)
        assert self.alpha.shape == (num_pts,1, 3)
        xyz = torch.bmm(self.alpha, self.triangles)
        self.xyz = xyz.squeeze(1)
        assert self.xyz.shape == (num_pts, 3)
        # self.xyz = xyz.reshape(num_pts, 3)
        self.rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        self.rot = self.rot.repeat(num_pts, 1) 
        self.scale = torch.ones((num_pts, 3), dtype=torch.float32)
        self.opacity = torch.ones((num_pts, 1), dtype=torch.float32)
        self.shs = torch.rand((num_pts, 3), dtype=torch.float32) / 255.0

        self.pcd = MeshPointCloud(alpha=self.alpha, points=self.xyz, colors=self.shs, normals=nomals, vertices=self.vertices,
                                  faces=self.faces, transform_vertices_function=None, triangles=self.triangles.cuda())
        self.storePly(ply_path, self.pcd.points, SH2RGB(self.shs) * 255)

    def create_meshpoint_cloud(self, num_pts_per_triangle, ply_path, densify_treshold = 1e-3):
        
        triangles = self.get_faces_and_triangles()
        remove_mask, densify_mask, max_lengths, max_idxes = self.get_valid_triangle_masks()
        # remove those primitives with small-area triangles
        triangles = triangles[~remove_mask]
        self.triangles = torch.cat([triangles]*num_pts_per_triangle, dim=0)
        faces = self.faces[~remove_mask]
        faces = torch.tensor(faces)
        self.faces = torch.cat([faces]*num_pts_per_triangle, dim=0)
        densify_mask = densify_mask[~remove_mask]
        # initialize the points with even per-triangle sampling
        num_pts = num_pts_per_triangle * triangles.shape[0]
        nomals = self.get_nomals(self.faces)
        alpha = torch.rand((triangles.shape[0] * num_pts_per_triangle, 1, 3))
        # normalize the alpha
        alpha = alpha / alpha.sum(dim=2, keepdim=True)
        self.alpha = alpha
        xyz = torch.bmm(alpha, self.triangles)
        self.xyz = xyz.squeeze(1)
        assert self.xyz.shape == (num_pts, 3)
        # self.xyz = xyz.reshape(num_pts, 3)
        self.rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        self.rot = self.rot.repeat(num_pts, 1)
        self.scale = torch.ones((num_pts, 3), dtype=torch.float32)
        self.opacity = torch.ones((num_pts, 1), dtype=torch.float32)
        self.shs = torch.rand((num_pts, 3), dtype=torch.float32) / 255.0
        
        # densely sample points on large-area triangles
        if densify_mask.sum() > 0:
            triangles_densify = triangles[densify_mask]
            faces_densify = faces[densify_mask]
            max_lengths = max_lengths[~remove_mask][densify_mask]
            max_idxes = max_idxes[~remove_mask][densify_mask]
            densify_triangle_num_pts = []
            densify_shs = []
            densify_rot = []
            densify_scale = []
            densify_opacity = []
            densify_xyz = []
            densify_alpha = []
            densify_triangles = []
            densify_faces = []
            edge_dict = {0:(0,1), 1:(1,2), 2:(2,0)}
            for i, (triangle,face, max_length, idx) in enumerate(zip(triangles_densify, faces_densify, max_lengths, max_idxes)):
                
                vertex1_id, vertex2_id  = edge_dict[idx.item()]
                # vertex1 = triangle[vertex1_id]
                # vertex2 = triangle[vertex2_id]
                num_pts = int(max_length / densify_treshold)
                alpha = torch.rand((num_pts, 3))
                alpha[:,vertex1_id] = torch.linspace(0, 1, num_pts)
                alpha[:,vertex2_id] = 1 - alpha[:,vertex1_id]
                # normalize the alpha
                alpha = alpha / alpha.sum(dim=1, keepdim=True)
                # xyz rot scale opacity shs
                xyz = torch.mm(alpha, triangle) # num_pts x 3
                rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
                rot = rot.repeat(num_pts, 1)
                scale = torch.ones((num_pts, 3), dtype=torch.float32)
                opacity = torch.ones((num_pts, 1), dtype=torch.float32)
                shs = torch.rand((num_pts, 3), dtype=torch.float32) / 255.0
                densify_triangle_num_pts.append(num_pts)
                densify_shs.append(shs)
                densify_rot.append(rot)
                densify_scale.append(scale)
                densify_opacity.append(opacity)
                densify_xyz.append(xyz)
                densify_alpha.append(alpha)
                densify_triangles += [triangle.unsqueeze(0)]*num_pts
                densify_faces += [face.unsqueeze(0)] * num_pts
                
            densify_triangles = torch.cat(densify_triangles, dim=0)
            densify_faces = torch.cat(densify_faces, dim=0)
            self.triangles = torch.cat([self.triangles, densify_triangles], dim=0)
            self.faces = torch.cat([self.faces, densify_faces], dim=0)
            densify_shs = torch.cat(densify_shs, dim=0)
            densify_rot = torch.cat(densify_rot, dim=0)
            densify_scale = torch.cat(densify_scale, dim=0)
            densify_opacity = torch.cat(densify_opacity, dim=0)
            densify_xyz = torch.cat(densify_xyz, dim=0)
            densify_alpha = torch.cat(densify_alpha, dim=0)
            self.xyz = torch.cat([self.xyz, densify_xyz], dim=0)
            self.rot = torch.cat([self.rot, densify_rot], dim=0)
            self.scale = torch.cat([self.scale, densify_scale], dim=0)
            self.opacity = torch.cat([self.opacity, densify_opacity], dim=0)
            self.shs = torch.cat([self.shs, densify_shs], dim=0)
            self.alpha = torch.cat([self.alpha, densify_alpha.unsqueeze(1)], dim=0)

        self.pcd = MeshPointCloud(alpha=self.alpha, points=self.xyz, colors=self.shs, normals=nomals, vertices=self.vertices,
                                  faces=self.faces, transform_vertices_function=None, triangles=self.triangles.cuda())
        self.storePly(ply_path, self.pcd.points, SH2RGB(self.shs) * 255)
        

    def gaussian_setup(self, gaussian_model):
        self.gaussian_model = gaussian_model
        self.gaussian_model: GaussianMeshModel


    def apply_transformation_mesh(self, transformation):
        if isinstance(transformation, torch.Tensor):
            transformation = transformation.detach().cpu().numpy()
        self.new_mesh = self.mesh.copy()
        self.new_mesh.apply_transform(transformation) 
        return self.new_mesh
    
    def apply_transformation_gaussian_(self, transformation, use_triangle_scale = True):    
        # applying transformation to the gaussian model and change the parameters in the gaussian models
        self.gaussian_model.apply_transform_(transformation, use_triangle_scale = use_triangle_scale)
    
    def apply_transformation_params(self, transformation, gs_grad = False):
        """
        Apply the transformation to the Gaussian models without changing the member variables.
        gaussian_grad = False while the transformation is differentialble.
        """
        keypoints =  self.keypoints @ transformation[:3,:3].t() + transformation[:3,3][None]
        
        render_parameters = self.gaussian_model.apply_transform_params(transformation, gs_grad)
        render_parameters['keypoints'] = keypoints
        self.render_params = render_parameters
        return render_parameters
    
    def densify_mesh(self, iterations=1):
        """
        Densify the mesh by subdividing each triangle.
        
        Parameters:
        iterations (int): Number of times to subdivide the mesh.
        """
        for _ in range(iterations):
            self.mesh = self.mesh.subdivide()