import trimesh
import numpy as np
import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 2] = -vertices[:, 2]
    vertices *= c
    return vertices

def is_point_in_triangle(pt, v1, v2, v3):
    # Use barycentric coordinates to check if the point is inside the triangle
    v2v1 = v2 - v1
    v3v1 = v3 - v1
    ptv1 = pt - v1

    d00 = np.dot(v2v1, v2v1)
    d01 = np.dot(v2v1, v3v1)
    d11 = np.dot(v3v1, v3v1)
    d20 = np.dot(ptv1, v2v1)
    d21 = np.dot(ptv1, v3v1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return (u >= 0) & (v >= 0) & (w >= 0)

def ray_mesh_intersection(mesh_path):
    """
    Finds the intersection points and intersecting triangle faces of a ray with a mesh.

    Args:
        mesh_path (str): Path to the mesh file (OBJ format).
        ray_origin (np.ndarray): Origin of the ray (3,).
        ray_direction (np.ndarray): Direction of the ray (3,).

    Returns:
        np.ndarray: Intersection points (N, 3).
        np.ndarray: Indices of the intersecting triangle faces (N,).
    """
    # Load the mesh
    mesh = trimesh.load_mesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([k for k in mesh.geometry.values()])
    mesh.vertices = transform_vertices_function(mesh.vertices)
    min_x = mesh.vertices[:, 0].min()
    max_x = mesh.vertices[:, 0].max()
    faces = mesh.faces
    vertices = mesh.vertices
    step_x = int((max_x - min_x) / 5e-4)
    xx = np.linspace(min_x, max_x, step_x)
    step_y = int(0.004163 * 2 / 5e-4)
    yy = np.linspace(-0.004163, 0.004163, step_y)
    step_z = int(0.004145 * 2 / 5e-4)
    zz = np.linspace(-0.004145, 0.004145, step_z)
    face_x_z = np.meshgrid(xx, zz)
    face_y_z = np.meshgrid(yy, zz)
    face_x_y = np.meshgrid(xx, yy)
    # x-z plane; y < 0
    ray_origins1 = np.array([face_x_z[0].flatten(), np.zeros_like(face_x_z[0].flatten()), face_x_z[1].flatten()]).T
    ray_directions1 = np.zeros_like(ray_origins1)
    ray_directions1[:, 1] = -1
    # x-z plane; y > 0
    ray_directions2 = np.zeros_like(ray_origins1)
    ray_directions2[:, 1] = 1
    # y-z plane; x < 0
    ray_origins3 = np.array([np.ones_like(face_y_z[0].flatten()), face_y_z[0].flatten(), face_y_z[1].flatten()]).T
    ray_directions3 = np.zeros_like(ray_origins3)
    ray_directions3[:, 0] = -1
    # x-y plane; z < 0
    ray_origins4 = np.array([face_x_y[0].flatten(), face_x_y[1].flatten(), np.ones_like(face_x_y[0].flatten())]).T
    ray_directions4 = np.zeros_like(ray_origins4)
    ray_directions4[:, 2] = -1
    # x-y plane; z > 0
    ray_origins5 = np.array([face_x_y[0].flatten(), face_x_y[1].flatten(), -np.ones_like(face_x_y[0].flatten())]).T
    ray_directions5 = np.zeros_like(ray_origins4)
    ray_directions5[:, 2] = 1

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=np.concatenate([ray_origins1, ray_origins1, ray_origins3, ray_origins4, ray_origins5], axis=0),
        ray_directions=np.concatenate([ray_directions1, ray_directions2, ray_directions3, ray_directions4, ray_directions5], axis=0),
        multiple_hits=False
    )
    triangles = faces[index_tri]
    hitting_verts = vertices[triangles]
    hitting_verts = hitting_verts.transpose(0, 2, 1)
    
    tri_indexes = []
    alphas = []
    verts = []
    points = []
    for i,vert, location in zip(index_tri,hitting_verts, locations):
        alpha = np.linalg.lstsq(vert, location[:, None], rcond=None)[0]
        if np.sum(alpha) > 1+1e-8 or np.sum(alpha) < 1-1e-8 or np.any(alpha < 0):
            continue
        tri_indexes.append(i)
        alphas.append(alpha[:,0])
        verts.append(vert.T)
        points.append(location)
    

    radius = 0.1
    min_x = mesh.vertices[:, 0].min()
    max_x = mesh.vertices[:, 0].max()
    rotations = []
    translations = []
    
    faces = mesh.faces
    vertices = mesh.vertices

    step = int((max_x - min_x)/5e-4)
    for t in tqdm.tqdm(np.linspace(0, 2 * np.pi, 180)):
        for x_trans in np.linspace(min_x, max_x, step):
            # if x_trans > (max_x - min_x) * 2 / 3:
            #     continue
            rotation = np.array([[1,0,0],[0, np.cos(t), np.sin(t)],[0, -np.sin(t), np.cos(t)]], np.float32).T
            trans = np.array([x_trans, radius * np.cos(t), radius * np.sin(t)], np.float32)
            rotations.append(rotation)
            translations.append(trans)

            # Perform ray-mesh intersection
            mesh: trimesh.Trimesh
            ray_origin = trans
            ray_direction = -1 * trans
            ray_direction[0] = 0
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=[ray_origin],
                ray_directions=[ ray_direction],
                multiple_hits=False
            )
            if len(index_tri)>0:
                triangles = faces[index_tri[0]]
                v1, v2, v3 = vertices[triangles]
                alpha = np.linalg.lstsq(np.array([v1, v2, v3]).T, np.asarray(locations).T, rcond=None)[0]
                if np.sum(alpha) > 1+1e-8 or np.sum(alpha) < 1-1e-8 or np.any(alpha < 0):
                    continue
                tri_indexes.append(index_tri[0])
                alphas.append(alpha[:,0])
                verts.append([v1, v2, v3])
                points.append(locations[0])

    return np.asarray(verts), np.asarray(alphas), np.asarray(points), np.asarray(tri_indexes)   


def ray_mesh_intersection_shaft(mesh_path):
    """
    Finds the intersection points and intersecting triangle faces of a ray with a mesh.

    Args:
        mesh_path (str): Path to the mesh file (OBJ format).
        ray_origin (np.ndarray): Origin of the ray (3,).
        ray_direction (np.ndarray): Direction of the ray (3,).

    Returns:
        np.ndarray: Intersection points (N, 3).
        np.ndarray: Indices of the intersecting triangle faces (N,).
    """
    # Load the mesh
    mesh = trimesh.load_mesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([k for k in mesh.geometry.values()])
    mesh.vertices = transform_vertices_function(mesh.vertices)

    min_x = mesh.vertices[:, 0].min()
    max_x = mesh.vertices[:, 0].max()
    faces = mesh.faces
    vertices = mesh.vertices
    step_x = int((max_x + 0.00305) / 5e-4)
    xx = np.linspace(-0.00305, max_x, step_x)
    step_y = int(0.004163 * 2 / 5e-4)
    yy = np.linspace(-0.004163, 0.004163, step_y)
    step_z = int(0.004145 * 2 / 5e-4)
    zz = np.linspace(-0.004145, 0.004145, step_z)
    face_x_z = np.meshgrid(xx, zz)
    face_y_z = np.meshgrid(yy, zz)
    face_x_y = np.meshgrid(xx, yy)
    # x-z plane; y < 0
    ray_origins1 = np.array([face_x_z[0].flatten(), np.zeros_like(face_x_z[0].flatten()), face_x_z[1].flatten()]).T
    ray_directions1 = np.zeros_like(ray_origins1)
    ray_directions1[:, 1] = -1
    # x-z plane; y > 0
    ray_directions2 = np.zeros_like(ray_origins1)
    ray_directions2[:, 1] = 1
    # y-z plane; x < 0
    ray_origins3 = np.array([np.ones_like(face_y_z[0].flatten()), face_y_z[0].flatten(), face_y_z[1].flatten()]).T
    ray_directions3 = np.zeros_like(ray_origins3)
    ray_directions3[:, 0] = -1
    # x-y plane; z < 0
    ray_origins4 = np.array([face_x_y[0].flatten(), face_x_y[1].flatten(), np.ones_like(face_x_y[0].flatten())]).T
    ray_directions4 = np.zeros_like(ray_origins4)
    ray_directions4[:, 2] = -1
    # x-y plane; z > 0
    ray_origins5 = np.array([face_x_y[0].flatten(), face_x_y[1].flatten(), -np.ones_like(face_x_y[0].flatten())]).T
    ray_directions5 = np.zeros_like(ray_origins5)
    ray_directions5[:, 2] = 1

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=np.concatenate([ray_origins1, ray_origins1, ray_origins3,ray_origins4,ray_origins5], axis=0),
        ray_directions=np.concatenate([ray_directions1, ray_directions2,ray_directions3,ray_directions4,ray_directions5], axis=0),
        multiple_hits=False
    )
    triangles = faces[index_tri]
    hitting_verts = vertices[triangles]
    hitting_verts = hitting_verts.transpose(0, 2, 1)
    
    tri_indexes = []
    alphas = []
    verts = []
    points = []
    for i, vert, location in zip(index_tri, hitting_verts, locations):
        alpha = np.linalg.lstsq(vert, location[:, None], rcond=None)[0]
        if np.sum(alpha) > 1+1e-8 or np.sum(alpha) < 1-1e-8 or np.any(alpha < 0):
            continue
        tri_indexes.append(i)
        alphas.append(alpha[:,0])
        verts.append(vert.T)
        points.append(location)
    

    radius = 0.1
    min_x = mesh.vertices[:, 0].min()
    max_x = mesh.vertices[:, 0].max()
    rotations = []
    translations = []
    
    faces = mesh.faces
    vertices = mesh.vertices

    step = int((max_x - min_x)/1e-3)
    for t in tqdm.tqdm(np.linspace(0, 2 * np.pi, 120)):
        for x_trans in np.linspace(min_x, -0.003, step):
            if x_trans < max_x - (max_x - min_x) * 2 / 3:
                continue
            rotation = np.array([[1,0,0],[0, np.cos(t), np.sin(t)],[0, -np.sin(t), np.cos(t)]], np.float32).T
            trans = np.array([x_trans, radius * np.cos(t), radius * np.sin(t)], np.float32)
            rotations.append(rotation)
            translations.append(trans)

            # Perform ray-mesh intersection
            mesh: trimesh.Trimesh
            ray_origin = trans
            ray_direction = -1 * trans
            ray_direction[0] = 0
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=[ray_origin],
                ray_directions=[ ray_direction],
                multiple_hits=False
            )
            if len(index_tri)>0:
                triangles = faces[index_tri[0]]
                v1, v2, v3 = vertices[triangles]
                alpha = np.linalg.lstsq(np.array([v1, v2, v3]).T, np.asarray(locations).T, rcond=None)[0]
                if np.sum(alpha) > 1+1e-8 or np.sum(alpha) < 1-1e-8 or np.any(alpha < 0):
                    continue
                tri_indexes.append(index_tri[0])
                alphas.append(alpha[:,0])
                verts.append([v1, v2, v3])
                points.append(locations[0])
    return np.asarray(verts), np.asarray(alphas), np.asarray(points), np.asarray(tri_indexes)
# Example usage
mesh_path = 'instrument_mesh/transformed_shaft.obj'
verts, alphas,points,tri_indexes = ray_mesh_intersection_shaft(mesh_path)

output_file = '/mnt/iMVR/shuojue/data/instrument_dataset_LND/shaft_ray_tracing_sparse.npz'

surf_alpha = alphas
surf_face_idxes = tri_indexes
surf_vertices = verts
surface_initialization = {'surf_face_idxes': surf_face_idxes, 'surf_vertices': surf_vertices, 'surf_alpha': surf_alpha}
np.savez(output_file, **surface_initialization)

