import torch
import torch.nn.functional as F
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
from datetime import datetime
import numpy as np
import cv2 
import trimesh
# from utils.instrument import Instrument
from utils.pose_utils import solve_pnp_ransac, solve_p3p, render_mesh, convert_w2c_to_c2w
from matplotlib import pyplot as plt

# Example usage
shaft_mesh = trimesh.load_mesh("/mnt/iMVR/shuojue/code/gaussian-mesh-splatting/instrument_mesh/transformed_shaft.obj", force='mesh')
wrist_mesh = trimesh.load_mesh("/mnt/iMVR/shuojue/code/gaussian-mesh-splatting/instrument_mesh/transformed_wrist.obj", force='mesh')
gripper_mesh_left = trimesh.load_mesh("/mnt/iMVR/shuojue/code/gaussian-mesh-splatting/instrument_mesh/transformed_gripper_left.obj", force='mesh')
gripper_mesh_right = trimesh.load_mesh("/mnt/iMVR/shuojue/code/gaussian-mesh-splatting/instrument_mesh/transformed_gripper_right.obj", force='mesh')
# instrument = Instrument(shaft_mesh, wrist_mesh, gripper_mesh_left, gripper_mesh_right)

"""
get the pose for endovis 2017 dataset 
"""
def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 2] = -vertices[:, 2]
    vertices *= c
    return vertices
points_2d = torch.tensor([[332.5, 294],
                         [279-2,312-2],
                         [270.6, 257.2],
                         [223.2, 244.7],
                         [322.8, 288.4],
                         [315.5, 312]], dtype=torch.float32)
points_2d = torch.tensor([[358, 212],
                            [301, 235],
                            [323, 300],
                            [301, 305],
                            [342, 211],
                            [342, 224]], dtype=torch.float32)
points_2d = torch.tensor([(321, 291),
                          (389, 293),
                          (379, 333),
                          (415, 348),
                          (336, 297),
                          (347, 282)], dtype=torch.float32)
points_2d = torch.tensor([(281, 294),
(341, 311),
(325, 347),
(360, 373),
(290, 303),
(305, 293)], dtype=torch.float32)
points_2d = torch.tensor([(206, 215),
(124, 286),
(113, 276),
(49, 333),
(182, 227),
(185, 248)], dtype=torch.float32)
points_3d = torch.tensor([[0.009, 0, 0.002344],
                         [0.002693, -0.000254,0.002975],
                         [0.003658, 0.003073, 0.001473],
                         [0, 0.003988, 0],
                         [0.007619, 0.00056, 0.002828],
                         [0.006383, -0.000722, 0.003056]], dtype=torch.float32)
points_3d = torch.tensor([[0.009, 0, 0.002344],
                         [0.002693, -0.000254,0.002975],
                         [0.003658, -0.003073, -0.001473],
                         [0, -0.00414, 0],
                         [0.007619, 0.00056, 0.002828],
                         [0.006383, -0.000722, 0.003056]], dtype=torch.float32)
points_3d_vis = points_3d[[0, 1, 2, 3, 4, 5],...]
# points_3d = points_3d[[0, 1, 3],...]
# points_2d = points_2d[[0, 1, 3],...]

camera_matrix = torch.tensor([[587.544  ,   0.     , 316.528  ], 
                            [  0.     , 587.544  , 257.41156],
                            [  0.     ,   0.     ,   1.     ]], dtype=torch.float32)
camera_matrix = torch.tensor([ [826.678097642519, 0.0, 277.646325317],
        [0.0, 938.0861301932741, 250.5282],
        [0.0, 0.0, 1.0]], dtype=torch.float32) # surgpose_4/5
camera_matrix = torch.tensor([
    [831.256,   0.000, 281.824],
        [  0.000, 943.905, 253.748],
        [  0.000,   0.000,   1.000]], dtype=torch.float32) # surgpose_3
camera_matrix = torch.tensor([[571.2951 ,   0.     , 314.80426],
       [  0.     , 571.2951 , 267.96167],
       [  0.     ,   0.     ,   1.     ]], dtype=torch.float32) # endovis 2018
# Solve PnP
rvec, tvec,_ = solve_pnp_ransac(points_3d, points_2d, camera_matrix) # w2c 
# transform rvec and tvec to SE3 matrix
rvec_matrix = cv2.Rodrigues(rvec.numpy())[0]
print(rvec_matrix)
print(tvec)
mesh = wrist_mesh
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate([k for k in mesh.geometry.values()])
vertices = transform_vertices_function(torch.tensor(mesh.vertices))
# update the vertices in mesh
mesh.vertices = vertices.numpy()
# R = torch.tensor([
#     [0.6805, 0.0490, 0.7311],
#     [-0.0623, -0.9903, 0.1244],
#     [0.7301, -0.1302, -0.6708]
# ])
# t = torch.tensor([[-0.0067],
#         [ 0.0031],
#         [ 0.0404]])
# T = torch.tensor([
#     [ 0.6805,  0.0490,  0.7311, -0.0067],
#     [-0.0623, -0.9903,  0.1244,  0.0031],
#     [ 0.7301, -0.1302, -0.6708,  0.0404],
#     [ 0.0000,  0.0000,  0.0000,  1.0000]
# ])

rvec_c2w, tvec_c2w = convert_w2c_to_c2w(rvec, tvec)
# Project 3D points to 2D using the camera matrix, rvec, and tvec
def project_points(points_3d, rvec, tvec, camera_matrix):
    # Convert rvec and tvec to numpy arrays
    rvec_np = rvec.numpy()
    tvec_np = tvec.numpy()
    points_3d_np = points_3d.numpy()

    # Project points
    points_2d, _ = cv2.projectPoints(points_3d_np, rvec_np, tvec_np, camera_matrix.numpy(), distCoeffs=None)
    return points_2d.squeeze()

projected_points_2d = project_points(points_3d_vis, rvec, tvec, camera_matrix)

# Visualize the projected points on the image
colors = ['r', 'g', 'b', 'c', 'm', 'y']
for i, point in enumerate(projected_points_2d):
    plt.scatter(point[0], point[1], color=colors[i], s=8)
    
rgb_image = cv2.imread("/mnt/iMVR/shuojue/data/instrument_dataset_LND/color/frame066.png")[...,::-1]
rgb_image = cv2.imread("/mnt/iMVR/shuojue/data/endovins18/color/frame008.png")[...,::-1]
plt.imshow(rgb_image)
plt.title("Projected 3D Points on RGB Image")
plt.savefig("projected_points_image.png")


rgb_image, silhouette_image, depth_image = render_mesh(mesh, rvec_c2w, tvec_c2w, camera_matrix, image_size=(640, 512))
# Display the images
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(rgb_image)
plt.title("RGB Image")
plt.subplot(1, 3, 2)
plt.imshow(silhouette_image, cmap='gray')
plt.title("Silhouette Image")
plt.subplot(1, 3, 3)
plt.imshow(depth_image, cmap='gray')
plt.title("Depth Image")
plt.savefig("output_image.png")



#============================Get the pose for SurgPose dataset============================

# points_2d = torch.tensor([332, 297],
#                          [279,312],
#                          [270.6, 257.2],
#                          [223.2, 244.7],
#                          [323.8, 290.4],
#                          [315.5, 312], dtype=torch.float32)
# points_3d = torch.tensor([0.009, 0, 0.002344],
#                          [0.002693, -0.000254,0.002975],
#                          [0.003658, 0.003073, 0.001473],
#                          [0, 0.003988, 0],
#                          [0.007619, 0.00056, 0.002828],
#                          [0.006383, -0.000722, 0.003056], dtype=torch.float32)

# camera_matrix = torch.tensor([[587.544  ,   0.     , 316.528  ],
#                             [  0.     , 587.544  , 257.41156],
#                             [  0.     ,   0.     ,   1.     ]], dtype=torch.float32)
# # Solve PnP
# rvec, tvec, inliers = solve_pnp_ransac(points_3d, points_2d, camera_matrix)


# mesh = trimesh.load_mesh('path/to/your/mesh.obj')
# # rvec = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
# # tvec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
# camera_matrix = torch.tensor([
#     [800.0, 0.0, 320.0],
#     [0.0, 800.0, 240.0],
#     [0.0, 0.0, 1.0]
# ], dtype=torch.float32)

# rgb_image, silhouette_image, depth_image = render_mesh(mesh, rvec, tvec, camera_matrix)

# # Display the images
# import matplotlib.pyplot as plt

# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(rgb_image)
# plt.title("RGB Image")
# plt.subplot(1, 3, 2)
# plt.imshow(silhouette_image, cmap='gray')
# plt.title("Silhouette Image")
# plt.subplot(1, 3, 3)
# plt.imshow(depth_image, cmap='gray')
# plt.title("Depth Image")
# plt.show()
