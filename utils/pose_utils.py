
import torch
import torch.nn.functional as F
import sys
from datetime import datetime
import numpy as np
import cv2
import pyrender
import trimesh
# from utils.instrument import Instrument

# Foundation Pose Estimation
def read_pose_from_txt(file_path):
    """
    Reads a pose from a text file and returns it as a 4x4 transformation matrix.

    Args:
        file_path (str): Path to the text file containing the pose.

    Returns:
        torch.Tensor: 4x4 transformation matrix.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Initialize an empty list to store the matrix rows
    matrix = []
    
    # Read each line and convert it to a list of floats
    for line in lines:
        row = list(map(float, line.strip().split()))
        matrix.append(row)
    
    # Convert the list of lists to a torch tensor
    pose_matrix = torch.tensor(matrix, dtype=torch.float32)
    
    return pose_matrix

def solve_pnp_ransac(object_points, image_points, camera_matrix, dist_coeffs=None, reprojection_error=8.0, confidence=0.99, iterations_count=100):
        """
        Solves the Perspective-n-Point problem using the RANSAC algorithm to find the pose of a camera.

        Args:
            object_points (Union[torch.Tensor, np.ndarray]): 3D points in the object coordinate space (Nx3).
            image_points (Union[torch.Tensor, np.ndarray]): 2D points in the image plane (Nx2).
            camera_matrix (Union[torch.Tensor, np.ndarray]): Camera intrinsic matrix (3x3).
            dist_coeffs (Union[torch.Tensor, np.ndarray], optional): Distortion coefficients (5x1). Defaults to None.
            reprojection_error (float, optional): Maximum allowed reprojection error to classify as inlier. Defaults to 8.0.
            confidence (float, optional): Confidence level. Defaults to 0.99.
            iterations_count (int, optional): Number of iterations. Defaults to 100.

        Returns:
            torch.Tensor: Rotation vector (3x1).
            torch.Tensor: Translation vector (3x1).
            np.ndarray: Inlier indices.
        """
        # Convert inputs to numpy arrays if they are torch tensors
        if isinstance(object_points, torch.Tensor):
            object_points = object_points.cpu().numpy()
        if isinstance(image_points, torch.Tensor):
            image_points = image_points.cpu().numpy()
        if isinstance(camera_matrix, torch.Tensor):
            camera_matrix = camera_matrix.cpu().numpy()
        if dist_coeffs is not None and isinstance(dist_coeffs, torch.Tensor):
            dist_coeffs = dist_coeffs.cpu().numpy()

        # Solve PnP using RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs, reprojectionError=reprojection_error, confidence=confidence, iterationsCount=iterations_count)

        if not success:
            raise ValueError("PnP RANSAC solution could not be found")

        # Convert the rotation and translation vectors to torch tensors
        rvec = torch.tensor(rvec, dtype=torch.float32)
        tvec = torch.tensor(tvec, dtype=torch.float32)

        return rvec, tvec, inliers

def solve_p3p(object_points, image_points, camera_matrix, dist_coeffs=None):
    """
    Solves the Perspective-Three-Point (P3P) problem to find the pose of a camera.

    Args:
        object_points (Union[torch.Tensor, np.ndarray]): 3D points in the object coordinate space (3x3).
        image_points (Union[torch.Tensor, np.ndarray]): 2D points in the image plane (3x2).
        camera_matrix (Union[torch.Tensor, np.ndarray]): Camera intrinsic matrix (3x3).
        dist_coeffs (Union[torch.Tensor, np.ndarray], optional): Distortion coefficients (5x1). Defaults to None.

    Returns:
        List[Tuple[torch.Tensor, torch.Tensor]]: List of possible solutions, each containing a rotation vector (3x1) and a translation vector (3x1).
    """
    # Convert inputs to numpy arrays if they are torch tensors
    if isinstance(object_points, torch.Tensor):
        object_points = object_points.cpu().numpy()
    if isinstance(image_points, torch.Tensor):
        image_points = image_points.cpu().numpy()
    if isinstance(camera_matrix, torch.Tensor):
        camera_matrix = camera_matrix.cpu().numpy()
    if dist_coeffs is not None and isinstance(dist_coeffs, torch.Tensor):
        dist_coeffs = dist_coeffs.cpu().numpy()

    # Solve P3P
    success, rvecs, tvecs = cv2.solveP3P(object_points, image_points, camera_matrix, dist_coeffs, cv2.SOLVEPNP_P3P)

    if not success:
        raise ValueError("P3P solution could not be found")

    # Convert the rotation and translation vectors to torch tensors
    solutions = []
    for rvec, tvec in zip(rvecs, tvecs):
        rvec = torch.tensor(rvec, dtype=torch.float32)
        tvec = torch.tensor(tvec, dtype=torch.float32)
        solutions.append((rvec, tvec))

    return solutions[0]

def solve_pnp(object_points, image_points, camera_matrix, dist_coeffs=None):
    """
    Solves the Perspective-n-Point problem to find the pose of a camera.

    Args:
        object_points (Union[torch.Tensor, np.ndarray]): 3D points in the object coordinate space (Nx3).
        image_points (Union[torch.Tensor, np.ndarray]): 2D points in the image plane (Nx2).
        camera_matrix (Union[torch.Tensor, np.ndarray]): Camera intrinsic matrix (3x3).
        dist_coeffs (Union[torch.Tensor, np.ndarray], optional): Distortion coefficients (5x1). Defaults to None.

    Returns:
        torch.Tensor: Rotation vector (3x1).
        torch.Tensor: Translation vector (3x1).
    """
    # Convert inputs to numpy arrays if they are torch tensors
    if isinstance(object_points, torch.Tensor):
        object_points = object_points.cpu().numpy()
    if isinstance(image_points, torch.Tensor):
        image_points = image_points.cpu().numpy()
    if isinstance(camera_matrix, torch.Tensor):
        camera_matrix = camera_matrix.cpu().numpy()
    if dist_coeffs is not None and isinstance(dist_coeffs, torch.Tensor):
        dist_coeffs = dist_coeffs.cpu().numpy()

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    if not success:
        raise ValueError("PnP solution could not be found")

    # Convert the rotation and translation vectors to torch tensors
    rvec = torch.tensor(rvec, dtype=torch.float32)
    tvec = torch.tensor(tvec, dtype=torch.float32)

    return rvec, tvec

def project_points(object_points, rvec, tvec, camera_matrix, dist_coeffs=None):
    """
    Projects 3D points to the image plane using the given camera parameters.
    """
    # Convert inputs to numpy arrays if they are torch tensors
    if isinstance(object_points, torch.Tensor):
        object_points = object_points.cpu().numpy()
    if isinstance(rvec, torch.Tensor):
        rvec = rvec.cpu().numpy()
    if isinstance(tvec, torch.Tensor):
        tvec = tvec.cpu().numpy()
    if isinstance(camera_matrix, torch.Tensor):
        camera_matrix = camera_matrix.cpu().numpy()
    if dist_coeffs is not None and isinstance(dist_coeffs, torch.Tensor):
        dist_coeffs = dist_coeffs.cpu().numpy()

    # Project the points
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)

    # # Convert the image points to a torch tensor
    # image_points = torch.tensor(image_points.squeeze(), dtype=torch.float32)

    return image_points

def convert_w2c_to_c2w(rvec, tvec):
    """
    Convert rotation and translation vectors from world-to-camera (w2c) to camera-to-world (c2w).

    Args:
        rvec (torch.Tensor): Rotation vector (3x1) in w2c format.
        tvec (torch.Tensor): Translation vector (3x1) in w2c format.

    Returns:
        torch.Tensor: Rotation vector (3x1) in axis-angle representation in c2w format.
        torch.Tensor: Translation vector (3x1) in c2w format.
    """
    # Convert torch tensors to numpy arrays
    rvec_np = rvec.cpu().numpy().flatten()
    tvec_np = tvec.cpu().numpy().flatten()

    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec_np)

    # Compute the inverse transformation (c2w)
    R_inv = R.T
    t_inv = -R_inv @ tvec_np

    # Convert the inverse rotation matrix back to axis-angle representation
    rvec_inv, _ = cv2.Rodrigues(R_inv)

    # Convert back to torch tensors
    rvec_inv = torch.tensor(rvec_inv.flatten(), dtype=torch.float32)
    t_inv = torch.tensor(t_inv, dtype=torch.float32)

    return rvec_inv, t_inv

def render_mesh(mesh, rvec, tvec, camera_matrix, image_size=(1280, 1024)):
    """
    Renders the RGB, silhouette, and depth images of a mesh given the pose and camera parameters.

    Args:
        mesh (trimesh.Trimesh): The mesh to render.
        rvec (torch.Tensor): Rotation vector (3x1).
        tvec (torch.Tensor): Translation vector (3x1).
        camera_matrix (torch.Tensor): Camera intrinsic matrix (3x3).
        image_size (tuple): Size of the output images (width, height).

    Returns:
        np.ndarray: RGB image.
        np.ndarray: Silhouette image.
        np.ndarray: Depth image.
    """
    # Convert torch tensors to numpy arrays
    rvec = rvec.cpu().numpy().flatten()
    tvec = tvec.cpu().numpy().flatten()
    camera_matrix = camera_matrix.cpu().numpy()

    # Create a scene
    scene = pyrender.Scene()

    # Add the mesh to the scene
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)

    # Create a camera
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

    # Create a camera pose
    R, _ = cv2.Rodrigues(rvec)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = R
    camera_pose[:3, 3] = tvec

    camera_pose[:, 1:3] *= -1
    # Add the camera to the scene
    scene.add(camera, pose=camera_pose)

    # Create a light source
    light = pyrender.PointLight(intensity = 1)
    camera_pose[:, 1:3] *= -1
    light_pose = camera_pose.copy()
    light_pose[2, 3] += 0.1
    light_pose[0, 3] += 0.1
    scene.add(light, pose=light_pose)

    # Create an offscreen renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=image_size[0], viewport_height=image_size[1])

    # Render the scene
    color, depth = renderer.render(scene)

    # Create the silhouette image
    silhouette = (depth > 0).astype(np.uint8) * 255

    return color, silhouette, depth

