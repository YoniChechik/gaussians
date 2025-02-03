import os

import cv2
import numpy as np

from mesh import Mesh
from transformation_utils import apply_rigid_motion_transformation


def write_mesh_with_texture(
    mesh: Mesh,
    bgr_im: np.ndarray,
    file_stem,
    save_dir_path,
    mesh_texture_subsampling_factor=1,
):
    os.makedirs(save_dir_path, exist_ok=True)
    if mesh_texture_subsampling_factor != 1:
        bgr_im_shape = bgr_im.shape
        vis_texture_wh = (
            bgr_im_shape[1] // mesh_texture_subsampling_factor,
            bgr_im_shape[0] // mesh_texture_subsampling_factor,
        )
        bgr_im = cv2.resize(bgr_im, vis_texture_wh, interpolation=cv2.INTER_AREA)

    # Save the texture image
    texture_file = f"{file_stem}.jpg"
    texture_path = os.path.join(save_dir_path, texture_file)
    cv2.imwrite(texture_path, bgr_im)

    # Write MTL file
    mtl_content = (
        "newmtl material_0\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "Ks 0.000 0.000 0.000\n"
        "d 1.0\n"
        "illum 2\n"
        f"map_Kd {texture_file}"
    )

    mtl_path = os.path.join(save_dir_path, f"{file_stem}.mtl")
    with open(mtl_path, "w") as mtl_file:
        mtl_file.write(mtl_content)

    _save_obj_file(mesh, save_dir_path, file_stem)


def _save_obj_file(mesh: Mesh, output_dir, file_stem):
    """optimized way to save this file"""
    obj_path = os.path.join(output_dir, f"{file_stem}.obj")

    # Precompute the lines for the OBJ file
    lines = [f"mtllib {file_stem}.mtl\n"]
    lines += [f"v {vert[0]:.3f} {vert[1]:.3f} {vert[2]:.3f}\n" for vert in mesh.vertices]
    lines += [f"vt {uv[0]:.6f} {uv[1]:.6f}\n" for uv in mesh.vertex_texture_uvs]
    lines += [f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n" for face in mesh.faces + 1]

    # Write all lines to the file at once
    with open(obj_path, "w") as f:
        f.writelines(lines)


def write_ply_pointcloud(filepath, points, rgb=(255, 0, 0)):
    """
    Write points to a PLY file, with all points set to the given color.

    Args:
        filepath (str): The path to the PLY file to create.
        points (np.ndarray): The Nx3 array of point coordinates.
        color (tuple): The RGB color of the points as a tuple (R, G, B).
    """
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    lines = [f"{point[0]} {point[1]} {point[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n" for point in points]
    with open(filepath, "w") as f:
        f.write(header)
        f.writelines(lines)


def write_ply_track_points(filename, points, start_color=(255, 0, 0), end_color=(0, 0, 255)):
    """
    Write points to a PLY file, with a color gradient from start_color to end_color.

    Args:
        filename (str): The path to the PLY file to create.
        points (np.ndarray): The Nx3 array of point coordinates.
        start_color (tuple): The RGB start color of the points as a tuple (R, G, B).
        end_color (tuple): The RGB end color of the points as a tuple (R, G, B).
    """
    num_points = len(points)
    edges = [(i, i + 1) for i in range(num_points - 1)]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element edge {len(edges)}\n"
        "property int vertex1\n"
        "property int vertex2\n"
        "end_header\n"
    )

    point_lines = []
    for i, point in enumerate(points):
        fraction = i / (num_points - 1)
        # Interpolate color directly within the loop
        interpolated_color = tuple(int(start_color[j] + (end_color[j] - start_color[j]) * fraction) for j in range(3))
        point_line = f"{point[0]} {point[1]} {point[2]} {interpolated_color[0]} {interpolated_color[1]} {interpolated_color[2]}\n"
        point_lines.append(point_line)

    edge_lines = [f"{edge[0]} {edge[1]}\n" for edge in edges]

    with open(filename, "w") as f:
        f.write(header)
        f.writelines(point_lines)
        f.writelines(edge_lines)


def write_camera_frustum_obj(file_name, cam_to_world_transform_matrix, far_distance=0.3, fov_degrees=35):
    # Calculate the size of the far plane
    far_size = 2 * far_distance * np.tan(np.radians(fov_degrees) / 2)

    # Define vertices of the pyramid in camera space
    vertices = np.array(
        [
            [0, 0, 0],  # Camera origin
            [-far_size / 2, -far_size / 2, far_distance],  # Far plane, bottom left
            [far_size / 2, -far_size / 2, far_distance],  # Far plane, bottom right
            [far_size / 2, far_size / 2, far_distance],  # Far plane, top right
            [-far_size / 2, far_size / 2, far_distance],  # Far plane, top left
        ]
    )

    # Apply the camera to world transformation
    xyz = apply_rigid_motion_transformation(vertices, cam_to_world_transform_matrix)

    # Write to OBJ file
    lines = []
    # Write vertices
    lines += [f"v {v[0]} {v[1]} {v[2]}\n" for v in xyz]

    # Write edges
    edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (4, 5), (5, 2)]
    lines += [f"l {start} {end}\n" for start, end in edges]

    with open(file_name, "w") as f:
        f.writelines(lines)
