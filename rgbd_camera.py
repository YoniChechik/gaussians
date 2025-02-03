import numpy as np

from pcd_generator import PointCloud


def color_intrinsics_to_depth_intrinsics(
    color_intrinsic_matrices: np.ndarray, color_to_depth_scale_factor: float
) -> np.ndarray:
    # Create depth intrinsic matrices based on the color intrinsic matrices
    # according to this the depth and color are aligned up to scale
    # https://developer.apple.com/forums/thread/663995
    depth_intrinsics_matrices = color_intrinsic_matrices.copy()
    depth_intrinsics_matrices[:, :2, :3] /= color_to_depth_scale_factor  # Apply scale factor to all but the last row

    return depth_intrinsics_matrices


def uv_and_pointcloud_to_xyz(pcd: PointCloud, input_uvs: np.ndarray, color_wh: list[int]) -> np.ndarray:
    depth_wh = pcd.valid_depth_mask.shape[::-1]
    uvs_rescaled_to_depth = input_uvs / np.asarray(color_wh) * depth_wh

    xyz_coords = np.zeros((input_uvs.shape[0], 3))

    for i, uv in enumerate(uvs_rescaled_to_depth):
        floor_uv = uv.astype(int)
        # check if out of bounds
        if np.any(floor_uv + [1, 1] >= depth_wh):
            return None

        uv_bbox = np.array(
            [
                floor_uv,
                floor_uv + [0, 1],  # Right
                floor_uv + [1, 1],  # Down Right
                floor_uv + [1, 0],  # Down
            ]
        )

        valid_bbox_mask = pcd.valid_depth_mask[uv_bbox[:, 1], uv_bbox[:, 0]]

        # check if not enough valid points in bbox
        if np.sum(valid_bbox_mask) <= 2:
            return None

        uv_bbox_dist = np.linalg.norm(uv_bbox[valid_bbox_mask] - uv, axis=1)
        # get closest 3
        closest_three_indices = np.argsort(uv_bbox_dist)[:3]
        closest_three_depth_uv = uv_bbox[closest_three_indices]

        # Calculate barycentric coordinates for a point with UV coordinates (u, v) in relation to a triangle defined by vertices with UVs (u1, v1), (u2, v2), (u3, v3).
        # This involves solving the linear system:
        # [u1, u2, u3]   [lambda1]   [u]
        # [v1, v2, v3] * [lambda2] = [v]
        # [ 1,  1,  1]   [lambda3]   [1]
        # The solution gives the barycentric coordinates (lambda1, lambda2, lambda3), representing the point's relative position within the triangle.
        A = np.c_[closest_three_depth_uv, np.ones((3, 1))].T  # Append a column of ones for affine solution
        b = np.array([uv[0], uv[1], 1])  # The input UVs plus an extra 1 for affine solution
        bary_coords = np.linalg.lstsq(A, b, rcond=None)[0]
        assert np.isclose(np.sum(bary_coords), 1), "Sum of barycentric coordinates should be close to 1"

        # check if inside the tri. A problem can happen if we had just 3 valid points and it outside this tri (other bbox half)
        if np.any((bary_coords < 0) | (bary_coords > 1)):
            return None

        # Use the barycentric coordinates to interpolate the 3D positions
        indices = closest_three_depth_uv[:, 1] * depth_wh[0] + closest_three_depth_uv[:, 0]

        # Fetch the XYZ coordinates for these indices
        xyz_verts = pcd.all_xyz[indices]
        xyz_coords[i] = bary_coords.reshape(1, 3) @ xyz_verts  # Interpolate using barycentric coords

    return xyz_coords
