import numpy as np

from params import params
from transformation_utils import apply_rigid_motion_transformation


def base_depth_meshgrid():
    ind_j, ind_i = np.meshgrid(
        np.arange(0, params.depth_wh[0]),
        np.arange(0, params.depth_wh[1]),
    )

    return ind_j, ind_i


class PointCloud:
    def __init__(self, xyz: np.ndarray, valid_depth_mask: np.ndarray, camera_pos: np.ndarray) -> None:
        self.all_xyz = xyz
        self.valid_depth_mask = valid_depth_mask
        self.camera_pos = camera_pos


class _PointCloudGenerator:
    def __init__(self) -> None:
        ind_j, ind_i = base_depth_meshgrid()

        self._base_vertices = np.column_stack(
            [ind_j.reshape(-1, 1), ind_i.reshape(-1, 1), np.zeros((ind_j.shape[0] * ind_j.shape[1], 1), dtype=int)]
        )

    def generate(self, depth_frame, confidence_frame, camera_to_world_transformation, depth_intrinsics_matrix):
        xyz = self._depth_to_xyz(depth_frame, camera_to_world_transformation, depth_intrinsics_matrix)

        valid_depth_mask = _get_valid_depth_mask(depth_frame, confidence_frame)

        camera_pos = camera_to_world_transformation[:3, 3]

        return PointCloud(xyz, valid_depth_mask, camera_pos)

    def _depth_to_xyz(self, depth: np.ndarray, camera_to_world_transformation, depth_intrinsics):
        z = depth.reshape(-1)
        x = (self._base_vertices[:, 0] - depth_intrinsics[0][2]) / depth_intrinsics[0][0] * z
        y = (self._base_vertices[:, 1] - depth_intrinsics[1][2]) / depth_intrinsics[1][1] * z

        xyz_cam = np.stack((x, y, z), axis=-1)

        xyz = apply_rigid_motion_transformation(xyz_cam, camera_to_world_transformation)
        return xyz


def _get_valid_depth_mask(depth, confidence):
    valid_depth_mask = np.ones_like(depth, dtype=bool)
    valid_depth_mask[confidence < params.confidence_min_th] = False
    valid_depth_mask[(depth < params.valid_depth_min_max_m[0]) | (depth > params.valid_depth_min_max_m[1])] = False
    return valid_depth_mask


pcd_generator = _PointCloudGenerator()
