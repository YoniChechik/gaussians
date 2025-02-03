from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.transform import Rotation

from data_handling.read_binary_ply_to_dataframe import read_binary_ply_to_dataframe
from params import params
from rgbd_camera import color_intrinsics_to_depth_intrinsics


@dataclass
class ParsedData:
    timestamps: np.ndarray
    color_ind_to_arpose_ind: list[int]
    depth_ind_to_arpose_ind: list[int]
    camera_to_world_transformations: np.ndarray
    depth_intrinsics_matrices_in_color_timetable: np.ndarray


def parse_raw_data() -> ParsedData:
    # === arpose
    arpose_pd = read_binary_ply_to_dataframe(params.input_dir_path / "ARPoses.ply")
    timestamps = arpose_pd["timestamp"].to_numpy()
    camera_to_world_transformations_with_sw_fixes = _parse_transformation_data(arpose_pd)
    # === depth
    depth_pd = read_binary_ply_to_dataframe(params.input_dir_path / "DepthMaps.ply")
    depth_ind_to_arpose_ind = _dest_timestamps_to_src_timestamps(depth_pd, arpose_pd)
    # === color
    color_pd = read_binary_ply_to_dataframe(params.input_dir_path / "Frames.ply")
    color_ind_to_arpose_ind = _dest_timestamps_to_src_timestamps(color_pd, arpose_pd)
    depth_intrinsics_matrices_in_color_timetable = _parse_depth_intrinsics(color_pd, params.color_wh, params.depth_wh)

    camera_to_world_transformations = camera_to_world_transformations_with_sw_fixes.copy()
    return ParsedData(
        timestamps,
        color_ind_to_arpose_ind,
        depth_ind_to_arpose_ind,
        camera_to_world_transformations,
        depth_intrinsics_matrices_in_color_timetable,
    ), camera_to_world_transformations_with_sw_fixes


def _parse_transformation_data(pd: pd.DataFrame, is_arkit_coo_to_opencv_coo=True):
    rot_x_180 = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )
    if not is_arkit_coo_to_opencv_coo:
        rot_x_180 = np.eye(3)

    # Extract translation vectors directly into a NumPy array
    camera_to_world_t = pd[["Loc.x", "Loc.y", "Loc.z"]].to_numpy()

    # Convert quaternions to rotation matrices
    quaternions = pd[["Quat.x", "Quat.y", "Quat.z", "Quat.w"]].to_numpy()
    # NOTE: in iphone when the phone is standing up- back facing cam has y pointing up and -z in cam view direction.
    # we need to flip both to get coordinates that are consistent with opencv. we do it by rotating around x 180 degrees.
    camera_to_world_R = np.array([Rotation.from_quat(q).as_matrix() @ rot_x_180 for q in quaternions])

    # Create the camera-to-world projection matrices
    num_poses = camera_to_world_t.shape[0]
    camera_to_world_projections = np.tile(np.eye(4), (num_poses, 1, 1))  # Initialize with identity matrices
    camera_to_world_projections[:, :3, :3] = camera_to_world_R  # Set rotation parts
    camera_to_world_projections[:, :3, 3] = camera_to_world_t  # Set translation parts

    return camera_to_world_projections


def _parse_depth_intrinsics(color_data: pd.DataFrame, color_wh, depth_wh):
    # Create color intrinsic matrices directly from the DataFrame
    zero_arr = np.zeros(len(color_data))
    color_intrinsic_matrices = np.array(
        [
            [color_data["EFL.x"], zero_arr, color_data["OC.x"]],
            [zero_arr, color_data["EFL.y"], color_data["OC.y"]],
            [zero_arr, zero_arr, zero_arr + 1],
        ]
    ).transpose(2, 0, 1)  # Re-arrange dimensions to list of matrices

    color_to_depth_scale_factor = color_wh[0] / depth_wh[0]
    depth_intrinsics_matrices = color_intrinsics_to_depth_intrinsics(
        color_intrinsic_matrices, color_to_depth_scale_factor
    )
    return depth_intrinsics_matrices


def _dest_timestamps_to_src_timestamps(dest_pd, src_pd):
    src_timestamps = src_pd["timestamp"]
    dest_timestamps = dest_pd["timestamp"]

    best_matches = []

    for timestamp in dest_timestamps:
        t_diffs = np.abs(src_timestamps - timestamp)
        idx = np.argmin(t_diffs)
        best_matches.append(idx)

        best_t_diff_signed = src_timestamps[idx] - timestamp
        if np.abs(best_t_diff_signed) > 0:
            logger.warning(
                f"We have some time-unaligned matched frames. The diff is {best_t_diff_signed:.4f} "
                f"and corresponds to {1 / np.abs(best_t_diff_signed):.4f} FPS (higher than 60 is not important)."
            )
            if 0:
                import matplotlib.pyplot as plt

                plt.plot(src_timestamps[max(0, idx - 5) : min(idx + 5, len(src_timestamps))], "*")
                plt.plot(idx, timestamp, "*")
                plt.show()

    best_matches_arr = np.asarray(best_matches)
    assert np.all(best_matches_arr[1:] - best_matches_arr[:-1] > 0), "The matches are not monotonically increasing"

    return best_matches
