import numpy as np

from io_3d import write_ply_track_points
from params import params


# TODO rerun sdk
def debug_results(
    raw_camera_poses: np.ndarray, camera_poses: np.ndarray, camera_to_world_transformations_with_sw_fixes: np.ndarray
):
    write_ply_track_points(
        params.save_dir_path / "track_raw.ply",
        raw_camera_poses,
        start_color=(255, 0, 0),
        end_color=(0, 0, 255),
    )
    write_ply_track_points(
        params.save_dir_path / "track_algo.ply",
        camera_poses,
        start_color=(0, 255, 0),
        end_color=(255, 0, 255),
    )
    sw_camera_poses = camera_to_world_transformations_with_sw_fixes[:, :3, 3]
    write_ply_track_points(
        params.save_dir_path / "track_sw.ply",
        sw_camera_poses,
        start_color=(0, 0, 0),
        end_color=(255, 255, 255),
    )
