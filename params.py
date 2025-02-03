from pathlib import Path

from pipe_utils.def_params import DefaultParams


class Params(DefaultParams):
    # ===== base
    input_dir_path: Path | None = None
    color_wh: tuple[int, int] = (1920, 1440)
    depth_wh: tuple[int, int] = (256, 192)
    debug_break_after_num_frames: int | None = None
    debug_is_show_raw_video: bool = False
    depth_fps: float = 30
    # ===== pointcloud
    confidence_min_th: int = 1
    valid_depth_min_max_m: tuple[float, float] = (0.1, 3)
    # ==== mesh debug
    tri_max_viewing_angle_th_deg: float = 80
    # ====== arkit jumps
    min_arkit_jump_dist_m_to_negate: float = 0.1
    is_negate_arkit_translation_jumps_only: bool = True


params = Params()
