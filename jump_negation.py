import numpy as np
from loguru import logger

from params import params
from transformation_utils import rotation_angle_degrees


class JumpNegation:
    def __init__(self) -> None:
        self._prev_global_transformation = None
        self._frame_ind = -1

    def step(self, curr_global_transformation: np.ndarray):
        self._frame_ind += 1
        # first frame is never a jump
        if self._prev_global_transformation is None:
            self._prev_global_transformation = curr_global_transformation.copy()

        # calc relative
        relative_transformation = np.linalg.inv(self._prev_global_transformation) @ curr_global_transformation
        jump_translation_m = np.linalg.norm(relative_transformation[:3, 3])
        if jump_translation_m <= params.min_arkit_jump_dist_m_to_negate:
            return

        logger.debug(
            f"==== negate arkit jump: frame index: {self._frame_ind}; "
            f"deg change={rotation_angle_degrees(relative_transformation[:3, :3]):.2f}; "
            f"m change={jump_translation_m:.2f}"
        )
        if params.is_negate_arkit_translation_jumps_only:
            arkit_jump_fix_transformation = np.eye(4)
            arkit_jump_fix_transformation[:3, 3] = (
                self._prev_global_transformation[:3, 3] - curr_global_transformation[:3, 3]
            )
        else:
            # NOTE this is not simply the inverse of the relative T{i->i+1}, because we are multiplying from left and not from right
            arkit_jump_fix_transformation = np.eye(4)
            R = self._prev_global_transformation[:3, :3] @ curr_global_transformation[:3, :3].T
            t = self._prev_global_transformation[:3, 3] - R @ curr_global_transformation[:3, 3]
            arkit_jump_fix_transformation[:3, :3] = R
            arkit_jump_fix_transformation[:3, 3] = t

        return arkit_jump_fix_transformation

    def set_previous_global_transformation(self, prev_global_transformation: np.ndarray):
        self._prev_global_transformation = prev_global_transformation.copy()
