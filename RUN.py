import numpy as np
from loguru import logger
from tqdm import tqdm

from data_handling.online_data import OnlineDataIterator
from data_handling.parse_raw_data import parse_raw_data
from debug_results import debug_results
from jump_negation import JumpNegation
from params import params
from pipe_utils.script_runner import script_runner


@script_runner(params)
def main():
    parsed_data, camera_to_world_transformations_with_sw_fixes = parse_raw_data()

    online_data_iter = OnlineDataIterator(parsed_data)
    jump_negation = JumpNegation()

    raw_camera_poses = []
    camera_poses = []

    global_transformation = np.eye(4)
    correction_transformation = np.eye(4)

    logger.debug("======================== start running main loop")
    for frame_ind, data_block in enumerate(tqdm(online_data_iter)):
        if params.debug_break_after_num_frames is not None and frame_ind > params.debug_break_after_num_frames:
            break

        # === get raw global transformation and fix it
        global_raw_transformation = data_block.camera_to_world_transformation.copy()
        raw_camera_poses.append(global_raw_transformation[:3, 3].copy())
        global_transformation = correction_transformation @ global_raw_transformation

        # ==== negate arkit jumps
        jump_fix_transformation = jump_negation.step(global_transformation.copy())
        if jump_fix_transformation is not None:
            correction_transformation = jump_fix_transformation @ correction_transformation
            global_transformation = correction_transformation @ global_raw_transformation

        # === save last global tansformation
        jump_negation.set_previous_global_transformation(global_transformation)
        camera_poses.append(global_transformation[:3, 3].copy())

    debug_results(raw_camera_poses, camera_poses, camera_to_world_transformations_with_sw_fixes)


if __name__ == "__main__":
    main()
