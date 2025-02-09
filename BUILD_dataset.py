import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from data_handling.online_data import OnlineDataIterator
from data_handling.parse_raw_data import parse_raw_data
from params import params
from pipe_utils.script_runner import script_runner

# TODO build ply of points with colors from joining the mesh together


def initialize_transform_data() -> dict:
    """Initialize the transform data structure."""
    return {
        "fl_x": None,
        "fl_y": None,
        "cx": None,
        "cy": None,
        "w": None,
        "h": None,
        "frames": [],
    }


def set_intrinsics(transform_data: dict, intrinsics_matrix: np.ndarray) -> None:
    """Set the intrinsic parameters in the transform data."""
    transform_data["fl_x"] = float(intrinsics_matrix[0, 0])
    transform_data["fl_y"] = float(intrinsics_matrix[1, 1])
    transform_data["cx"] = float(intrinsics_matrix[0, 2])
    transform_data["cy"] = float(intrinsics_matrix[1, 2])


def set_image_size(transform_data: dict, color_frame: np.ndarray) -> None:
    """Set the width and height in the transform data based on the color frame size."""
    transform_data["h"], transform_data["w"] = color_frame.shape[:2]


def process_frames(online_data_iter: OnlineDataIterator, output_dir: Path, transform_data: dict) -> None:
    """Iterate over frames, process every second frame up to 100 frames, and save image and transform data."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    intrinsics_set = False
    size_set = False

    logger.info("Starting iteration over every second frame for 100 frames...")
    for frame_ind, data_block in enumerate(tqdm(online_data_iter)):
        if frame_ind >= 200:  # Process up to 100 frames with step=2
            break
        if frame_ind % 2 != 0:
            continue

        global_raw_transformation = data_block.camera_to_world_transformation.copy()

        if not intrinsics_set and data_block.depth_intrinsics_matrix is not None:
            set_intrinsics(transform_data, data_block.depth_intrinsics_matrix)
            intrinsics_set = True

        if data_block.color_index is not None:
            bgr_image = data_block.bgr_image

            if not size_set:
                set_image_size(transform_data, bgr_image)
                size_set = True

            image_path = images_dir / f"frame_{frame_ind:04d}.png"
            cv2.imwrite(image_path, bgr_image)

            frame_entry = {
                "file_path": f"./images/frame_{frame_ind:04d}.png",
                "transform_matrix": global_raw_transformation.tolist(),
            }
            transform_data["frames"].append(frame_entry)

    if not intrinsics_set:
        logger.error("No intrinsics data found in any frame.")
        raise RuntimeError("Failed to set intrinsics.")
    if not size_set:
        logger.error("No color frame data found.")
        raise RuntimeError("Failed to set image size.")


def save_transform_data(transform_data: dict, output_path: Path) -> None:
    """Save the transform data as a JSON file."""
    with open(output_path, "w") as f:
        json.dump(transform_data, f, indent=4)
    logger.info(f"Transforms saved to {output_path}")


@script_runner(params)
def main():
    output_dir = Path(params.save_dir_path)
    transform_data = initialize_transform_data()

    parsed_data, _ = parse_raw_data()
    online_data_iter = OnlineDataIterator(parsed_data)

    try:
        process_frames(online_data_iter, output_dir, transform_data)
        save_transform_data(transform_data, output_dir / "transforms.json")
    except RuntimeError as e:
        logger.error(f"Processing failed: {e}")


if __name__ == "__main__":
    main()
