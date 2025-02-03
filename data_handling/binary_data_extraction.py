import numpy as np

from params import params


def _binary_data_extraction(file_path, frame_number, wh, input_type, output_type):
    # Calculate the size of one frame in bytes
    bytes_per_pixel = np.dtype(input_type).itemsize
    frame_size = wh[0] * wh[1] * bytes_per_pixel

    # Calculate the byte offset for the desired frame
    offset = frame_size * frame_number

    # Open the file and seek to the offset
    with open(file_path, "rb") as file:
        file.seek(offset)
        # Read the frame data and convert it to the desired output type
        frame_data = np.frombuffer(file.read(frame_size), dtype=input_type)
        frame_data = frame_data.astype(output_type).reshape((wh[1], wh[0]))

    return frame_data


def get_depth_frame(depth_ind):
    depth = _binary_data_extraction(
        params.input_dir_path / "Depth.hdep",
        depth_ind,
        wh=params.depth_wh,
        input_type=np.float16,
        output_type=np.float32,
    )
    return depth


def get_confidence_frame(depth_ind):
    confidence = _binary_data_extraction(
        params.input_dir_path / "Confidence.L008",
        depth_ind,
        wh=params.depth_wh,
        input_type=np.int8,
        output_type=np.float32,
    )

    return confidence
