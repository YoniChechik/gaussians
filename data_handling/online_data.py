from loguru import logger

from data_handling.frame_extractor import color_frame_extractor
from data_handling.online_data_block import OnlineDataBlock
from data_handling.parse_raw_data import ParsedData
from params import params


class OnlineDataIterator:
    """
    step every arpose since this is the highest FPS
    """

    def __init__(self, parse_data: ParsedData) -> None:
        self._parsed_data = parse_data
        color_frame_extractor.initialize(params.input_dir_path)
        self._curr_arpose_ind = 0
        self._tracked_depth_ind = 0
        self._tracked_color_ind = 0

    def __iter__(self):
        return self  # An iterator must return itself

    def __len__(self):
        return len(self._parsed_data.camera_to_world_transformations)

    def __next__(self) -> OnlineDataBlock:
        # Stop iteration if current exceeds limit
        if self._curr_arpose_ind >= len(self):
            raise StopIteration
        if self._tracked_color_ind >= len(self._parsed_data.color_ind_to_arpose_ind):
            logger.warning("Color data finished before arpose data finished")
            raise StopIteration
        if self._tracked_depth_ind >= len(self._parsed_data.depth_ind_to_arpose_ind):
            logger.warning("Depth data finished before arpose data finished")
            raise StopIteration

        curr_color_ind = None
        curr_depth_intrinsics_matrix = None
        if self._parsed_data.color_ind_to_arpose_ind[self._tracked_color_ind] == self._curr_arpose_ind:
            curr_color_ind = self._tracked_color_ind
            curr_depth_intrinsics_matrix = self._parsed_data.depth_intrinsics_matrices_in_color_timetable[
                self._tracked_color_ind
            ]
            self._tracked_color_ind += 1

        curr_depth_ind = None
        if self._parsed_data.depth_ind_to_arpose_ind[self._tracked_depth_ind] == self._curr_arpose_ind:
            curr_depth_ind = self._tracked_depth_ind
            self._tracked_depth_ind += 1

        data_block = OnlineDataBlock(
            self._parsed_data.timestamps[self._curr_arpose_ind],
            self._parsed_data.camera_to_world_transformations[self._curr_arpose_ind],
            curr_depth_intrinsics_matrix,
            curr_depth_ind,
            curr_color_ind,
            self._curr_arpose_ind,
        )

        self._curr_arpose_ind += 1

        return data_block
