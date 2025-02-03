import abc
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import av
import cv2
from loguru import logger


class _AbstractColorExtractor(abc.ABC):
    @abc.abstractmethod
    def extract_bgr_frame(self, color_ind):
        """Extract and return the frame as BGR from the specific index or None if not exist"""
        pass


class VideoFrameExtractor(_AbstractColorExtractor):
    def __init__(self, video_file):
        with _pyav_clean_stderr():
            self._video_file = video_file
            self._open_container()

    def __del__(self):
        with _pyav_clean_stderr():
            self.container.close()

    def extract_bgr_frame(self, requested_frame_idx):
        with _pyav_clean_stderr():
            if requested_frame_idx < self.current_frame_idx:
                # === Restart the container for a fresh iteration
                self.container.close()
                self._open_container()

            try:
                for frame in self.container.decode(self.video_stream):
                    if self.current_frame_idx == requested_frame_idx:
                        # Convert the PyAV frame to an BGR NumPy array
                        img = cv2.cvtColor(frame.to_rgb().to_ndarray(), cv2.COLOR_RGB2BGR)
                        self.current_frame_idx += 1
                        return img
                    self.current_frame_idx += 1
            except EOFError:
                # EOF
                return None

    def _open_container(self):
        self.container = av.open(self._video_file)
        self.video_stream = self.container.streams.video[0]
        self.current_frame_idx = 0


KNOWN_PYAV_WARNINGS_START_TO_IGNORE = [
    # no idea what this is...
    "Duplicated SBGP sync atom",
    # explanation: https://superuser.com/questions/1273920/deprecated-pixel-format-used-make-sure-you-did-set-range-correctly
    "deprecated pixel format used, make sure you did set range correctly",
    # (repeated x times)
    " (repeated",
    # warnings in macs
    "No accelerated colorspace conversion found from",
]


@contextmanager
def _pyav_clean_stderr():
    stderr_list: list[str] = []
    old_stderr = sys.stderr  # Save the current stdout and stderr
    temp_err = StringIO()  # Temporary string buffers
    try:
        sys.stderr = temp_err  # Redirect stdout and stderr to the buffers
        yield
    finally:
        sys.stderr = old_stderr  # Restore stdout and stderr
        stderr_list.extend(temp_err.getvalue().splitlines())

        stderr_list_truncated = [
            line for line in stderr_list if not any([line.startswith(x) for x in KNOWN_PYAV_WARNINGS_START_TO_IGNORE])
        ]
        if len(stderr_list_truncated) > 0:
            logger.warning("---- pyav unknown output from stderr:\n" + "\n".join(stderr_list_truncated))


class _JpgDirExtractor(_AbstractColorExtractor):
    def __init__(self, dir_path: Path) -> None:
        self._jpg_dict: dict[int, Path] = {}

        for fp in dir_path.glob("*"):
            if fp.suffix.lower() not in [".jpg", ".jpeg"]:
                continue

            file_number = int(fp.stem)
            self._jpg_dict[file_number] = fp

    def extract_bgr_frame(self, color_ind):
        if color_ind not in self._jpg_dict:
            return None

        return cv2.imread(self._jpg_dict[color_ind])


class GeneralColorFrameExtractor(_AbstractColorExtractor):
    def __init__(self) -> None:
        self._color_extractor = None

    def initialize(self, input_dir_path: Path):
        algo_jpg_color_frames_path = input_dir_path / "algo_debug" / "jpg_color_frames"
        if algo_jpg_color_frames_path.is_dir():
            self._color_extractor = _JpgDirExtractor(algo_jpg_color_frames_path)
            logger.debug(f"Using color frames from jpg dir: {algo_jpg_color_frames_path}")
        else:
            logger.warning(
                "NOTE: use the script 'scripts/BUILD_jpg_color_frames.py' to preprocess the video and run faster"
            )
            video_path = input_dir_path / "Frames.m4v"
            assert video_path.is_file(), f"{video_path=} doesn't exists"

            self._color_extractor = VideoFrameExtractor(video_path)
            logger.debug(f"Using color frames from raw video: {video_path}")

    def extract_bgr_frame(self, color_ind):
        assert self._color_extractor is not None, " must initialize: color_frame_extractor.initialize(input_dir_path)"

        return self._color_extractor.extract_bgr_frame(color_ind)


color_frame_extractor = GeneralColorFrameExtractor()
