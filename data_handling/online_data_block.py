import numpy as np

from data_handling.binary_data_extraction import get_confidence_frame, get_depth_frame
from data_handling.frame_extractor import color_frame_extractor
from mesh import Mesh
from mesh_generator import mesh_generator
from pcd_generator import PointCloud, pcd_generator


class OnlineDataBlock:
    def __init__(
        self,
        timestamp: float,
        camera_to_world_transformation: np.ndarray,
        depth_intrinsics_matrix: np.ndarray | None,
        depth_index: int | None,
        color_index: int | None,
        arpose_index: int,
    ) -> None:
        self.timestamp = timestamp
        self.camera_to_world_transformation = camera_to_world_transformation
        self.depth_intrinsics_matrix = depth_intrinsics_matrix
        self.depth_index = depth_index
        self.color_index = color_index
        self.arpose_index = arpose_index

        # === lazy init if needed
        self._pcd: PointCloud | None = None
        self._mesh: Mesh | None = None
        self._bgr_image: np.ndarray | None = None
        self._is_image_extracted = False

    # === helper
    def get_cam_pos(self) -> np.ndarray:
        return self.camera_to_world_transformation[:3, 3].copy()

    # ===== lazy init properties
    @property
    def bgr_image(self) -> np.ndarray | None:
        if not self._is_image_extracted:
            self._is_image_extracted = True
            self._bgr_image = color_frame_extractor.extract_bgr_frame(self.color_index)
        return self._bgr_image

    @property
    def pcd(self) -> PointCloud:
        if self._pcd is None:
            depth_frame = get_depth_frame(self.depth_index)
            confidence_frame = get_confidence_frame(self.depth_index)

            self._pcd = pcd_generator.generate(
                depth_frame, confidence_frame, self.camera_to_world_transformation, self.depth_intrinsics_matrix
            )

        return self._pcd

    @property
    def mesh(self) -> Mesh:
        if self._mesh is None:
            self._mesh = mesh_generator.generate(self.pcd)
        return self._mesh
