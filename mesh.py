from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray  # Nx3 floats
    faces: np.ndarray  # Mx3 ints
    normed_vertex_texture_uvs: np.ndarray  # Nx2 floats 0-1
    face_segmentation_image: np.ndarray  # HxW with int mask matching pixel to face
