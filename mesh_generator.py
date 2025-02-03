import cv2
import numpy as np

from mesh import Mesh
from params import params
from pcd_generator import PointCloud, base_depth_meshgrid


class _MeshGenerator:
    def __init__(self) -> None:
        ind_j, ind_i = base_depth_meshgrid()

        self._base_vertex_texture_uvs = np.column_stack(
            (ind_j.reshape(-1, 1) / params.depth_wh[0], 1 - ind_i.reshape(-1, 1) / params.depth_wh[1])
        )

        faces = []
        h, w = ind_j.shape
        for i in range(h - 1):
            for j in range(w - 1):
                faces.append([w * i + j, w * (i + 1) + j, w * i + (j + 1)])
                faces.append([w * i + (j + 1), w * (i + 1) + j, w * (i + 1) + (j + 1)])
        self._base_faces = np.asarray(faces)

        self._base_face_segmentation_image = self._build_base_face_segmentation_image()

    def _build_base_face_segmentation_image(self):
        # Initialize an empty image; using -1 for initialization to represent pixels not covered by any triangle
        face_segmentation_image = np.full((params.color_wh[1], params.color_wh[0]), -1, dtype=np.int32)

        # Iterate over each face (triangle) and draw it on the segmentation_image
        for tri_index, face in enumerate(self._base_faces):
            # Get the UV coordinates for each vertex of the triangle
            # Scale them according to the image size
            uv_coords = (
                self._base_vertex_texture_uvs[face] * params.color_wh
            )  # Assuming face is [v1, v2, v3] and UVs are normalized
            uv_coords = uv_coords.astype(np.int32)  # Convert to integer for pixel indices

            # Create a contour for the triangle
            contour = uv_coords.reshape((-1, 1, 2))

            # Fill the triangle with the tri_index
            cv2.fillConvexPoly(face_segmentation_image, contour, color=tri_index)
        face_segmentation_image = np.flipud(face_segmentation_image)

        return face_segmentation_image

    def generate(self, pcd: PointCloud):
        if np.all(~pcd.valid_depth_mask):
            return

        valid_depth_mask_flatten = pcd.valid_depth_mask.reshape(-1)
        vertex_texture_uvs_valid = self._base_vertex_texture_uvs[valid_depth_mask_flatten]

        # ==== build valid mesh
        valid_vertices_inds = np.argwhere(valid_depth_mask_flatten).reshape(-1)
        valid_tri_mask = self._get_valid_tri_mask(pcd.all_xyz, valid_vertices_inds, pcd.camera_pos)

        # === reindex faces with valid vertices
        valid_faces_old_vertex_inds = self._base_faces[valid_tri_mask]
        faces_reindexed = _reindex(valid_vertices_inds, valid_faces_old_vertex_inds)

        # === reindex tri_segmentation_image with valid faces
        faces_valid_inds = np.argwhere(valid_tri_mask).reshape(-1)
        tri_segmentation_image_reindexed = _reindex(faces_valid_inds, self._base_face_segmentation_image)

        return Mesh(
            pcd.all_xyz[pcd.valid_depth_mask.reshape(-1)].copy(),
            faces_reindexed,
            vertex_texture_uvs_valid,
            tri_segmentation_image_reindexed,
        )

    def _get_valid_tri_mask(self, xyz, valid_vertices_inds, camera_pos):
        tri_centers, tri_normals = self._calc_faces_attributes(xyz)

        angles_rad = _tri_angle_to_camera(tri_centers, tri_normals, camera_pos)
        good_angles_tri_mask = angles_rad < params.tri_max_viewing_angle_th_deg / 180 * np.pi

        all_vertex_valid_tri_mask = np.all(np.isin(self._base_faces, valid_vertices_inds), axis=1)
        valid_tri_mask = good_angles_tri_mask & all_vertex_valid_tri_mask
        return valid_tri_mask

    def _calc_faces_attributes(self, xyz):
        v0 = xyz[self._base_faces[:, 0]]
        v1 = xyz[self._base_faces[:, 1]]
        v2 = xyz[self._base_faces[:, 2]]

        # Calculate the centers of each triangle
        tri_centers = (v0 + v1 + v2) / 3

        # Calculate normals for each triangle
        tri_normals = np.cross(v1 - v0, v2 - v0)
        tri_normals /= np.linalg.norm(tri_normals, axis=1, keepdims=True)  # Normalize

        return tri_centers, tri_normals


def _tri_angle_to_camera(tri_centers, tri_normals, cam_pos_world):
    # Compute the camera rays from the origin to the triangle centers
    to_camera_rays = cam_pos_world - tri_centers  # Since camera is at origin
    to_camera_rays /= np.linalg.norm(to_camera_rays, axis=1)[:, np.newaxis]  # Normalize

    # Compute the angle between the normals and the camera rays
    dots = np.einsum("ij,ij->i", tri_normals, to_camera_rays)  # Dot product for each pair
    angles_rad = np.arccos(np.clip(dots, -1.0, 1.0))  # Clamp values for numerical stability
    return angles_rad


def _reindex(orig_inds, array_with_orig_inds):
    num_elements_in_lut_excluding_minus_one = max(orig_inds.max(), array_with_orig_inds.max()) + 1
    # account for the possibility of -1 by adding +1 in the next 3 lines
    lut = np.full(num_elements_in_lut_excluding_minus_one + 1, fill_value=-1 + 1, dtype=int)
    lut[orig_inds + 1] = np.arange(len(orig_inds)) + 1
    reindexed_arr_plus_1 = lut[array_with_orig_inds + 1]

    # return the -1 invalid flag
    return reindexed_arr_plus_1 - 1


mesh_generator = _MeshGenerator()
