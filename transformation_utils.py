import numpy as np


def apply_rigid_motion_transformation(xyz: np.ndarray, P: np.ndarray):
    return xyz @ P[:3, :3].T + P[:3, 3]


def nearest_rotation_matrix(A):
    """
    Find the nearest rotation matrix to a given 3x3 matrix A.
    from: https://mathoverflow.net/a/86544

    Parameters:
    - A: a 3x3 numpy array

    Returns:
    - R: a 3x3 rotation matrix closest to A
    """
    # Perform SVD of A
    U, _, Vt = np.linalg.svd(A)

    # Check if the determinant of U*V^T is -1, indicating a reflection
    if np.linalg.det(U @ Vt) < 0:
        # To ensure a rotation, we adjust the sign of the last column of U
        U[:, -1] *= -1

    # Construct the rotation matrix R = U V^T ensuring it's a rotation
    R = U @ Vt

    return R


def find_best_translation(src_center, dest_center):
    src_to_dst_transformation_matrix = np.eye(4)

    translation_vector = dest_center - src_center
    src_to_dst_transformation_matrix[:3, 3] = translation_vector

    return src_to_dst_transformation_matrix


def find_best_yaw(src, src_center, dest, dest_center):
    # Centralize data
    src_centered = src - src_center
    dest_centered = dest - dest_center
    R_yaw_only = _find_best_yaw_rotation_3d(src_centered, dest_centered)

    # Create homogeneous transformation matrix
    src_to_dst_transformation_matrix = np.eye(4)  # Initialize as 4x4 identity matrix
    src_to_dst_transformation_matrix[:3, :3] = R_yaw_only  # Insert the rotation
    translation_vector = dest_center - (R_yaw_only @ src_center)  # Compute the translation
    src_to_dst_transformation_matrix[:3, 3] = translation_vector  # Insert the translation

    return src_to_dst_transformation_matrix


def _find_best_yaw_rotation_3d(src_centered, dest_centered):
    """
    good proof best unconstrained rotation (and translation) Using SVD: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    """
    # NOTE: when removing the y axis (up- against gravity), we left with x and z which looks like this from above:
    # ----->x
    # |
    # |
    # \/
    # z
    # this is NOT the right hand rule! so we get here theta that is clockwise (instead of regular CCW)
    # regular 2D rotation:
    # | cos(theta) -sin(theta) |
    # | sin(theta)  cos(theta) |
    # and when using CW notation or left hand rule (theta -> -theta):
    # |  cos(theta) sin(theta) |
    # | -sin(theta) cos(theta) |
    # regular 3D Rotation Matrix around the Y-axis:
    # |  cos(theta)  0  sin(theta) |
    # |      0       1      0      |
    # | -sin(theta)  0  cos(theta) |

    src_centered_2d = src_centered[:, [0, 2]]
    dest_centered_2d = dest_centered[:, [0, 2]]

    # Compute rotation using SVD
    H = src_centered_2d.T @ dest_centered_2d
    U, _, Vt = np.linalg.svd(H)
    R_2d_left_hand_rule = Vt.T @ U.T

    # Ensure R_2d is a proper rotation matrix
    if np.linalg.det(R_2d_left_hand_rule) < 0:
        Vt[1, :] *= -1
        R_2d_left_hand_rule = Vt.T @ U.T

    # Convert 2D rotation matrix to 3D rotation matrix (around y-axis)
    R_yaw_only = np.array(
        [
            [R_2d_left_hand_rule[0, 0], 0, R_2d_left_hand_rule[0, 1]],
            [0, 1, 0],
            [R_2d_left_hand_rule[1, 0], 0, R_2d_left_hand_rule[1, 1]],
        ]
    )

    return R_yaw_only


def rotation_angle_degrees(R_src, R_dst=np.eye(3)):
    """
    implicitly finds the rotation vector of the needed rotation from src to dest and then calc the rotation in degrees.
    """
    # Compute the relative rotation and translation
    R_rel = R_dst @ R_src.T  # Relative rotation matrix

    # Compute the angle of rotation in degrees
    # this can be derived from Rodrigues: https://mathworld.wolfram.com/RodriguesRotationFormula.html
    # trace(R_rel) = [cos(t) + w_x^2(1-cos(t))] + [cos(t) + w_y^2(1-cos(t))] + [cos(t) + w_z^2(1-cos(t))] =
    #              = 3cos(t) + (1-cos(t))[w_x^2 + w_y^2 + w_z^2] =
    #              = 2cos(t) + 1
    arccos_arg = (np.trace(R_rel) - 1) / 2
    arccos_arg_clipped = np.clip(arccos_arg, -1, 1)
    assert np.isclose(arccos_arg, arccos_arg_clipped)
    theta_degrees = np.arccos(arccos_arg_clipped) * (180 / np.pi)
    return theta_degrees
