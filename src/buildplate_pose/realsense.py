"""RealSense inverse Brown-Conrady pixel mappings and saved-frame rotation.

Equations follow librealsense rs2_project_point_to_pixel and
rs2_deproject_pixel_to_point (Apache-2.0, RealSense, Inc.):
https://github.com/IntelRealSense/librealsense/blob/master/src/rs.cpp
In particular, SDK deprojection iteratively inverts the radial mapping; its
inverse-Brown projection evaluates tangential terms with radially scaled XY.
"""

import cv2
import numpy as np

from .d405 import native_pixels_to_rays, rays_to_native_pixels


def _profile(camera):
    matrix = np.asarray(camera["camera_matrix"], dtype=float)
    coefficients = np.asarray(camera["distortion"], dtype=float)
    rotation = camera["saved_frame_rotation_degrees"]
    if camera["distortion_model"] != "inverse_brown_conrady":
        raise ValueError("Expected a RealSense inverse Brown-Conrady profile")
    if (matrix.shape != (3, 3) or coefficients.shape != (5,)
            or not np.isfinite(matrix).all() or not np.isfinite(coefficients).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or rotation not in (0, 180)):
        raise ValueError("Invalid camera matrix, coefficients, or saved rotation")
    width, height = camera["capture_stream"]["image_size"]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size")
    return matrix, coefficients, rotation, np.array([width-1, height-1])


def _pixels(value):
    pixels = np.asarray(value, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError("Expected finite pixel coordinates shaped (N, 2)")
    return pixels


def saved_pixels_to_pinhole(saved_pixels, camera):
    """Undo saved rotation and deproject as librealsense does at unit depth."""
    matrix, _, rotation, limits = _profile(camera)
    native = _pixels(saved_pixels)
    if rotation == 180:
        native = limits-native
    focal = np.array([matrix[0, 0], matrix[1, 1]])
    rays = native_pixels_to_rays(native, camera)
    return rays[:, :2]*focal + matrix[:2, 2]


def pinhole_pixels_to_saved(pinhole_pixels, camera):
    """Project using the SDK's inverse-Brown branch, then apply saved rotation."""
    matrix, _, rotation, limits = _profile(camera)
    focal = np.array([matrix[0, 0], matrix[1, 1]])
    normalized = (_pixels(pinhole_pixels)-matrix[:2, 2])/focal
    native = rays_to_native_pixels(np.column_stack([normalized, np.ones(len(normalized))]), camera)
    return limits-native if rotation == 180 else native


def rectify_saved_image(saved_image, camera):
    """Create a native pinhole image using the same mapping as point projection."""
    _, _, _, limits = _profile(camera)
    width, height = (limits+1).astype(int)
    if saved_image.shape[:2] != (height, width):
        raise ValueError("Image size differs from camera profile")
    yy, xx = np.mgrid[:height, :width]
    mapping = pinhole_pixels_to_saved(np.column_stack([xx.ravel(), yy.ravel()]), camera)
    mapping = mapping.reshape(height, width, 2).astype(np.float32)
    return cv2.remap(saved_image, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR)
