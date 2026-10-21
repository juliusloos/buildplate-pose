"""D405 preparation in explicit native, saved, and rectified camera frames.

The inverse Brown model follows librealsense rs2_deproject_pixel_to_point and
rs2_project_point_to_pixel (src/rs.cpp), including its modified tangential term.
Do not hand these coefficients to OpenCV's ordinary Brown distortion routines.
The old run_pose_views.py's direct polynomial is not SDK deprojection.
"""
from copy import deepcopy

import cv2
import numpy as np

from .estimate import estimate_pose

DEFAULT_STREAM = "d405_top_yuyv"
RECTIFICATION_VERSION = "librealsense-inverse-brown-v1"


def validate_calibration(calibration, stream=DEFAULT_STREAM):
    """Require exact stream identity; legacy unlabelled calibration is YUYV only."""
    expected = calibration.get("camera_stream", DEFAULT_STREAM)
    capture = calibration.get("capture_stream", {})
    if stream != expected:
        raise ValueError(f"Calibration stream {expected!r} does not match {stream!r}")
    if (calibration.get("device", {}).get("model") != "Intel RealSense D405"
            or capture.get("type") != "color" or capture.get("format") != "YUYV"):
        raise ValueError("Expected D405 color YUYV stream calibration")
    if calibration.get("intrinsics_frame") != "native_unrotated":
        raise ValueError("D405 calibration must describe the native_unrotated frame")
    if calibration.get("distortion_model") != "inverse_brown_conrady":
        raise ValueError("Expected inverse_brown_conrady distortion")
    k = np.asarray(calibration.get("camera_matrix"), dtype=float)
    d = np.asarray(calibration.get("distortion"), dtype=float)
    size = capture.get("image_size")
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0
            or d.shape != (5,) or not np.isfinite(d).all()):
        raise ValueError("Invalid D405 intrinsics or distortion")
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("Invalid calibration image size")


def native_pixels_to_rays(pixels, calibration):
    """SDK inverse-model deprojection at depth=1, in native OpenCV camera axes."""
    k = np.asarray(calibration["camera_matrix"], float)
    k1, k2, p1, p2, k3 = calibration["distortion"]
    xy = (np.asarray(pixels, float) - k[:2, 2]) / [k[0, 0], k[1, 1]]
    xo, yo = xy[..., 0], xy[..., 1]
    x, y = xo.copy(), yo.copy()
    for _ in range(10):
        r2 = x*x + y*y
        radial = 1 + ((k3*r2+k2)*r2+k1)*r2
        xq, yq = x*radial, y*radial
        dx = 2*p1*xq*yq + p2*(r2+2*xq*xq)
        dy = 2*p2*xq*yq + p1*(r2+2*yq*yq)
        x, y = (xo-dx)/radial, (yo-dy)/radial
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def rays_to_native_pixels(rays, calibration):
    """SDK projection for inverse Brown; used as the output-to-input remap."""
    rays = np.asarray(rays, float)
    x, y = rays[..., 0]/rays[..., 2], rays[..., 1]/rays[..., 2]
    k1, k2, p1, p2, k3 = calibration["distortion"]
    r2 = x*x+y*y
    f = 1 + ((k3*r2+k2)*r2+k1)*r2
    xq, yq = x*f, y*f
    xd = xq + 2*p1*xq*yq + p2*(r2+2*xq*xq)
    yd = yq + 2*p2*xq*yq + p1*(r2+2*yq*yq)
    k = np.asarray(calibration["camera_matrix"], float)
    return np.stack([xd*k[0, 0]+k[0, 2], yd*k[1, 1]+k[1, 2]], axis=-1)


def prepare_d405(image, calibration, *, stream=DEFAULT_STREAM, source_rotation_degrees=0):
    """Rotate a source exactly once by 180°, then rectify in that processed frame.

    source_rotation_degrees is the source file's rotation relative to native
    sensor pixels, explicitly supplied for this capture pipeline. The historical
    saved_frame_rotation_degrees field is provenance only, never a job default.
    Returns (rotated_copy, rectified_copy, zero_distortion_profile).
    """
    validate_calibration(calibration, stream)
    if source_rotation_degrees not in (0, 180):
        raise ValueError("Source rotation must be 0 or 180 degrees")
    width, height = calibration["capture_stream"]["image_size"]
    if image is None or image.shape[:2] != (height, width):
        raise ValueError("Source image size does not match exact stream calibration")
    rotated = cv2.rotate(image, cv2.ROTATE_180)
    orientation = (source_rotation_degrees + 180) % 360
    q = np.diag([-1., -1., 1.]) if orientation else np.eye(3)
    k = np.array(calibration["camera_matrix"], float)
    if orientation:
        k[0, 2], k[1, 2] = width-1-k[0, 2], height-1-k[1, 2]
    v, u = np.indices((height, width), dtype=float)
    rays = np.stack([(u-k[0, 2])/k[0, 0], (v-k[1, 2])/k[1, 1], np.ones_like(u)], axis=-1)
    native_rays = rays @ q
    source_pixels = rays_to_native_pixels(native_rays, calibration)
    if orientation:
        source_pixels = [width-1, height-1] - source_pixels
    rectified = cv2.remap(rotated, source_pixels[..., 0].astype(np.float32),
                          source_pixels[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    profile = {
        "schema_version": 1, "camera_model": "D405", "camera_role": "top",
        "camera_stream": stream, "image_size": [width, height],
        "camera_matrix": k.tolist(), "distortion": [0.]*5, "distortion_model": "none",
        "image_frame": "processed_rectified_opencv", "R_processed_camera_from_native": q.tolist(),
        "rectification": RECTIFICATION_VERSION,
        "image_transform": {"source_rotation_degrees": source_rotation_degrees,
                            "applied_rotation_degrees": 180, "rotation_count": 1,
                            "processed_rotation_from_native_degrees": orientation,
                            "rotated_pixel_from_source": [[-1,0,width-1],[0,-1,height-1],[0,0,1]],
                            "historical_calibration_saved_rotation_degrees": calibration.get("saved_frame_rotation_degrees")},
        "source_calibration": deepcopy(calibration),
    }
    return rotated, rectified, profile


def sheet_registration(landmarks):
    """Extract the experimental metric registration without inventing G-code offsets."""
    keys = ["lower_left", "lower_right", "upper_right", "upper_left"]
    pixels = np.float32([landmarks["sheet_outer_virtual_corners_image_px"][key] for key in keys])
    width, depth = landmarks["sheet_outer_size_mm"]
    if min(width, depth) <= 0 or not np.isfinite(pixels).all():
        raise ValueError("Invalid sheet dimensions or reference landmarks")
    outside_mm = np.float32([[0,0],[width,0],[width,depth],[0,depth]])
    h = cv2.getPerspectiveTransform(pixels, outside_mm)
    origin = cv2.perspectiveTransform(np.float32(landmarks["bed_origin_image_px"]).reshape(1,1,2), h).reshape(2)
    sheet = (outside_mm-origin)/1000
    return pixels, sheet, {
        "landmark_status": landmarks["status"], "physical_validation": False,
        "origin": "lower-left printed rectangle", "axes": "X right, Y towards sheet top, Z out of sheet",
        "origin_from_outer_lower_left_mm": origin.tolist(),
        "sheet_outer_size_mm": [width, depth],
        "gcode_registration": "unknown; sheet coordinates are not measured G-code coordinates",
    }


def estimate_sheet_pose(reference_image, rectified_image, landmarks, camera_profile):
    """Use the package's estimator; a numerical pass remains physically provisional."""
    pixels, sheet_m, registration = sheet_registration(landmarks)
    result = estimate_pose(reference_image, rectified_image, pixels, sheet_m,
                           camera_profile["camera_matrix"], np.zeros(5))
    result["numerical_status"] = "passed" if result["status"] == "accepted" else "failed"
    if result["status"] == "accepted":
        result["status"] = "provisional"
        result["T_camera_from_sheet"] = result.pop("T_camera_from_bed")
        result["camera_position_sheet_m"] = result.pop("camera_position_bed_m")
    result["sheet_registration"] = registration
    result["physical_validation"] = False
    return result


def pose_overlay(image, result, camera_profile, landmarks):
    """Draw registered sheet extent and axes; label uncertainty on the image itself."""
    overlay = image.copy()
    if "T_camera_from_sheet" in result:
        _, sheet_m, _ = sheet_registration(landmarks)
        points = np.vstack([np.column_stack([sheet_m, np.zeros(4)]),
                            [[0,0,0],[.05,0,0],[0,.05,0],[0,0,.05]]])
        t = np.asarray(result["T_camera_from_sheet"])
        r, _ = cv2.Rodrigues(t[:3,:3])
        p, _ = cv2.projectPoints(points, r, t[:3,3], np.array(camera_profile["camera_matrix"]), np.zeros(5))
        p = np.rint(p.reshape(-1,2)).astype(int)
        cv2.polylines(overlay, [p[:4]], True, (0,255,255), 2)
        for end, color, label in zip(p[5:], [(0,0,255),(0,255,0),(255,0,0)], "XYZ"):
            cv2.line(overlay, tuple(p[4]), tuple(end), color, 2)
            cv2.putText(overlay, label, tuple(end), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
    text = result["status"] + ": " + result.get("reason", "numerical pass; physical calibration UNVALIDATED")
    cv2.rectangle(overlay, (0,0), (overlay.shape[1],48), (0,0,0), -1)
    cv2.putText(overlay, text[:115], (10,30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,210,255), 2)
    return overlay
