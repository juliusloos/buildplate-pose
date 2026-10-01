"""Versioned accepted-pose contract for downstream consumers."""

import json
from pathlib import Path

import numpy as np


def load_pose(path, *, expected_camera_role=None):
    """Load and validate an accepted OpenCV bed-to-camera transform."""
    pose = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(pose, dict) or pose.get("schema_version") != 1:
        raise ValueError("Unsupported camera pose schema")
    if pose.get("status") != "accepted":
        raise ValueError("Camera pose is not accepted")
    role = pose.get("camera_role")
    if not isinstance(role, str) or not role:
        raise ValueError("Camera role is missing")
    if expected_camera_role is not None and role != expected_camera_role:
        raise ValueError(f"Wrong camera role: expected {expected_camera_role!r}, got {role!r}")
    if not isinstance(pose.get("calibration_id"), str) or not pose["calibration_id"]:
        raise ValueError("Calibration ID is missing")
    size = pose.get("image_size")
    if (
        not isinstance(size, list) or len(size) != 2
        or any(type(value) is not int or value <= 0 for value in size)
    ):
        raise ValueError("Invalid image size")
    try:
        transform = np.asarray(pose["T_camera_from_bed"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid rigid camera transform") from exc
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Invalid rigid camera transform")
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("Invalid rigid camera transform")
    return pose
