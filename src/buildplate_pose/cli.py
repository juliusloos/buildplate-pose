"""Estimate a plate pose from explicit images and calibration profiles."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .estimate import estimate_pose


def _read_profile(path, label):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"Unsupported {label} profile schema")
    return value


def _write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--capture-image", required=True)
    parser.add_argument("--sheet-profile", required=True)
    parser.add_argument("--camera-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay", required=True)
    args = parser.parse_args(argv)

    sheet = _read_profile(args.sheet_profile, "sheet")
    camera = _read_profile(args.camera_profile, "camera")
    reference = cv2.imread(args.reference_image, cv2.IMREAD_GRAYSCALE)
    capture = cv2.imread(args.capture_image, cv2.IMREAD_COLOR)
    if reference is None or capture is None:
        raise ValueError("Could not read reference or capture image")
    width, height = camera["image_size"]
    if (capture.shape[1], capture.shape[0]) != (width, height):
        raise ValueError("Capture image size does not match camera profile")
    if not all(isinstance(camera.get(key), str) and camera[key]
               for key in ("calibration_id", "camera_role")):
        raise ValueError("Camera role and calibration ID are required")
    if not all(isinstance(sheet.get(key), str) and sheet[key]
               for key in ("sheet_id", "side")):
        raise ValueError("Sheet ID and side are required")

    result = estimate_pose(
        reference, capture, sheet["reference_points_image"],
        sheet["reference_points_bed"], camera["camera_matrix"],
        camera["distortion"],
    )
    result.update({
        "schema_version": 1,
        "camera_role": camera["camera_role"],
        "calibration_id": camera["calibration_id"],
        "image_size": camera["image_size"],
        "sheet_id": sheet["sheet_id"],
        "sheet_side": sheet["side"],
        "capture_image_sha256": hashlib.sha256(
            Path(args.capture_image).read_bytes()
        ).hexdigest(),
    })
    _write_json(args.output, result)

    overlay = capture.copy()
    if result["status"] == "accepted":
        transform = np.asarray(result["T_camera_from_bed"], dtype=float)
        rotation_vector, _ = cv2.Rodrigues(transform[:3, :3])
        object_xy = np.asarray(sheet["reference_points_bed"], dtype=float)
        object_xyz = np.column_stack([object_xy, np.zeros(len(object_xy))])
        projected, _ = cv2.projectPoints(
            object_xyz, rotation_vector, transform[:3, 3],
            np.asarray(camera["camera_matrix"], dtype=float),
            np.asarray(camera["distortion"], dtype=float),
        )
        for point in projected.reshape(-1, 2):
            cv2.circle(overlay, tuple(np.rint(point).astype(int)), 8, (0, 255, 0), 2)
        label = f"accepted p95={result['quality']['reprojection_p95_px']:.2f}px"
        color = (0, 180, 0)
    else:
        label = f"rejected: {result['reason']}"
        color = (0, 0, 255)
    cv2.putText(overlay, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2)
    destination = Path(args.overlay)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), overlay):
        raise OSError(f"Could not write diagnostic image: {destination}")
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
