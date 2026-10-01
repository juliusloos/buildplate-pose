"""Generate a deterministic synthetic plate example with known camera geometry."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    reference = np.full((500, 500), 230, dtype=np.uint8)
    rng = np.random.default_rng(42)
    for _ in range(260):
        x, y = rng.integers(15, 485, size=2)
        radius = int(rng.integers(2, 8))
        color = int(rng.integers(0, 150))
        cv2.circle(reference, (int(x), int(y)), radius, color, -1)
    cv2.putText(reference, "BUILD PLATE", (20, 255),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 15, 3)

    reference_corners = np.array(
        [[0, 0], [499, 0], [499, 499], [0, 499]], dtype=np.float32
    )
    bed_corners = np.array(
        [[0, 0], [0.25, 0], [0.25, 0.25], [0, 0.25]], dtype=np.float32
    )
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
    distortion = np.zeros(5)
    rotation = np.diag([1.0, -1.0, -1.0])
    camera_center = np.array([0.125, 0.125, 0.5])
    translation = -rotation @ camera_center
    world_corners = np.column_stack([bed_corners, np.zeros(4)])
    rvec, _ = cv2.Rodrigues(rotation)
    image_corners, _ = cv2.projectPoints(
        world_corners, rvec, translation, K, distortion
    )
    warp = cv2.getPerspectiveTransform(
        reference_corners, image_corners.reshape(-1, 2).astype(np.float32)
    )
    capture = cv2.warpPerspective(reference, warp, (640, 480), borderValue=255)
    if not cv2.imwrite(str(output / "reference.png"), reference):
        raise OSError("Could not write reference image")
    if not cv2.imwrite(str(output / "capture.png"), capture):
        raise OSError("Could not write capture image")

    sheet = {
        "schema_version": 1,
        "sheet_id": "synthetic-sheet",
        "side": "A",
        "reference_points_image": reference_corners.tolist(),
        "reference_points_bed": bed_corners.tolist(),
    }
    camera = {
        "schema_version": 1,
        "camera_role": "top",
        "calibration_id": "synthetic-calibration",
        "image_size": [640, 480],
        "camera_matrix": K.tolist(),
        "distortion": distortion.tolist(),
    }
    ground_truth = np.eye(4)
    ground_truth[:3, :3] = rotation
    ground_truth[:3, 3] = translation
    (output / "sheet.json").write_text(json.dumps(sheet, indent=2) + "\n")
    (output / "camera.json").write_text(json.dumps(camera, indent=2) + "\n")
    (output / "ground_truth.json").write_text(
        json.dumps({"T_camera_from_bed": ground_truth.tolist()}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
