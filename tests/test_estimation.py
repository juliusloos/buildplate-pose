"""Synthetic projection checks for image-based build-sheet pose estimation."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from buildplate_pose.estimate import estimate_pose
from buildplate_pose.contract import load_pose


class PlatePoseEstimationTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.full((500, 500), 230, dtype=np.uint8)
        rng = np.random.default_rng(42)
        for index in range(260):
            x, y = rng.integers(15, 485, size=2)
            radius = int(rng.integers(2, 8))
            color = int(rng.integers(0, 150))
            cv2.circle(self.reference, (int(x), int(y)), radius, color, -1)
        cv2.putText(self.reference, "PRUSA MK3", (40, 255),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, 15, 3)
        self.reference_points = np.array([
            [0, 0], [499, 0], [499, 499], [0, 499]
        ], dtype=np.float32)
        self.bed_points = np.array([
            [0, 0], [0.25, 0], [0.25, 0.25], [0, 0.25]
        ], dtype=np.float32)
        self.K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
        self.distortion = np.zeros(5)
        self.true_rotation = np.diag([1.0, -1.0, -1.0])
        center = np.array([0.125, 0.125, 0.5])
        self.true_translation = -self.true_rotation @ center
        world_corners = np.column_stack([self.bed_points, np.zeros(4)])
        rvec, _ = cv2.Rodrigues(self.true_rotation)
        image_corners, _ = cv2.projectPoints(
            world_corners, rvec, self.true_translation, self.K, self.distortion
        )
        warp = cv2.getPerspectiveTransform(
            self.reference_points, image_corners.reshape(-1, 2).astype(np.float32)
        )
        self.capture = cv2.warpPerspective(
            self.reference, warp, (640, 480), borderValue=255
        )

    def test_recovers_camera_above_bed_from_perspective_image(self):
        result = estimate_pose(
            self.reference, self.capture, self.reference_points,
            self.bed_points, self.K, self.distortion,
        )
        self.assertEqual(result["status"], "accepted", result)
        estimated = np.array(result["T_camera_from_bed"])
        center = -estimated[:3, :3].T @ estimated[:3, 3]
        np.testing.assert_allclose(center, [0.125, 0.125, 0.5], atol=0.02)
        self.assertIn("camera_position_bed_m", result)
        np.testing.assert_allclose(result["camera_position_bed_m"],
                                   [0.125, 0.125, 0.5], atol=0.02)
        self.assertLess(result["quality"]["reprojection_p95_px"], 5)

    def test_rejects_image_without_plate_features(self):
        blank_capture = np.full((480, 640), 255, dtype=np.uint8)
        result = estimate_pose(
            self.reference, blank_capture, self.reference_points,
            self.bed_points, self.K, self.distortion,
        )
        self.assertEqual(result["status"], "rejected")
        self.assertIn("features", result["reason"])

    def test_rejects_sparse_match_groups(self):
        with patch("buildplate_pose.estimate.cv2.BFMatcher") as matcher_factory:
            matcher_factory.return_value.knnMatch.return_value = [[]]
            result = estimate_pose(
                self.reference, self.capture, self.reference_points,
                self.bed_points, self.K, self.distortion,
            )
        self.assertEqual(result["status"], "rejected")
        self.assertIn("features", result["reason"])

    def test_cli_writes_accepted_pose_and_diagnostic_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference_path = root / "reference.png"
            capture_path = root / "capture.png"
            sheet_path = root / "sheet.json"
            camera_path = root / "camera.json"
            pose_path = root / "pose.json"
            overlay_path = root / "overlay.png"
            cv2.imwrite(str(reference_path), self.reference)
            cv2.imwrite(str(capture_path), self.capture)
            sheet_path.write_text(json.dumps({
                "schema_version": 1,
                "sheet_id": "mk3-smooth-001",
                "side": "A",
                "reference_points_image": self.reference_points.tolist(),
                "reference_points_bed": self.bed_points.tolist(),
            }))
            camera_path.write_text(json.dumps({
                "schema_version": 1,
                "camera_role": "top",
                "calibration_id": "mk3-top-2026-10",
                "image_size": [640, 480],
                "camera_matrix": self.K.tolist(),
                "distortion": self.distortion.tolist(),
            }))
            process = subprocess.run([
                sys.executable, "-m", "buildplate_pose.cli",
                "--reference-image", str(reference_path),
                "--capture-image", str(capture_path),
                "--sheet-profile", str(sheet_path),
                "--camera-profile", str(camera_path),
                "--output", str(pose_path),
                "--overlay", str(overlay_path),
            ], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            loaded = load_pose(pose_path, expected_camera_role="top")
            self.assertEqual(loaded["sheet_id"], "mk3-smooth-001")
            self.assertEqual(loaded["image_size"], [640, 480])
            self.assertTrue(overlay_path.exists())
