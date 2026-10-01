"""Consumers must not accept an invalid or reversed camera pose."""

import json
from pathlib import Path
import tempfile
import unittest

from buildplate_pose.contract import load_pose


def _accepted_pose():
    return {
        "schema_version": 1,
        "status": "accepted",
        "camera_role": "top",
        "calibration_id": "mk3-top-2026-10",
        "image_size": [1280, 720],
        "T_camera_from_bed": [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0.5],
            [0, 0, 0, 1],
        ],
    }


class PlatePoseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "pose.json"

    def test_load_pose_rejects_a_failed_estimate(self):
        pose = _accepted_pose()
        pose["status"] = "rejected"
        self.path.write_text(json.dumps(pose))
        with self.assertRaisesRegex(ValueError, "not accepted"):
            load_pose(self.path)

    def test_load_pose_rejects_wrong_camera_role(self):
        self.path.write_text(json.dumps(_accepted_pose()))
        with self.assertRaisesRegex(ValueError, "camera role"):
            load_pose(self.path, expected_camera_role="front")

    def test_load_pose_rejects_nonrigid_transform(self):
        pose = _accepted_pose()
        pose["T_camera_from_bed"][0][0] = 2
        self.path.write_text(json.dumps(pose))
        with self.assertRaisesRegex(ValueError, "rigid"):
            load_pose(self.path)
