"""The public example should work without any private printer assets."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SyntheticExampleTests(unittest.TestCase):
    def test_generated_fixture_runs_through_public_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generator = subprocess.run([
                sys.executable, "examples/generate_synthetic_fixture.py",
                "--output-dir", str(root),
            ], capture_output=True, text=True)
            self.assertEqual(generator.returncode, 0, generator.stderr)
            result = subprocess.run([
                sys.executable, "-m", "buildplate_pose.cli",
                "--reference-image", str(root / "reference.png"),
                "--capture-image", str(root / "capture.png"),
                "--sheet-profile", str(root / "sheet.json"),
                "--camera-profile", str(root / "camera.json"),
                "--output", str(root / "pose.json"),
                "--overlay", str(root / "overlay.png"),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            pose = json.loads((root / "pose.json").read_text())
            self.assertEqual(pose["status"], "accepted")
            self.assertLess(pose["quality"]["reprojection_p95_px"], 5)
            self.assertTrue((root / "overlay.png").exists())
