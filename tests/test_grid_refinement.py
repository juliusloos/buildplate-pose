"""Grid constraints must improve independently rendered plate geometry."""

import unittest

import cv2
import numpy as np

try:
    from buildplate_pose.grid import refine_pose_from_grid
except ModuleNotFoundError:
    refine_pose_from_grid = None
try:
    from buildplate_pose.grid import detect_grid_lines
except ImportError:
    detect_grid_lines = None


class GridRefinementTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
        self.true_pose = np.eye(4)
        tilt, _ = cv2.Rodrigues(np.array([.14, -.09, .03]))
        self.true_pose[:3, :3] = tilt @ np.diag([1., -1., -1.])
        self.true_pose[:3, 3] = [-.12, .11, .45]
        self.initial = self.true_pose.copy()
        self.initial[:3, 3] += [.003, -.002, .005]
        self.lines = np.array(
            [[[x, .01, 0], [x, .21, 0]] for x in [.04, .08, .12, .16, .20]]
            + [[[.01, y, 0], [.23, y, 0]] for y in [.03, .07, .11, .15, .19]]
        )
        self.image = np.full((480, 640), 25, dtype=np.uint8)
        for start, end in self.lines:
            for fraction in np.arange(0, .99, .04):
                a, b = start+(end-start)*fraction, start+(end-start)*(fraction+.022)
                points = self.project([a, b], self.true_pose)
                cv2.line(self.image, tuple(np.rint(points[0]).astype(int)),
                         tuple(np.rint(points[1]).astype(int)), 170, 2, cv2.LINE_AA)
        # Occlusion and clutter must not displace the entire plate.
        cv2.rectangle(self.image, (180, 100), (280, 170), 25, -1)
        cv2.line(self.image, (230, 300), (440, 180), 200, 2)

    def project(self, points, transform):
        xyz = np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]
        return xyz[:, :2] / xyz[:, 2:] * 600 + [320, 240]

    def test_grid_improves_corner_projection_with_occlusion_and_clutter(self):
        self.assertTrue(callable(refine_pose_from_grid), "Grid refinement is missing")
        result = refine_pose_from_grid(self.image, self.lines, self.matrix, self.initial)
        self.assertEqual(result["status"], "refined", result)
        corners = [[0, 0, 0], [.24, 0, 0], [.24, .22, 0], [0, .22, 0]]
        expected = self.project(corners, self.true_pose)
        before = np.linalg.norm(self.project(corners, self.initial)-expected, axis=1).mean()
        after = np.linalg.norm(self.project(corners, np.array(result["T_camera_from_bed"]))-expected, axis=1).mean()
        self.assertLess(after, 1.5)
        self.assertLess(after, before*.4)
        self.assertGreaterEqual(result["quality"]["line_count"], 6)

    def test_blank_image_keeps_original_pose(self):
        self.assertTrue(callable(refine_pose_from_grid), "Grid refinement is missing")
        result = refine_pose_from_grid(np.full_like(self.image, 25), self.lines, self.matrix, self.initial)
        self.assertEqual(result["status"], "unchanged")
        np.testing.assert_array_equal(result["T_camera_from_bed"], self.initial)

    def test_parallel_lines_alone_cannot_change_pose(self):
        result = refine_pose_from_grid(self.image, self.lines[:5], self.matrix, self.initial)
        self.assertEqual(result["status"], "unchanged")
        np.testing.assert_array_equal(result["T_camera_from_bed"], self.initial)

    def test_dash_detection_retains_support_near_both_ends(self):
        self.assertTrue(callable(detect_grid_lines), "Full-span detection is missing")
        image = np.full((120, 520), 25, dtype=np.uint8)
        for x in range(41, 459, 10):
            cv2.line(image, (x, 60), (x+4, 60), 180, 2, cv2.LINE_AA)
        detections = detect_grid_lines(image, np.array([[[40, 60], [460, 60]]]), 8)
        self.assertEqual(len(detections), 1)
        self.assertLess(detections[0]["support_fraction_range"][0], .025)
        self.assertGreater(detections[0]["support_fraction_range"][1], .975)

    def test_fixed_observations_are_used_without_redetecting_image(self):
        observations = [{"line_index": i,
                         "points_pinhole_px": self.project(np.linspace(*segment, 30), self.true_pose)}
                        for i, segment in enumerate(self.lines)]
        result = refine_pose_from_grid(np.full_like(self.image, 25), self.lines,
                                       self.matrix, self.initial, observed_lines=observations)
        self.assertEqual(result["status"], "refined", result)
        corners = [[0, 0, 0], [.24, .22, 0]]
        np.testing.assert_allclose(self.project(corners, np.array(result["T_camera_from_bed"])),
                                   self.project(corners, self.true_pose), atol=1e-4)

    def test_large_correction_is_rejected_even_with_consistent_observations(self):
        observations = [{"line_index": i,
                         "points_pinhole_px": self.project(np.linspace(*segment, 30), self.true_pose)}
                        for i, segment in enumerate(self.lines)]
        initial = self.true_pose.copy()
        initial[0, 3] += .04
        result = refine_pose_from_grid(self.image, self.lines, self.matrix, initial,
                                       observed_lines=observations)
        self.assertEqual(result["status"], "unchanged", result)
        np.testing.assert_array_equal(result["T_camera_from_bed"], initial)


if __name__ == "__main__":
    unittest.main()
