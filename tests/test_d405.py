"""Frame and SDK distortion regression tests; no camera required."""
import unittest

import cv2
import numpy as np

from buildplate_pose.d405 import (
    prepare_d405, native_pixels_to_rays, rays_to_native_pixels,
    validate_calibration, sheet_registration,
)


def calibration():
    return {
        "device": {"model": "Intel RealSense D405"},
        "capture_stream": {"type": "color", "format": "YUYV", "image_size": [80, 60]},
        "intrinsics_frame": "native_unrotated",
        "saved_frame_rotation_degrees": 180,  # historical, must NOT drive job rotation
        "camera_matrix": [[65, 0, 42], [0, 67, 28], [0, 0, 1]],
        "distortion_model": "inverse_brown_conrady",
        "distortion": [-.0543, .063, .00004, .00067, -.0212],
    }


class D405Tests(unittest.TestCase):
    def test_factory_calibration_matches_librealsense_2591_oracle(self):
        # Independently obtained with pyrealsense2 2.59.1.11053 using the
        # inverse_brown_conrady model, NOT cv2.undistortPoints or this module.
        c = calibration()
        c["camera_matrix"] = [[650.9052734375,0,646.8974609375],
                              [0,650.032958984375,363.260009765625],[0,0,1]]
        c["distortion"] = [-0.0543382428586483,0.0629889518022537,
                           0.0000392620349884965,0.000666297506541014,-0.0211622957140207]
        pixels = np.array([[0,0],[1279,0],[0,719],[1279,719],[646.8974609375,363.260009765625],[200,600]],float)
        sdk_rays = [[-1.0068025588989258,-.5656678080558777,1],
                    [.9795717000961304,-.5642462372779846,1],
                    [-1.006719946861267,.5538127422332764,1],
                    [.9795379042625427,.5524416565895081,1], [0,0,1],
                    [-.6977369785308838,.3698672950267792,1]]
        np.testing.assert_allclose(native_pixels_to_rays(pixels,c), sdk_rays, atol=2e-7, rtol=0)
        np.testing.assert_allclose(rays_to_native_pixels(sdk_rays,c), pixels, atol=1.5e-4, rtol=0)

    def test_rotate_once_with_matching_intrinsics_and_camera_axes(self):
        c = calibration()
        c["distortion"] = [0]*5
        image = np.arange(60*80, dtype=np.uint8).reshape(60, 80)
        rotated, rectified, profile = prepare_d405(image, c)
        np.testing.assert_array_equal(rotated, image[::-1, ::-1])
        np.testing.assert_array_equal(rectified, rotated)
        k = np.asarray(profile["camera_matrix"])
        self.assertEqual(k[0, 2], 79-42)
        self.assertEqual(k[1, 2], 59-28)
        q = np.asarray(profile["R_processed_camera_from_native"])
        xyz = np.array([.1, .2, 1.])
        p = np.asarray(c["camera_matrix"]) @ xyz
        rotated_p = k @ q @ xyz
        np.testing.assert_allclose(rotated_p[:2], [79-p[0], 59-p[1]])
        self.assertEqual(profile["image_transform"]["applied_rotation_degrees"], 180)
        self.assertEqual(profile["image_transform"]["source_rotation_degrees"], 0)

    def test_already_rotated_source_is_explicit_not_inferred_from_old_calibration(self):
        c = calibration()
        c["distortion"] = [0]*5
        image = np.zeros((60, 80), np.uint8)
        _, _, profile = prepare_d405(image, c, source_rotation_degrees=180)
        np.testing.assert_allclose(profile["camera_matrix"], c["camera_matrix"])
        np.testing.assert_allclose(profile["R_processed_camera_from_native"], np.eye(3))

    def test_inverse_model_roundtrip_and_differs_from_standard_opencv(self):
        c = calibration()
        pixels = np.array([[2., 3.], [75., 53.], [42., 28.]])
        rays = native_pixels_to_rays(pixels, c)
        np.testing.assert_allclose(rays_to_native_pixels(rays, c), pixels, atol=1e-7)
        ordinary = cv2.undistortPoints(pixels[:, None], np.array(c["camera_matrix"], float),
                                     np.array(c["distortion"])).reshape(-1, 2)
        self.assertGreater(np.max(np.abs(ordinary-rays[:, :2])), 1e-7)

    def test_no_stream_mixing_and_no_size_guessing(self):
        with self.assertRaisesRegex(ValueError, "stream"):
            validate_calibration(calibration(), "d405_depth")
        with self.assertRaisesRegex(ValueError, "size"):
            prepare_d405(np.zeros((30, 40), np.uint8), calibration())

    def test_landmark_origin_is_printed_rectangle_and_remains_provisional(self):
        landmarks = {
            "status": "provisional_visual_annotation", "sheet_outer_size_mm": [254, 240],
            "sheet_outer_virtual_corners_image_px": dict(zip(
                ["lower_left", "lower_right", "upper_right", "upper_left"],
                [[12,1044],[1067,1044],[1067,36],[12,36]])),
            "bed_origin_image_px": [19,975],
        }
        pixels, sheet, details = sheet_registration(landmarks)
        h, _ = cv2.findHomography(np.float32(pixels), np.float32(sheet))
        origin = cv2.perspectiveTransform(np.float32([[[19,975]]]), h)
        np.testing.assert_allclose(origin, 0, atol=1e-7)
        self.assertEqual(details["physical_validation"], False)
        self.assertAlmostEqual(details["origin_from_outer_lower_left_mm"][0], 7*254/1055, places=4)


if __name__ == "__main__":
    unittest.main()
