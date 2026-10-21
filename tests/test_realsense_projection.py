"""Golden values independently evaluated from librealsense's float routines."""

import unittest

import numpy as np

try:
    from buildplate_pose.realsense import saved_pixels_to_pinhole, pinhole_pixels_to_saved
except ModuleNotFoundError:
    saved_pixels_to_pinhole = pinhole_pixels_to_saved = None


class RealSenseProjectionTests(unittest.TestCase):
    def setUp(self):
        self.camera = {
            "capture_stream": {"image_size": [1280, 720]},
            "saved_frame_rotation_degrees": 180,
            "camera_matrix": [[650.9052734375, 0, 646.8974609375],
                              [0, 650.032958984375, 363.260009765625], [0, 0, 1]],
            "distortion_model": "inverse_brown_conrady",
            "distortion": [-.0543382428586483, .0629889518022537,
                           .0000392620349884965, .000666297506541014, -.0211622957140207],
        }

    def test_deprojection_matches_sdk_instead_of_reversing_radial_correction(self):
        self.assertTrue(callable(saved_pixels_to_pinhole), "SDK-compatible mapping is missing")
        actual = saved_pixels_to_pinhole([[350, 80], [900, 650], [639.5, 359.5]], self.camera)
        np.testing.assert_allclose(actual, [[932.277709961, 642.355590820],
                                           [375.201904297, 65.003997803],
                                           [639.499755859, 359.499908447]], atol=1e-4, rtol=0)

    def test_projection_matches_sdk_scaled_tangential_terms(self):
        self.assertTrue(callable(pinhole_pixels_to_saved), "SDK-compatible mapping is missing")
        actual = pinhole_pixels_to_saved([[350, 80], [900, 650], [639.5, 359.5]], self.camera)
        np.testing.assert_allclose(actual, [[924.787292480, 635.135314941],
                                           [381.784362793, 72.315490723],
                                           [639.499755859, 359.499908447]], atol=1e-4, rtol=0)

    def test_roundtrip_across_frame(self):
        self.assertTrue(callable(saved_pixels_to_pinhole), "SDK-compatible mapping is missing")
        pixels = np.array([[30, 30], [1200, 30], [1200, 690], [30, 690], [640, 360]])
        np.testing.assert_allclose(pinhole_pixels_to_saved(saved_pixels_to_pinhole(pixels, self.camera), self.camera), pixels, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
