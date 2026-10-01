# Buildplate Pose

Buildplate Pose estimates a camera's six-degree-of-freedom pose relative to a
registered 3D-printer build plate. It uses local image features, geometric
verification, calibrated camera intrinsics, and planar PnP. It produces a
versioned JSON pose and a diagnostic overlay.

This is an early prototype. Tests verify synthetic perspective images; pose
accuracy on physical Prusa MK3 smooth PEI sheets has not been measured. The
printed grid can be repetitive and real captures can be reflective or occluded.
Rejected estimates are explicit.

## Install

    python -m pip install .

Python 3.10 or newer is required. The package does not control a printer,
depend on OctoPrint, or include any real printer images or trained weights.

## Inputs

The reference image must show the same physical sheet and side as the capture.
Register at least four non-collinear image points to measured coordinates in
metres in the printer bed/G-code frame. A sheet profile contains:

    {
      "schema_version": 1,
      "sheet_id": "example-sheet",
      "side": "A",
      "reference_points_image": [[0, 0], [499, 0], [499, 499], [0, 499]],
      "reference_points_bed": [[0, 0], [0.25, 0], [0.25, 0.25], [0, 0.25]]
    }

Those numbers describe a synthetic example and are not MK3 dimensions. Measure
the offset between visible sheet features and G-code (0, 0).

A camera profile contains measured intrinsics for the exact stream:

    {
      "schema_version": 1,
      "camera_role": "top",
      "calibration_id": "example-calibration",
      "image_size": [640, 480],
      "camera_matrix": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
      "distortion": [0, 0, 0, 0, 0]
    }

These camera numbers are synthetic examples. Use a calibrated camera matrix and
distortion coefficients for real images.

## Try a synthetic example

    python examples/generate_synthetic_fixture.py --output-dir examples/output
    buildplate-pose \
      --reference-image examples/output/reference.png \
      --capture-image examples/output/capture.png \
      --sheet-profile examples/output/sheet.json \
      --camera-profile examples/output/camera.json \
      --output examples/output/pose.json \
      --overlay examples/output/overlay.png

The generator also writes ground_truth.json. Its synthetic geometry is not a
real MK3 calibration or a real-world accuracy benchmark.

## Estimate

    buildplate-pose \
      --reference-image reference.png \
      --capture-image capture.png \
      --sheet-profile sheet.json \
      --camera-profile camera.json \
      --output pose.json \
      --overlay overlay.png

Exit code 0 means the prototype quality gates accepted the estimate. Exit code
2 means it was rejected; both pose.json and overlay.png are still written.
The accepted result contains T_camera_from_bed, a 4x4 transform from the metric
bed frame to the OpenCV camera frame, plus match and reprojection metrics.

Run tests from a checkout with:

    python -m unittest discover -s tests -v

## Contributing data

See CONTRIBUTING.md. Small approved fixtures may be added to this code
repository. Larger image sets should be released as a separate versioned
dataset with a dataset card, measured annotations, and fixed evaluation
splits. No physical camera images or pretrained models are included in this
initial source release.

Code is licensed under Apache-2.0; see LICENSE.
