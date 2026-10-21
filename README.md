# Buildplate Pose

Buildplate Pose estimates a camera's six-degree-of-freedom pose relative to a
registered 3D-printer build plate. It uses local image features, geometric
verification, calibrated camera intrinsics, and planar PnP. It produces a
versioned JSON pose and a diagnostic overlay.

This is an early prototype. Tests verify synthetic perspective images; pose
accuracy on physical Prusa MK3 smooth PEI sheets has not been measured. The
printed grid can be repetitive and real captures can be reflective or occluded.
Rejected estimates are explicit.

The current command matches a reference photograph to a capture. A photograph
showing the MK3 sheet's grid, text and warning symbols can serve as the visual
reference even when later captures show only part of the sheet. A planned
vector-feature path will also match registered geometry from the PrusaSlicer
bed SVG and a separately measured physical-sheet outline. Either path still
needs measured feature coordinates in the printer's bed frame and calibrated
camera intrinsics before it can produce a metric pose. The SVG texture does
not by itself establish sheet dimensions or the offset from G-code (0, 0).

## Install

    python -m pip install .

Python 3.10 or newer is required. The package does not control a printer,
depend on OctoPrint, or include any real printer images or trained weights.

## Inputs

For the current photo-matching command, the reference image must show the same
physical sheet and side as the capture.
Register at least four non-collinear image points to measured coordinates in
metres in the printer bed/G-code frame. A sheet profile contains:

    {
      "schema_version": 1,
      "sheet_id": "example-sheet",
      "side": "A",
      "reference_points_image": [[0, 0], [499, 0], [499, 499], [0, 499]],
      "reference_points_bed": [[0, 0], [0.25, 0], [0.25, 0.25], [0, 0.25]]
    }

Those numbers describe a synthetic example and are not MK3 dimensions. The
current physical MK3 smooth sheet was reported as 0.254 m wide and 0.240 m
deep. Its corners still need to be registered to the reference image, and the
installed sheet's offset from G-code (0, 0) must be measured separately.

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
distortion coefficients for real images. For a RealSense, query the **same color
stream profile used to capture the images** on the connected device with
`rs-enumerate-devices -c` or the SDK's `get_intrinsics()`. Record its resolution,
serial number, `fx`, `fy`, `ppx`, `ppy`, distortion model and coefficients.
Camera-model specifications found online are not a device calibration. A known
camera mounting pose is not needed to read the stream intrinsics. The current
JSON/OpenCV estimator expects distortion coefficients compatible with OpenCV's
projection model; if the RealSense reports another model or the saved image is
resized/cropped/rectified, transform or rectify the pixels and profile first.
The D415 color stream may report Modified Brown-Conrady distortion; do not
silently copy its coefficients into an unverified OpenCV profile.
The connected D405's factory color intrinsics and capture details are preserved in
[`calibration/realsense_inverse_brown_conrady_2026-10-05.json`](calibration/realsense_inverse_brown_conrady_2026-10-05.json).
They apply to the native, unrotated 1280 × 720 YUYV stream. Saved frames are
rotated 180°. For detected pixels in a saved frame, use
`u_native = 1279 - u_saved` and `v_native = 719 - v_saved` before librealsense
deprojection with these intrinsics. Its Inverse Brown Conrady coefficients must
not be passed as OpenCV-standard distortion coefficients to `solvePnP` or
`undistortPoints`; this JSON is a reference calibration, not a ready-to-use
`--camera-profile`. An OpenCV-only path needs a separate calibration through
the exact capture and rotation pipeline. Plate dimensions and matched points
are also needed to estimate the camera-to-plate pose.

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
bed frame to the OpenCV camera frame, `camera_position_bed_m` (the camera
center in bed coordinates), plus match and reprojection metrics. The position
is computed as `-R.T @ t` from the transform's rotation `R` and translation
`t`; `t` itself is not the camera position.

For an already estimated pose, `buildplate_pose.grid.refine_pose_from_grid`
provides an optional grid refinement helper. It accepts an 8-bit **pinhole**
image, registered grid line endpoints shaped `(N, 2, 3)` in bed-frame metres,
the matching camera matrix, and the initial `T_camera_from_bed`. Rectify the
capture first using its actual distortion model. The helper finds bright dash
centers near the predicted grid and fits one rigid pose to their perpendicular
line distances. Thus registered parallel and perpendicular lines retain their
geometry in the bed plane; their angles in a perspective image need not be 90°.
Insufficient support, evidence in only one direction, or implausible corrections
leave the original pose unchanged. This helper is opt-in and is not invoked by
the photo-matching CLI. Reference grid locations and the physical sheet outline
still require registration; a good printed-grid fit does not validate a guessed
outer outline or the camera calibration.

The local D405/MK3 replay example combines fresh feature matching with full-span
grid observations and the SDK-compatible projection routines in
`buildplate_pose.d405` and `buildplate_pose.realsense`. Unlike older exploratory
pose-view scripts, it uses iterative SDK inverse-Brown deprojection and the
matching forward projection. It spatially balances line support and evaluates
points withheld from pose refinement, including the left ends. The final output
is then fitted to all observations. Install `matplotlib` as well to generate its
3D view, and run from the checkout with the local images/profiles available:

    PYTHONPATH=src python examples/estimate_mk3_grid.py \
      --capture-image capture.jpg \
      --source-rotation-degrees 180 \
      --output-dir examples/output/mk3_grid

The source rotation is an explicit input relative to native sensor pixels; use
the value established by the capture pipeline. The historical rotation recorded
in the factory profile is not used as a job default. The example writes a report,
the original-image plate/grid overlays, a left-side detail, observations, and a
3D view. A good image fit still does not establish physical pose accuracy or
calibration applicability to that JPEG.

Run tests against the source checkout with:

    PYTHONPATH=src python -m unittest discover -s tests -v

## Contributing data

See CONTRIBUTING.md. Small approved fixtures may be added to this code
repository. Larger image sets should be released as a separate versioned
dataset with a dataset card, measured annotations, and fixed evaluation
splits. No physical camera images or pretrained models are included in this
initial source release.

Code is licensed under Apache-2.0; see LICENSE.
