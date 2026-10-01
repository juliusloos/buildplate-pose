# Contributing

Code and documentation pull requests are welcome. Run:

    python -m unittest discover -s tests -v

Keep printer control, NAS paths, credentials, private captures, and defect
detection code out of this repository.

For an image contribution, first open an issue describing the sheet model and
side, camera/stream and calibration method, lighting and occlusion conditions,
and how the plate-to-G-code coordinates were measured. Include ground-truth
landmarks or a separately measured pose, not an estimate from this tracker
itself. Contributors must have permission to release the image and metadata.
Check that the frame and EXIF contain no people, credentials, network names,
or other unintended information. Release approved large sets as a separate
dataset artifact; do not add raw capture collections to a code pull request.

Report failures as well as successful detections. Keep benchmark test frames
separate from images used to tune thresholds. Include the package version,
profile IDs, acceptance rate, and pose error when ground truth is available.
