"""Estimate a calibrated camera pose from a metrically registered sheet image."""

import cv2
import numpy as np


MIN_MATCHES = 12
MIN_COVERAGE = 0.05
MAX_REPROJECTION_P95_PX = 5.0


def _reject(reason, **quality):
    return {"status": "rejected", "reason": reason, "quality": quality}


def estimate_pose(reference_image, capture_image, reference_points_image,
                  reference_points_bed, camera_matrix, distortion):
    """Match reference pixels to a capture and solve the plate's planar pose.

    reference_points_image and reference_points_bed are corresponding measured
    registration points. The latter are XY coordinates in metres in the G-code
    bed frame. The reference must show the same physical sheet and side.
    """
    reference = np.asarray(reference_image)
    capture = np.asarray(capture_image)
    if reference.ndim == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    if capture.ndim == 3:
        capture = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    if reference.ndim != 2 or capture.ndim != 2:
        raise ValueError("Images must be grayscale or BGR")
    image_points = np.asarray(reference_points_image, dtype=np.float32)
    bed_points = np.asarray(reference_points_bed, dtype=np.float32)
    if image_points.shape != bed_points.shape or image_points.ndim != 2 or image_points.shape[1] != 2 or len(image_points) < 4:
        raise ValueError("At least four paired image-to-bed registration points are required")
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist = np.asarray(distortion, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsic matrix")
    if not np.isfinite(dist).all():
        raise ValueError("Invalid camera distortion")
    registration, _ = cv2.findHomography(image_points, bed_points, 0)
    if registration is None:
        raise ValueError("Sheet registration points are degenerate")

    detector = cv2.SIFT_create()
    reference_keypoints, reference_descriptors = detector.detectAndCompute(reference, None)
    capture_keypoints, capture_descriptors = detector.detectAndCompute(capture, None)
    if reference_descriptors is None or capture_descriptors is None:
        return _reject("Insufficient visible sheet features")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(reference_descriptors, capture_descriptors, k=2)
    matches = [
        pair[0] for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(matches) < MIN_MATCHES:
        return _reject("Insufficient distinct sheet features", match_count=len(matches))

    ref_pixels = np.float32([reference_keypoints[m.queryIdx].pt for m in matches])
    capture_pixels = np.float32([capture_keypoints[m.trainIdx].pt for m in matches])
    homography, homography_mask = cv2.findHomography(
        ref_pixels, capture_pixels, cv2.RANSAC, 4.0
    )
    if homography is None or homography_mask is None:
        return _reject("Sheet features have no consistent geometry", match_count=len(matches))
    geometric_inliers = homography_mask.ravel().astype(bool)
    if int(geometric_inliers.sum()) < MIN_MATCHES:
        return _reject("Too few geometrically consistent sheet features",
                       match_count=len(matches), inlier_count=int(geometric_inliers.sum()))
    ref_inliers = ref_pixels[geometric_inliers]
    capture_inliers = capture_pixels[geometric_inliers]
    coverage = cv2.contourArea(cv2.convexHull(ref_inliers)) / float(reference.shape[0] * reference.shape[1])
    if coverage < MIN_COVERAGE:
        return _reject("Sheet feature matches cover too little area",
                       match_count=len(matches), inlier_count=len(ref_inliers),
                       inlier_coverage=float(coverage))

    xy_bed = cv2.perspectiveTransform(ref_inliers.reshape(-1, 1, 2), registration).reshape(-1, 2)
    object_points = np.column_stack([xy_bed, np.zeros(len(xy_bed))]).astype(np.float32)
    ok, rvec, tvec, pnp_inlier_indices = cv2.solvePnPRansac(
        object_points, capture_inliers, matrix, dist,
        iterationsCount=500, reprojectionError=4.0, confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or pnp_inlier_indices is None or len(pnp_inlier_indices) < MIN_MATCHES:
        return _reject("Planar camera pose could not be verified",
                       match_count=len(matches), inlier_count=len(ref_inliers),
                       inlier_coverage=float(coverage))
    selected = pnp_inlier_indices.ravel()
    object_selected = object_points[selected]
    capture_selected = capture_inliers[selected]
    rvec, tvec = cv2.solvePnPRefineLM(
        object_selected, capture_selected, matrix, dist, rvec, tvec
    )
    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3)
    camera_center = -rotation.T @ translation
    camera_depths = (rotation @ object_selected.T + translation[:, None])[2]
    if camera_center[2] <= 0 or np.any(camera_depths <= 0):
        return _reject("Camera pose is outside the physically valid half-space",
                       match_count=len(matches), inlier_count=len(selected),
                       inlier_coverage=float(coverage))
    projected, _ = cv2.projectPoints(object_selected, rvec, tvec, matrix, dist)
    errors = np.linalg.norm(projected.reshape(-1, 2) - capture_selected, axis=1)
    median = float(np.median(errors))
    p95 = float(np.percentile(errors, 95))
    quality = {
        "match_count": len(matches),
        "inlier_count": len(selected),
        "inlier_coverage": float(coverage),
        "reprojection_median_px": median,
        "reprojection_p95_px": p95,
    }
    if p95 > MAX_REPROJECTION_P95_PX:
        return _reject("Reprojection error exceeds the pose gate", **quality)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return {
        "status": "accepted",
        "T_camera_from_bed": transform.tolist(),
        "quality": quality,
    }
