"""Refine an initial metric pose using observed bright, dashed plate lines.

Inputs are an already rectified pinhole image and registered 3D grid segments.
An initial pose is required to disambiguate the repetitive grid. This helper
does not estimate intrinsics, sheet geometry, or arbitrary image homographies.
"""

import cv2
import numpy as np


def _project(lines, matrix, parameters):
    rotation, _ = cv2.Rodrigues(parameters[:3])
    xyz = lines @ rotation.T + parameters[3:]
    if np.any(xyz[..., 2] <= 0):
        raise ValueError("Grid lies behind camera")
    xy = xyz[..., :2] / xyz[..., 2:3]
    return xy * [matrix[0, 0], matrix[1, 1]] + matrix[:2, 2]


def _detect_line(image, response, endpoints, radius):
    start, end = endpoints
    length = np.linalg.norm(end-start)
    if length < 50:
        return None
    tangent = (end-start)/length
    normal = np.array([-tangent[1], tangent[0]])
    along = np.linspace(.003, .997, int(np.clip(np.ceil(length*1.4), 160, 650)))
    across = np.arange(-radius, radius+.25, .5)
    samples = start + along[:, None, None]*(end-start) + across[None, :, None]*normal
    profiles = cv2.remap(response, samples[..., 0].astype(np.float32),
                         samples[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    profiles = cv2.GaussianBlur(profiles, (3, 1), 0)
    maxima = profiles.argmax(axis=1)
    rows = np.arange(len(along))
    peaks = profiles[rows, maxima]
    points = samples[rows, maxima]
    # Refine the ridge location below the half-pixel profile sampling step.
    left = profiles[rows, np.maximum(maxima-1, 0)].astype(float)
    right = profiles[rows, np.minimum(maxima+1, len(across)-1)].astype(float)
    curvature = left-2*peaks.astype(float)+right
    shift = np.zeros(len(points))
    curved = curvature < -1e-6
    shift[curved] = np.clip(.5*(left[curved]-right[curved])/curvature[curved], -.5, .5)
    points = points + .5*shift[:, None]*normal
    raw = cv2.remap(image, points[:, 0].astype(np.float32)[:, None],
                   points[:, 1].astype(np.float32)[:, None], cv2.INTER_LINEAR).ravel()
    good = ((peaks > 8) & (peaks-np.median(profiles, axis=1) > 6)
            & (maxima > 1) & (maxima < len(across)-2) & (raw < 245))
    points = points[good]
    if len(points) < 20:
        return None
    # Reject crossings, lettering, glare, and scratches away from the long line.
    for _ in range(3):
        vx, vy, x, y = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        distances = (points-[x, y]) @ [-vy, vx]
        points = points[np.abs(distances) < 1.5]
        if len(points) < 20:
            return None
    span = np.ptp((points-start) @ tangent)/length
    if span < .45:
        return None
    return points


def detect_grid_lines(image, projected_segments, search_radius_px=10):
    """Locate dash support over full segments, independently of pose fitting.

    Returned image observations can be split into fitting and validation sets
    or transformed through different explicitly selected camera models.
    """
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    segments = np.asarray(projected_segments, dtype=float)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected an 8-bit image")
    if (segments.ndim != 3 or segments.shape[1:] != (2, 2)
            or not np.isfinite(segments).all() or search_radius_px < 2):
        raise ValueError("Invalid projected grid segments or search radius")
    response = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    detections = []
    for index, segment in enumerate(segments):
        points = _detect_line(gray, response, segment, search_radius_px)
        if points is None:
            continue
        delta = segment[1]-segment[0]
        fraction = (points-segment[0]) @ delta/np.dot(delta, delta)
        detections.append({"line_index": index, "points_image_px": points.tolist(),
                           "support_fraction_range": [float(fraction.min()), float(fraction.max())]})
    return detections


def refine_pose_from_grid(pinhole_image, grid_lines_bed_m, camera_matrix,
                          initial_transform, search_radius_px=10, observed_lines=None):
    """Fit one rigid pose to multiple grid lines in a pinhole camera image.

    Bright dash centers are located near each predicted segment. A robust fit
    minimizes their perpendicular distance to projected 3D lines, weighting
    each line equally. Right angles and parallelism come from the registered
    metric grid, not from forcing 90-degree angles in the perspective image.
    Weak evidence or an implausible correction leaves the input pose intact.
    Optional observed_lines provide fixed, already detected pinhole points;
    supplying them prevents model-dependent re-detection during comparisons.
    """
    image = np.asarray(pinhole_image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    matrix = np.asarray(camera_matrix, dtype=float)
    lines = np.asarray(grid_lines_bed_m, dtype=float)
    transform = np.asarray(initial_transform, dtype=float)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected an 8-bit pinhole image")
    if lines.ndim != 3 or lines.shape[1:] != (2, 3) or len(lines) < 4:
        raise ValueError("At least four registered 3D line segments are required")
    if not np.isfinite(lines).all() or not np.allclose(lines[..., 2], 0):
        raise ValueError("Grid must lie on the finite bed Z=0 plane")
    lengths = np.linalg.norm(lines[:, 1]-lines[:, 0], axis=1)
    if np.any(lengths < 1e-6) or matrix.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("Invalid grid, camera matrix, or pose")
    initial_rvec, _ = cv2.Rodrigues(transform[:3, :3])
    initial = np.r_[initial_rvec.ravel(), transform[:3, 3]]
    unchanged = {"status": "unchanged", "T_camera_from_bed": transform.tolist()}
    projections = _project(lines, matrix, initial)
    if observed_lines is None:
        observed_lines = [dict(item, points_pinhole_px=item["points_image_px"])
                          for item in detect_grid_lines(image, projections, search_radius_px)]
    groups = []
    seen = set()
    for item in observed_lines:
        index = item["line_index"]
        points = np.asarray(item["points_pinhole_px"], dtype=float)
        if (not isinstance(index, (int, np.integer)) or index < 0 or index >= len(lines)
                or index in seen or points.ndim != 2 or points.shape[1] != 2
                or len(points) < 2 or not np.isfinite(points).all()):
            raise ValueError("Invalid or duplicate line observations")
        seen.add(index)
        groups.append((index, points))
    detections = [{"line_index": int(i), "points_pinhole_px": p.tolist()} for i, p in groups]
    unchanged["detected_lines"] = detections
    if len(groups) < 4:
        return {**unchanged, "reason": "Too few supported grid lines"}
    directions = np.array([lines[i, 1, :2]-lines[i, 0, :2] for i, _ in groups])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    _, singular, _ = np.linalg.svd(directions, full_matrices=False)
    if singular[-1] < .6:
        return {**unchanged, "reason": "Grid evidence covers only one direction"}

    def residuals(parameters):
        projected = _project(lines, matrix, parameters)
        errors = []
        for index, points in groups:
            a, b = projected[index]
            delta = b-a
            normal = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
            errors.append((points-a) @ normal)
        return errors

    # Equalize spatial bins within each line so a densely sampled center cannot
    # overwhelm the endpoints. Each line still receives equal total weight.
    weights_by_line = []
    for index, points in groups:
        a, b = projections[index]
        fraction = (points-a) @ (b-a)/np.dot(b-a, b-a)
        bins = np.clip((fraction*8).astype(int), 0, 7)
        counts = np.bincount(bins, minlength=8)
        weights_by_line.append(1/np.sqrt(np.count_nonzero(counts)*counts[bins]))
    group_weights = np.concatenate(weights_by_line)

    def objective(residual):
        absolute = np.abs(residual)
        huber = np.where(absolute <= 1.5, .5*residual**2, 1.5*(absolute-.75))
        return float(np.sum(huber*group_weights**2))

    parameters = initial.copy()
    before = np.concatenate(residuals(initial))
    for _ in range(25):
        residual = np.concatenate(residuals(parameters))
        weights = group_weights*np.sqrt(np.minimum(1, 1.5/np.maximum(np.abs(residual), 1e-9)))
        jacobian = np.empty((len(residual), 6))
        for column in range(6):
            trial = parameters.copy()
            trial[column] += 1e-6
            jacobian[:, column] = (np.concatenate(residuals(trial))-residual)/1e-6
        delta, _, rank, _ = np.linalg.lstsq(jacobian*weights[:, None], -residual*weights, rcond=None)
        if rank < 6:
            return {**unchanged, "reason": "Grid constraints are degenerate"}
        improved = False
        for scale in [1, .5, .25, .125, .0625]:
            candidate = parameters + scale*delta
            try:
                cost = objective(np.concatenate(residuals(candidate)))
            except ValueError:
                continue
            if cost < objective(residual):
                parameters = candidate
                improved = True
                break
        if not improved or np.linalg.norm(scale*delta) < 1e-9:
            break
    rotation, _ = cv2.Rodrigues(parameters[:3])
    new_center = -rotation.T @ parameters[3:]
    old_center = -transform[:3, :3].T @ transform[:3, 3]
    angle, _ = cv2.Rodrigues(rotation @ transform[:3, :3].T)
    after = np.concatenate(residuals(parameters))
    quality = {
        "line_count": len(groups), "dash_sample_count": len(after),
        "grid_distance_median_before_px": float(np.median(np.abs(before))),
        "grid_distance_p95_before_px": float(np.percentile(np.abs(before), 95)),
        "grid_distance_median_after_px": float(np.median(np.abs(after))),
        "grid_distance_p95_after_px": float(np.percentile(np.abs(after), 95)),
        "camera_shift_mm": float(np.linalg.norm(new_center-old_center)*1000),
        "rotation_change_deg": float(np.rad2deg(np.linalg.norm(angle))),
    }
    if (new_center[2] <= 0 or quality["camera_shift_mm"] > 30
            or quality["rotation_change_deg"] > 5
            or quality["grid_distance_p95_after_px"] > 3
            or objective(after) >= objective(before)):
        return {**unchanged, "reason": "Grid correction failed quality gates", "quality": quality}
    refined = np.eye(4)
    refined[:3, :3] = rotation
    refined[:3, 3] = parameters[3:]
    return {"status": "refined", "T_camera_from_bed": refined.tolist(),
            "quality": quality, "detected_lines": detections}
