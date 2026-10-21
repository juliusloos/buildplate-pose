"""Estimate and validate the provisional MK3 pose using SDK-correct D405 pixels.

Run from the checkout with PYTHONPATH=src. Source rotation is an explicit job
input; the calibration's historical saved rotation is never used as a default.
"""

import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np

from buildplate_pose.d405 import validate_calibration
from buildplate_pose.grid import detect_grid_lines, refine_pose_from_grid
from buildplate_pose.realsense import (
    saved_pixels_to_pinhole, pinhole_pixels_to_saved, rectify_saved_image,
)


OUTLINE = np.float32([[12, 36], [234, 36], [246, 52], [836, 52],
                      [848, 36], [1067, 36], [1067, 1044], [12, 1044]])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def project_pinhole(points_m, transform, matrix):
    points = np.asarray(points_m) @ transform[:3, :3].T + transform[:3, 3]
    if np.any(points[:, 2] <= 0):
        raise ValueError("Projected geometry is behind the camera")
    return points[:, :2]/points[:, 2:] * [matrix[0, 0], matrix[1, 1]] + matrix[:2, 2]


def error_summary(errors):
    values = np.asarray(errors)
    return {"sample_count": len(values),
            "median_px": float(np.median(values)) if len(values) else None,
            "p95_px": float(np.percentile(values, 95)) if len(values) else None}


def legacy_mappings(camera):
    """Reproduce the old bug ONLY to score the previous output on common points."""
    matrix = np.asarray(camera["camera_matrix"])
    focal = np.array([matrix[0, 0], matrix[1, 1]])
    limits = np.array(camera["capture_stream"]["image_size"])-1
    k1, k2, p1, p2, k3 = camera["distortion"]

    def polynomial(xy):
        x, y = xy.T
        r = x*x+y*y
        radial = 1+k1*r+k2*r*r+k3*r*r*r
        return np.column_stack([x*radial+2*p1*x*y+p2*(r+2*x*x),
                                y*radial+p1*(r+2*y*y)+2*p2*x*y])

    def forward(saved):
        native = limits-saved if camera["saved_frame_rotation_degrees"] == 180 else saved
        return polynomial((native-matrix[:2, 2])/focal)*focal+matrix[:2, 2]

    def backward(pinhole):
        ideal = (pinhole-matrix[:2, 2])/focal
        distorted = ideal.copy()
        for _ in range(30):
            distorted += ideal-polynomial(distorted)
        native = distorted*focal+matrix[:2, 2]
        return limits-native if camera["saved_frame_rotation_degrees"] == 180 else native

    return forward, backward


def assess(observations, grid, matrix, transform, to_pinhole, to_saved, held_out_only=False):
    """Score identical raw-image observations, including the formerly omitted left end."""
    endpoints = project_pinhole(grid.reshape(-1, 3), transform, matrix).reshape(-1, 2, 2)
    errors_all, errors_left, errors_top_left = [], [], []
    per_line = []
    for item in observations:
        index = item["line_index"]
        pixels = np.array(item["points_image_px"])
        fractions = np.array(item["sample_fractions"])
        selected = np.array(item["held_out_mask"]) if held_out_only else np.ones(len(pixels), bool)
        pixels, fractions = pixels[selected], fractions[selected]
        ideal = to_pinhole(pixels)
        start, end = endpoints[index]
        delta = end-start
        nearest = start + (((ideal-start) @ delta)/np.dot(delta, delta))[:, None]*delta
        errors = np.linalg.norm(to_saved(nearest)-pixels, axis=1)
        errors_all.extend(errors.tolist())
        if index >= 4:
            left = errors[fractions < .2]
            errors_left.extend(left.tolist())
            if index == 4:
                errors_top_left.extend(left.tolist())
        per_line.append({"line_index": index, **error_summary(errors)})
    return {"all": error_summary(errors_all), "left_horizontal": error_summary(errors_left),
            "top_left": error_summary(errors_top_left), "per_line": per_line}


def draw_pose(path, transform, outer_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    center = -transform[:3, :3].T @ transform[:3, 3]*1000
    forward = transform[:3, :3].T @ [0., 0., 1.]
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    closed = np.vstack([outer_m, outer_m[0]])*1000
    ax.plot(closed[:, 0], closed[:, 1], np.zeros(len(closed)), color="#475569")
    ax.scatter(*center, color="#c65d16", s=65)
    ax.quiver(*center, *(forward*70), color="#c65d16", linewidth=2)
    for vector, color in [([50, 0, 0], "red"), ([0, 50, 0], "green"), ([0, 0, 50], "blue")]:
        ax.quiver(0, 0, 0, *vector, color=color)
    ax.set(xlabel="X [mm]", ylabel="Y [mm]", zlabel="Z [mm]",
           xlim=(-50, 300), ylim=(-50, 280), zlim=(0, 350),
           title="SDK-corrected grid pose · provisional physical registration")
    ax.set_box_aspect((350, 330, 350))
    ax.view_init(elev=25, azim=-62)
    fig.text(.08, .04, f"Camera: {center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f} mm; calibration applicability unverified")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-rotation-degrees", required=True, type=int, choices=[0, 180])
    parser.add_argument("--camera-stream", default="d405_top_yuyv")
    parser.add_argument("--reference-image", default="prusamk3smoothpei.webp")
    parser.add_argument("--landmarks", default="calibration/mk3_smooth_pei_reference_landmarks_2026-10-05.json")
    parser.add_argument("--camera-profile", default="calibration/realsense_inverse_brown_conrady_2026-10-05.json")
    parser.add_argument("--previous-pose", help="Optional old JSON for an explicitly labelled legacy-mapping comparison")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.capture_image).resolve()
    camera = json.loads(Path(args.camera_profile).read_text())
    validate_calibration(camera, args.camera_stream)
    historical_rotation = camera.get("saved_frame_rotation_degrees")
    camera["saved_frame_rotation_degrees"] = args.source_rotation_degrees
    landmarks = json.loads(Path(args.landmarks).read_text())
    reference_path = Path(args.reference_image)
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == landmarks["reference_image_sha256"]
    image = cv2.imread(str(source))
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    width, height = camera["capture_stream"]["image_size"]
    if image is None or image.shape[:2] != (height, width) or reference is None:
        raise ValueError("Unreadable images or capture resolution differs from calibration")
    matrix = np.asarray(camera["camera_matrix"])
    keys = ["lower_left", "lower_right", "upper_right", "upper_left"]
    corners = np.float32([landmarks["sheet_outer_virtual_corners_image_px"][k] for k in keys])
    w, d = landmarks["sheet_outer_size_mm"]
    registration = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [w, 0], [w, d], [0, d]]))
    origin = cv2.perspectiveTransform(np.float32(landmarks["bed_origin_image_px"]).reshape(1, 1, 2), registration).reshape(2)

    def bed(reference_pixels):
        xy = cv2.perspectiveTransform(np.float32(reference_pixels).reshape(-1, 1, 2), registration).reshape(-1, 2)-origin
        return np.column_stack([xy, np.zeros(len(xy))])/1000

    mask = np.zeros_like(reference)
    cv2.fillPoly(mask, [OUTLINE.astype(np.int32)], 255)
    detector = cv2.SIFT_create(nfeatures=6000)
    kr, dr = detector.detectAndCompute(reference, mask)
    kc, dc = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    if dr is None or dc is None:
        raise RuntimeError("Images have insufficient sheet features")
    matches = [a for a, b in cv2.BFMatcher().knnMatch(dr, dc, k=2) if a.distance < .75*b.distance]
    if len(matches) < 15:
        raise RuntimeError("Too few distinct sheet matches")
    src = np.float32([kr[m.queryIdx].pt for m in matches])
    dst = np.float32([kc[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(0)
    _, hm = cv2.findHomography(src, dst, cv2.RANSAC, 4)
    if hm is None or int(hm.sum()) < 15:
        raise RuntimeError("Insufficient feature geometry for a grid initialization")
    src, dst = src[hm.ravel().astype(bool)], dst[hm.ravel().astype(bool)]
    if cv2.contourArea(cv2.convexHull(src))/cv2.countNonZero(mask) < .07:
        raise RuntimeError("Feature matches cover too little sheet area")
    obj = bed(src).astype(np.float32)
    observed = saved_pixels_to_pinhole(dst, camera).astype(np.float32)
    cv2.setRNGSeed(0)
    ok, rvec, tvec, indices = cv2.solvePnPRansac(obj, observed, matrix, np.zeros(5), iterationsCount=500,
                                               reprojectionError=4, confidence=.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or indices is None or len(indices) < 12:
        raise RuntimeError("PnP initialization rejected")
    indices = indices.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[indices], observed[indices], matrix, np.zeros(5), rvec, tvec)
    initial = np.eye(4)
    initial[:3, :3] = cv2.Rodrigues(rvec)[0]
    initial[:3, 3] = tvec.ravel()
    if (-initial[:3, :3].T @ initial[:3, 3])[2] <= 0:
        raise RuntimeError("Camera is below the sheet")
    grid_px = [[[x, 114], [x, 952]] for x in [229, 437, 645, 853]]
    grid_px += [[[19, y], [1062, y]] for y in [114, 324, 533, 743, 952]]
    grid = np.array([bed(line) for line in grid_px])
    segments = pinhole_pixels_to_saved(project_pinhole(grid.reshape(-1, 3), initial, matrix), camera).reshape(-1, 2, 2)
    observations = detect_grid_lines(image, segments, 10)
    if len(observations) != 9:
        raise RuntimeError(f"This replay requires all nine grid lines; found {len(observations)}")
    for item in observations:
        segment = segments[item["line_index"]]
        delta = segment[1]-segment[0]
        fractions = (np.array(item["points_image_px"])-segment[0]) @ delta/np.dot(delta, delta)
        item["sample_fractions"] = fractions.tolist()
        # Withhold entire short spatial blocks, rather than adjacent duplicate pixels.
        item["held_out_mask"] = ((np.floor(np.clip(fractions, 0, .9999)*32).astype(int) % 4) == 1).tolist()

    def observations_for(profile, training_only):
        groups = []
        for item in observations:
            points = np.array(item["points_image_px"])
            if training_only:
                points = points[~np.array(item["held_out_mask"])]
            groups.append({"line_index": item["line_index"],
                           "points_pinhole_px": saved_pixels_to_pinhole(points, profile).tolist()})
        return groups

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    validation = {}
    for name, profile in [("sdk_factory", camera), ("zero_distortion_hypothesis", dict(camera, distortion=[0.]*5))]:
        fitted = refine_pose_from_grid(gray, grid, matrix, initial, observed_lines=observations_for(profile, True))
        validation[name] = {"status": fitted["status"], "fit_point_count": sum(len(g["points_pinhole_px"]) for g in observations_for(profile, True))}
        if fitted["status"] == "refined":
            validation[name]["held_out_errors_saved_px"] = assess(
                observations, grid, matrix, np.array(fitted["T_camera_from_bed"]),
                lambda p: saved_pixels_to_pinhole(p, profile), lambda p: pinhole_pixels_to_saved(p, profile), True)
    if validation["sdk_factory"]["status"] != "refined":
        raise RuntimeError("Corrected SDK fit did not pass held-out preparation")
    fitted = refine_pose_from_grid(gray, grid, matrix, initial, observed_lines=observations_for(camera, False))
    if fitted["status"] != "refined":
        raise RuntimeError("Final corrected pose failed quality gates")
    transform = np.array(fitted["T_camera_from_bed"])
    errors = assess(observations, grid, matrix, transform,
                    lambda p: saved_pixels_to_pinhole(p, camera), lambda p: pinhole_pixels_to_saved(p, camera))
    previous_comparison = None
    if args.previous_pose:
        previous = json.loads(Path(args.previous_pose).read_text())["results"][0]
        if previous["capture_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("Previous pose describes a different capture")
        old_to, old_back = legacy_mappings(camera)
        previous_comparison = assess(observations, grid, matrix, np.array(previous["T_native_camera_from_bed"]), old_to, old_back)
    # Score the original feature inliers again, retaining the tradeoff in the JSON.
    sift_initial = np.linalg.norm(project_pinhole(obj[indices], initial, matrix)-observed[indices], axis=1)
    sift_final = np.linalg.norm(project_pinhole(obj[indices], transform, matrix)-observed[indices], axis=1)
    if np.percentile(sift_final, 95) > 5:
        raise RuntimeError("Grid fit degraded independent feature geometry beyond its gate")
    center = -transform[:3, :3].T @ transform[:3, 3]
    result = {
        "schema_version": 1, "status": "provisional", "frame": 1, "capture": source.name,
        "image": "camera_pose_3d.png", "pose_overlay": "pose_overlay.png",
        "grid_overlay": "grid_constraints.png", "left_detail": "left_detail.png",
        "capture_path": str(source), "capture_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "camera_position_bed_mm": (center*1000).tolist(), "T_native_camera_from_bed": transform.tolist(),
        "initial_feature_pose": initial.tolist(), "pnp_inliers": len(indices),
        "feature_reprojection_p95_before_px": float(np.percentile(sift_initial, 95)),
        "feature_reprojection_p95_after_px": float(np.percentile(sift_final, 95)),
        "grid_quality": fitted["quality"], "all_observed_grid_errors_saved_px": errors,
        "validation": validation, "previous_output_on_same_observations": previous_comparison,
        "source_rotation_degrees": args.source_rotation_degrees,
        "source_rotation_status": "explicit job input; capture pipeline not independently verified",
        "historical_factory_saved_rotation_degrees": historical_rotation,
        "camera_calibration": str(Path(args.camera_profile).resolve()),
        "camera_calibration_sha256": hashlib.sha256(Path(args.camera_profile).read_bytes()).hexdigest(),
        "camera_model_implementation": "librealsense inverse Brown-Conrady via buildplate_pose.d405",
        "reference_grid_lines_image_px": grid_px,
        "bed_frame": "provisional product-photo printed-rectangle lower-left origin; G-code offset unmeasured",
        "physical_validation": False,
        "limitations": ["Exact camera identity and JPEG processing remain unverified for this capture.",
                       "Reference grid and outer outline are provisional product-photo geometry.",
                       "Final pose uses all observations; held-out errors come from the separate training-only fits."]}
    write_json(out/"poses.json", {"results": [result]})
    write_json(out/"grid_fit.json", {"observations_in_original_image": observations, "refinement": fitted,
                                    "validation": validation, "evaluation": errors})
    write_json(out/"camera_used.json", camera)
    write_json(out/"run_settings.json", vars(args))
    (out/"selection.csv").write_text(f"frame,capture,status\n1,{source.name},provisional\n")

    def project(points):
        return pinhole_pixels_to_saved(project_pinhole(np.asarray(points), transform, matrix), camera)

    def footer(canvas, lines):
        canvas = cv2.copyMakeBorder(canvas, 0, 32*len(lines)+12, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (12, height+28+32*i), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    diagnostic = image.copy()
    for item in observations:
        for point, held in zip(item["points_image_px"], item["held_out_mask"]):
            color = (0, 165, 255) if held else (0, 255, 0)
            cv2.circle(diagnostic, tuple(np.rint(point).astype(int)), 1, color, -1)
        segment = grid[item["line_index"]]
        curve = project(np.linspace(*segment, 400))
        cv2.polylines(diagnostic, [np.rint(curve).astype(np.int32)], False, (255, 0, 255), 1, cv2.LINE_AA)
    assert cv2.imwrite(str(out/"grid_constraints.png"), footer(diagnostic, ["SDK-corrected grid | green: fitting support | orange: validation support | magenta: final pose projection"]))
    overlay = image.copy()
    outer_m = bed(OUTLINE)
    def contour(points):
        dense = np.concatenate([np.linspace(a, b, 70, endpoint=False) for a, b in zip(points, np.roll(points, -1, axis=0))])
        return np.rint(project(dense)).astype(np.int32)
    outer = contour(outer_m)
    tint = overlay.copy()
    cv2.fillPoly(tint, [outer], (235, 195, 30))
    overlay = cv2.addWeighted(tint, .13, overlay, .87, 0)
    cv2.polylines(overlay, [outer], True, (255, 230, 0), 2, cv2.LINE_AA)
    inner = bed([landmarks["print_area_border_corners_image_px"][k] for k in keys])
    cv2.polylines(overlay, [contour(inner)], True, (0, 225, 255), 2, cv2.LINE_AA)
    axes = project([[0, 0, 0], [.045, 0, 0], [0, .045, 0], [0, 0, .045]])
    start = tuple(np.rint(axes[0]).astype(int))
    for endpoint, label, color in zip(axes[1:], "XYZ", [(20, 20, 255), (20, 240, 20), (255, 90, 20)]):
        endpoint = tuple(np.rint(endpoint).astype(int))
        cv2.arrowedLine(overlay, start, endpoint, color, 3, cv2.LINE_AA, tipLength=.15)
        cv2.putText(overlay, label, (endpoint[0]+5, endpoint[1]-7), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    assert cv2.imwrite(str(out/"pose_overlay.png"), footer(overlay, ["SDK-corrected 3D pose | cyan: provisional sheet outline | yellow: provisional print border | RGB: X/Y/Z", "The grid is image-validated; physical dimensions, origin and calibration applicability remain provisional."]))
    assert cv2.imwrite(str(out/"rectified.png"), rectify_saved_image(image, camera))
    draw_pose(out/"camera_pose_3d.png", transform, bed(corners))
    # Close view makes residual errors on the previously omitted left end reviewable.
    x0 = max(0, int(np.min(segments[4:, :, 0]))-25)
    x1 = min(width, x0+300)
    assert cv2.imwrite(str(out/"left_detail.png"), diagnostic[:, x0:x1])
    held = validation["sdk_factory"]["held_out_errors_saved_px"]
    left_text = f"{held['left_horizontal']['p95_px']:.2f} px"
    old_text = (f"On all common left-side observations, previous → corrected p95 is "
                f"{previous_comparison['left_horizontal']['p95_px']:.2f} → "
                f"{errors['left_horizontal']['p95_px']:.2f} px. ") if previous_comparison else ""
    report = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Corrected D405 grid pose</title><style>body{font:16px system-ui;max-width:1300px;margin:auto;padding:24px;color:#18212c}img{max-width:100%;height:auto;border:1px solid #d8dee5}section{margin:32px 0}code{overflow-wrap:anywhere}</style>'
    report += f'<h1>Corrected D405 grid pose</h1><p><code>{html.escape(source.name)}</code></p><p>Fixed the earlier reversed distortion mapping using SDK-compatible projection/deprojection. Dense full-length observations and spatial balancing now include the left ends. Original-image errors use the same raw observations for every comparison.</p>'
    report += f'<p>Withheld-point p95: <strong>{held["all"]["p95_px"]:.2f} px overall</strong>, <strong>{left_text} on the left</strong>. {old_text}All-point output pose uses {fitted["quality"]["dash_sample_count"]} samples over nine lines. Physical calibration remains provisional.</p>'
    report += '<p><a href="poses.json">Pose and validation JSON</a> · <a href="grid_fit.json">All observations</a> · <a href="run_settings.json">Run settings</a></p>'
    for title, filename in [("Full-length dotted grid", "grid_constraints.png"), ("Left-side detail", "left_detail.png"), ("Plate projected from the corrected pose", "pose_overlay.png"), ("3D camera pose", "camera_pose_3d.png")]:
        report += f'<section><h2>{title}</h2><a href="{filename}"><img src="{filename}" alt="{title}"></a></section>'
    report += '</html>\n'
    (out/"index.html").write_text(report)
    print(json.dumps({"camera_position_bed_mm": result["camera_position_bed_mm"], "held_out_sdk": held,
                      "all_points": errors, "previous_same_points": previous_comparison,
                      "feature_p95": result["feature_reprojection_p95_after_px"]}, indent=2))
    print("Report:", (out/"index.html").resolve())


if __name__ == "__main__":
    main()
