"""
C4D2GS Lite — Cinema 4D Script 
=======================================
A single-script, parameter-only version of the C4D2GS plugin.

Usage
-----
1. Edit the parameters in the USER PARAMETERS section below.
2. Select the target object in your object manager.
3. Run this script from the Cinema 4D Script Manager
   (Extensions ▸ Script Manager).
4. Render the animation to produce the image sequence
5. Import the synthetic COLMAP data folder into your GS pipeline 


License: CC BY-NC 4.0
Author: Joel Tenenberg              
"""

import c4d
import math
import json
import os
import random
import bisect

# =============================================================================
# USER PARAMETERS — Edit these before running
# =============================================================================

# Camera rig setup
CAMERA_COUNT = 50               # Number of viewpoints around the object
SPHERE_RADIUS = 200.0           # Distance from object center to each camera

# Render output
# Must be set before execution.  Example: r"C:\renders\my_capture"
OUTPUT_PATH = r""
RESOLUTION_X = 1080
RESOLUTION_Y = 1080
FPS = 30
ENABLE_STRAIGHT_ALPHA = True    # If True, render with straight alpha.

# Exports
EXPORT_CAMERA_POSES_JSON = False    # Exports camera_poses.json (for nerfstudio)
EXPORT_COLMAP_DATA = True           # Exports synthetic COLMAP data files

# Sparse point cloud
SPARSE_POINT_COUNT = 20000      # Number of 3D points sampled from object surface
REPLACE_EXISTING_RIG = True     # Remove old GS_CameraRig before building new one

# =============================================================================
# INTERNALS — Do not edit below this line
# =============================================================================

_NUMERIC_CLEAN_EPS = 1e-10
_MAX_TRACK_OBS_PER_POINT = 12
_RIG_NAME = "GS_CameraRig"
_TARGET_NULL_NAME = "GS_Target"
_RENDER_CAM_NAME = "GS_RenderCam_Animated"


# ---------------------------------------------------------------------------
# Math / geometry helpers
# ---------------------------------------------------------------------------

def _normalize(v):
    length = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if length <= 0.0:
        return c4d.Vector(0, 1, 0)
    return c4d.Vector(v.x / length, v.y / length, v.z / length)


def _cross(a, b):
    return c4d.Vector(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def _dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


def _clean_small(value, eps=_NUMERIC_CLEAN_EPS):
    try:
        v = float(value)
    except Exception:
        return value
    return 0.0 if abs(v) < float(eps) else v


def _clean_vec3(v, eps=_NUMERIC_CLEAN_EPS):
    return [_clean_small(v.x, eps), _clean_small(v.y, eps), _clean_small(v.z, eps)]


def _clean_matrix_rows(rows, eps=_NUMERIC_CLEAN_EPS):
    return [[_clean_small(cell, eps) for cell in row] for row in rows]


def _copy_matrix(mg):
    out = c4d.Matrix()
    out.off = c4d.Vector(mg.off.x, mg.off.y, mg.off.z)
    out.v1  = c4d.Vector(mg.v1.x,  mg.v1.y,  mg.v1.z)
    out.v2  = c4d.Vector(mg.v2.x,  mg.v2.y,  mg.v2.z)
    out.v3  = c4d.Vector(mg.v3.x,  mg.v3.y,  mg.v3.z)
    return out


_FLIP_Y_MATRIX = None


def _get_flip_y_matrix():
    """Return a cached diag(1, -1, 1) C4D Matrix (constructed once)."""
    global _FLIP_Y_MATRIX
    if _FLIP_Y_MATRIX is None:
        m = c4d.Matrix()
        m.v1  = c4d.Vector(1,  0, 0)
        m.v2  = c4d.Vector(0, -1, 0)
        m.v3  = c4d.Vector(0,  0, 1)
        m.off = c4d.Vector(0,  0, 0)
        _FLIP_Y_MATRIX = m
    return _FLIP_Y_MATRIX


def _apply_world_flip_y(mat):
    """Return a copy of *mat* with Y negated on all columns and the offset."""
    out = c4d.Matrix()
    out.v1  = c4d.Vector(mat.v1.x,  -mat.v1.y,  mat.v1.z)
    out.v2  = c4d.Vector(mat.v2.x,  -mat.v2.y,  mat.v2.z)
    out.v3  = c4d.Vector(mat.v3.x,  -mat.v3.y,  mat.v3.z)
    out.off = c4d.Vector(mat.off.x, -mat.off.y, mat.off.z)
    return out


def _flip_y_vec(v):
    """Return *v* with its Y component negated (C4D world → COLMAP world)."""
    return c4d.Vector(v.x, -v.y, v.z)


def fibonacci_sphere_points(count):
    """Return *count* roughly-evenly-spaced unit vectors on a sphere (Fibonacci lattice)."""
    if count <= 0:
        return []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for i in range(count):
        # max(1, count-1) avoids division by zero when count == 1
        y = 1.0 - (2.0 * i) / float(max(1, count - 1))
        radius = max(0.0, 1.0 - y * y) ** 0.5
        theta = golden_angle * i
        points.append(c4d.Vector(math.sin(theta) * radius, y, math.cos(theta) * radius))
    return points


def look_at_matrix(camera_pos, target_pos, up_hint=None):
    """Camera-to-world matrix that looks toward *target_pos* (C4D -Z convention)."""
    if up_hint is None:
        up_hint = c4d.Vector(0, 1, 0)
    z_axis = _normalize(camera_pos - target_pos)
    if abs(_dot(z_axis, up_hint)) > 0.999:
        up_hint = c4d.Vector(0, 0, 1)
    x_axis = _normalize(_cross(up_hint, z_axis))
    y_axis = _normalize(_cross(z_axis, x_axis))
    mg = c4d.Matrix()
    mg.off = camera_pos
    mg.v1  = x_axis
    mg.v2  = y_axis
    mg.v3  = z_axis
    return mg


def matrix_to_rows(mg):
    """4 × 4 row-major camera-to-world matrix as a list of lists."""
    return [
        [mg.v1.x, mg.v2.x, mg.v3.x, mg.off.x],
        [mg.v1.y, mg.v2.y, mg.v3.y, mg.off.y],
        [mg.v1.z, mg.v2.z, mg.v3.z, mg.off.z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def nerf_matrix_to_rows(mg):
    """c2w matrix converted to NeRF/OpenGL convention (flip Y and Z columns)."""
    rows = matrix_to_rows(mg)
    for row in rows[:3]:
        row[1] *= -1.0
        row[2] *= -1.0
    return rows


def rotation_matrix_to_quaternion(r):
    """Return (qw, qx, qy, qz) from a 3 × 3 rotation matrix."""
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return 0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s
    elif (r[0][0] > r[1][1]) and (r[0][0] > r[2][2]):
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        return (r[2][1] - r[1][2]) / s, 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        return (r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        return (r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s


def c2w_to_colmap_extrinsics(mg):
    """Convert a C4D camera-to-world matrix to COLMAP extrinsics (q, t, R_w2c)."""
    mg1 = mg * _get_flip_y_matrix()    # undo camera-local Y flip
    mg2 = _apply_world_flip_y(mg1)     # undo world Y flip

    c_pos = mg2.off
    r_w2c = [
        [mg2.v1.x, mg2.v1.y, mg2.v1.z],
        [mg2.v2.x, mg2.v2.y, mg2.v2.z],
        [mg2.v3.x, mg2.v3.y, mg2.v3.z],
    ]
    tx = -(r_w2c[0][0] * c_pos.x + r_w2c[0][1] * c_pos.y + r_w2c[0][2] * c_pos.z)
    ty = -(r_w2c[1][0] * c_pos.x + r_w2c[1][1] * c_pos.y + r_w2c[1][2] * c_pos.z)
    tz = -(r_w2c[2][0] * c_pos.x + r_w2c[2][1] * c_pos.y + r_w2c[2][2] * c_pos.z)
    qw, qx, qy, qz = rotation_matrix_to_quaternion(r_w2c)
    return (qw, qx, qy, qz), (tx, ty, tz), r_w2c


def project_world_to_image(mg, world_point, world_normal, fx, fy, cx, cy,
                            require_front_facing=True):
    mg1 = mg * _get_flip_y_matrix()
    mg2 = _apply_world_flip_y(mg1)   # c2w in COLMAP frame

    p_col = _flip_y_vec(world_point)  # world → COLMAP
    if world_normal is not None:
        n_col = _flip_y_vec(world_normal)
        if require_front_facing:
            to_cam_col = _normalize(mg2.off - p_col)
            if _dot(to_cam_col, n_col) <= 0.0:
                return None

    local = (~mg2) * p_col
    x_cv = local.x
    y_cv = local.y  # already Y-down from flip
    z_cv = local.z  # forward
    if z_cv <= 1e-6:
        return None
    return (fx * (x_cv / z_cv)) + cx, (fy * (y_cv / z_cv)) + cy


def _project_in_colmap_frame(colmap_mg, colmap_inv_mg, world_point, world_normal,
                              fx, fy, cx, cy, require_front_facing=True):
    p_col = _flip_y_vec(world_point)
    if world_normal is not None:
        n_col = _flip_y_vec(world_normal)
        if require_front_facing:
            to_cam_col = _normalize(colmap_mg.off - p_col)
            if _dot(to_cam_col, n_col) <= 0.0:
                return None

    local = colmap_inv_mg * p_col
    z_cv = local.z
    if z_cv <= 1e-6:
        return None
    return (fx * (local.x / z_cv)) + cx, (fy * (local.y / z_cv)) + cy


def _cap_observations(candidates, max_count):
    if len(candidates) <= max_count:
        return candidates
    if max_count <= 0:
        return []
    out = []
    n = len(candidates)
    denom = float(max_count - 1) if max_count > 1 else 1.0
    for i in range(max_count):
        idx = int(round(i * (n - 1) / denom)) if max_count > 1 else 0
        out.append(candidates[idx])
    return out


# ---------------------------------------------------------------------------
# Object / bounding-box helpers
# ---------------------------------------------------------------------------

def _iter_hierarchy(op):
    node = op
    while node:
        yield node
        child = node.GetDown()
        if child:
            for sub in _iter_hierarchy(child):
                yield sub
        node = node.GetNext()


def _iter_cache_hierarchy(op):
    if op is None:
        return
    yield op
    deform = op.GetDeformCache()
    if deform is not None:
        for sub in _iter_hierarchy(deform):
            yield sub
    cache = op.GetCache()
    if cache is not None:
        for sub in _iter_hierarchy(cache):
            yield sub
    child = op.GetDown()
    if child is not None:
        for sub in _iter_hierarchy(child):
            yield sub


def center_of_object(op):
    """Geometry-based world-space bounding-box centre of *op*."""
    if op is None:
        return c4d.Vector(0)
    mn = mx = None
    for node in _iter_cache_hierarchy(op):
        try:
            rad = node.GetRad()
            if rad is None:
                continue
            if abs(rad.x) < 1e-9 and abs(rad.y) < 1e-9 and abs(rad.z) < 1e-9:
                continue
            mp = node.GetMp()
            mg = node.GetMg()
            corners = [
                c4d.Vector(mp.x - rad.x, mp.y - rad.y, mp.z - rad.z),
                c4d.Vector(mp.x + rad.x, mp.y - rad.y, mp.z - rad.z),
                c4d.Vector(mp.x - rad.x, mp.y + rad.y, mp.z - rad.z),
                c4d.Vector(mp.x + rad.x, mp.y + rad.y, mp.z - rad.z),
                c4d.Vector(mp.x - rad.x, mp.y - rad.y, mp.z + rad.z),
                c4d.Vector(mp.x + rad.x, mp.y - rad.y, mp.z + rad.z),
                c4d.Vector(mp.x - rad.x, mp.y + rad.y, mp.z + rad.z),
                c4d.Vector(mp.x + rad.x, mp.y + rad.y, mp.z + rad.z),
            ]
            for lp in corners:
                wp = lp * mg
                if mn is None:
                    mn = c4d.Vector(wp.x, wp.y, wp.z)
                    mx = c4d.Vector(wp.x, wp.y, wp.z)
                else:
                    mn.x = min(mn.x, wp.x); mx.x = max(mx.x, wp.x)
                    mn.y = min(mn.y, wp.y); mx.y = max(mx.y, wp.y)
                    mn.z = min(mn.z, wp.z); mx.z = max(mx.z, wp.z)
        except Exception:
            continue
    if mn is not None and mx is not None:
        return (mn + mx) * 0.5
    return op.GetMg().off


def find_object_by_name(start_obj, name):
    obj = start_obj
    while obj:
        if obj.GetName() == name:
            return obj
        child = obj.GetDown()
        if child:
            found = find_object_by_name(child, name)
            if found is not None:
                return found
        obj = obj.GetNext()
    return None


# ---------------------------------------------------------------------------
# Mesh sampling (sparse point cloud)
# ---------------------------------------------------------------------------

def _triangle_area(a, b, c):
    return _cross(b - a, c - a).GetLength() * 0.5


def _sample_on_triangle_with_normal(a, b, c):
    r1 = random.random()
    r2 = random.random()
    s1 = math.sqrt(r1)
    point = a * (1.0 - s1) + b * (s1 * (1.0 - r2)) + c * (s1 * r2)
    normal = _normalize(_cross(b - a, c - a))
    return point, normal


def _collect_world_triangles(root_obj):
    triangles = []
    for obj in _iter_cache_hierarchy(root_obj):
        if obj.CheckType(c4d.Opolygon):
            mg = obj.GetMg()
            pts = obj.GetAllPoints()
            if pts:
                wpts = [p * mg for p in pts]
                for poly in obj.GetAllPolygons():
                    triangles.append((wpts[poly.a], wpts[poly.b], wpts[poly.c]))
                    if poly.c != poly.d:
                        triangles.append((wpts[poly.a], wpts[poly.c], wpts[poly.d]))
    return triangles


def _make_editable(op):
    if not op or op.CheckType(c4d.Opolygon) or op.CheckType(c4d.Ospline):
        return op
    tmp_doc = c4d.documents.BaseDocument()
    clone = op.GetClone()
    tmp_doc.InsertObject(clone, None, None)
    clone.SetMg(op.GetMg())
    try:
        result = c4d.utils.SendModelingCommand(
            command=c4d.MCOMMAND_MAKEEDITABLE,
            list=[clone],
            mode=c4d.MODELINGCOMMANDMODE_ALL,
            doc=tmp_doc,
        )
    except Exception:
        return None
    return result[0] if result else None


def _get_current_state_object(doc, obj):
    try:
        result = c4d.utils.SendModelingCommand(
            command=c4d.MCOMMAND_CURRENTSTATETOOBJECT,
            list=[obj],
            mode=c4d.MODELINGCOMMANDMODE_ALL,
            doc=doc,
            flags=c4d.MODELINGCOMMANDFLAGS_NONE,
        )
    except Exception:
        return None
    return result[0] if result else None


def generate_sparse_points_from_surface(doc, target_obj, count=256):
    """Area-weighted sampling of *count* surface points with normals (world space)."""
    count = max(8, int(count))
    triangles = []
    for bake_fn in [
        lambda: _make_editable(target_obj),
        lambda: _get_current_state_object(doc, target_obj),
        lambda: target_obj,
    ]:
        try:
            baked = bake_fn()
            if baked is not None:
                triangles = _collect_world_triangles(baked)
        except Exception:
            pass
        if triangles:
            break
    if not triangles:
        return None

    areas, cumulative, running = [], [], 0.0
    for tri in triangles:
        ar = _triangle_area(*tri)
        if ar <= 1e-12:
            continue
        areas.append(tri)
        running += ar
        cumulative.append(running)

    if running <= 0.0 or not areas:
        return None

    out = []
    for _ in range(count):
        r = random.random() * running
        idx = bisect.bisect_left(cumulative, r)
        if idx >= len(areas):
            idx = len(areas) - 1
        out.append(_sample_on_triangle_with_normal(*areas[idx]))
    return out


def generate_sparse_points_in_core_volume(target_obj, target_pos, count=256,
                                          radius_factor=0.35):
    """Fallback: uniform random points inside a sphere at the object's centre."""
    if target_obj is None:
        return None
    count = max(8, int(count))
    try:
        rad = target_obj.GetRad()
        base_radius = max(1.0, math.sqrt(rad.x**2 + rad.y**2 + rad.z**2) * 2.5)
    except Exception:
        base_radius = 100.0
    core_radius = max(1.0, base_radius * max(0.05, float(radius_factor)))
    out = []
    for _ in range(count):
        u = random.random(); v = random.random(); w = random.random()
        theta = 2.0 * math.pi * u
        phi = math.acos(max(-1.0, min(1.0, 2.0 * v - 1.0)))
        r = core_radius * (w ** (1.0 / 3.0))
        sx = r * math.sin(phi) * math.cos(theta)
        sy = r * math.cos(phi)
        sz = r * math.sin(phi) * math.sin(theta)
        out.append((target_pos + c4d.Vector(sx, sy, sz), None))
    return out


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _normalize_path(path):
    p = os.path.expandvars(os.path.expanduser(str(path).strip()))
    if not p:
        return ""
    p = os.path.normpath(p)
    return p if os.path.isabs(p) else os.path.abspath(p)


def _output_folder():
    return _normalize_path(OUTPUT_PATH)


def _images_output_dir():
    return os.path.join(_output_folder(), "images")


def _render_output_pattern():
    return os.path.join(_images_output_dir(), "gs_")


def _frame_image_path(frame_index):
    return os.path.join(_images_output_dir(), "gs_{:04d}.png".format(frame_index))


def _pose_json_path():
    return os.path.join(_output_folder(), "camera_poses.json")


def _colmap_output_dir():
    return _output_folder()


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

def _intrinsics_from_camera(cam):
    focus_pid   = getattr(c4d, "CAMERA_FOCUS",        None)
    aperture_pid = getattr(c4d, "CAMERAOBJECT_APERTURE", None)
    if focus_pid is None or aperture_pid is None:
        return None
    try:
        focal_mm     = float(cam[focus_pid])
        aperture_w_mm = float(cam[aperture_pid])
    except Exception:
        return None
    if focal_mm <= 0.0 or aperture_w_mm <= 0.0 or RESOLUTION_X <= 0 or RESOLUTION_Y <= 0:
        return None
    aperture_h_mm = aperture_w_mm * (float(RESOLUTION_Y) / float(RESOLUTION_X))
    fx = (focal_mm / aperture_w_mm) * float(RESOLUTION_X)
    fy = (focal_mm / aperture_h_mm) * float(RESOLUTION_Y)
    model = "SIMPLE_PINHOLE" if abs(fx - fy) <= 1e-6 else "PINHOLE"
    return {"model": model, "fx": fx, "fy": fy,
            "cx": RESOLUTION_X * 0.5, "cy": RESOLUTION_Y * 0.5,
            "source": "render_camera"}


def _get_intrinsics(render_cam=None):
    if render_cam is not None:
        auto = _intrinsics_from_camera(render_cam)
        if auto is not None:
            return auto
    # Fallback: compute a reasonable pinhole from the image size.
    # Mirrors the C4D default camera: 36 mm focal length on a 36 mm sensor,
    # yielding fx = fy = image_width_pixels (square pixels, symmetric FOV).
    fx = float(RESOLUTION_X)
    return {
        "model": "SIMPLE_PINHOLE",
        "fx": fx, "fy": fx,
        "cx": RESOLUTION_X * 0.5, "cy": RESOLUTION_Y * 0.5,
        "source": "computed",
    }


# ---------------------------------------------------------------------------
# Export: camera poses JSON (nerfstudio format)
# ---------------------------------------------------------------------------

def export_camera_poses_json(world_points, target_pos, render_cam=None,
                              camera_matrices=None):
    """Write camera_poses.json to OUTPUT_PATH.  Returns the written path."""
    if not world_points:
        return None
    intrinsics = _get_intrinsics(render_cam)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    w, h   = int(RESOLUTION_X), int(RESOLUTION_Y)

    frames = []
    for i, world_pos in enumerate(world_points):
        mg = camera_matrices[i] if (camera_matrices and i < len(camera_matrices)) \
             else look_at_matrix(world_pos, target_pos)
        hpb = c4d.utils.MatrixToHPB(mg)
        rel_path = "images/{}".format(os.path.basename(_frame_image_path(i)))
        frames.append({
            "frame": i,
            "camera_name": "GS_Cam_{:04d}".format(i),
            "file_path": "./{}".format(rel_path.replace("\\", "/")),
            "position": _clean_vec3(world_pos),
            "rotation_hpb_rad": [hpb.x, hpb.y, hpb.z],
            "transform_matrix": _clean_matrix_rows(nerf_matrix_to_rows(mg)),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "fl_x": fx, "fl_y": fy, "w": w, "h": h, "focal_length": fx,
        })

    payload = {
        "coordinate_system": "Cinema4D_Yup_RightHanded",
        "camera_looks_along": "-Z",
        "camera_model": str(intrinsics["model"]).upper(),
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": w, "height": h,
        "focal_length": fx, "fl_x": fx, "fl_y": fy, "w": w, "h": h,
        "frame_count": len(world_points),
        "frames": frames,
    }

    out_path = _pose_json_path()
    out_dir  = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Export: synthetic COLMAP data files
# ---------------------------------------------------------------------------

def _write_cameras_txt(path, intrinsics):
    model = str(intrinsics["model"]).strip().upper()
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        if model == "SIMPLE_PINHOLE":
            f.write("1 SIMPLE_PINHOLE {} {} {} {} {}\n".format(
                RESOLUTION_X, RESOLUTION_Y,
                intrinsics["fx"], intrinsics["cx"], intrinsics["cy"]))
        else:
            f.write("1 PINHOLE {} {} {} {} {} {}\n".format(
                RESOLUTION_X, RESOLUTION_Y,
                intrinsics["fx"], intrinsics["fy"],
                intrinsics["cx"], intrinsics["cy"]))


def export_colmap_data(world_points, target_pos, doc, target_obj,
                       render_cam=None, camera_matrices=None):
    if not world_points:
        return None

    output_dir = _normalize_path(_colmap_output_dir())
    if not output_dir:
        raise ValueError(
            "OUTPUT_PATH is empty.  Set it at the top of the script before running.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    cameras_txt  = os.path.join(output_dir, "cameras.txt")
    images_txt   = os.path.join(output_dir, "images.txt")
    points3d_txt = os.path.join(output_dir, "points3D.txt")

    intrinsics = _get_intrinsics(render_cam)
    _write_cameras_txt(cameras_txt, intrinsics)

    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    width  = float(RESOLUTION_X)
    height = float(RESOLUTION_Y)

    if doc is None or target_obj is None:
        raise ValueError("A target object is required for COLMAP export.")

    # ------------------------------------------------------------------
    # Sample sparse points from the object surface
    # ------------------------------------------------------------------
    sparse_pts = generate_sparse_points_from_surface(doc, target_obj,
                                                     SPARSE_POINT_COUNT)
    if not sparse_pts:
        raise ValueError(
            "Could not sample sparse points from the object surface.  "
            "Ensure the object has polygonal geometry.")

    # ------------------------------------------------------------------
    # Build per-image extrinsics
    # ------------------------------------------------------------------
    def _build_image_entries(use_cam_matrices):
        entries = []
        flip_y = _get_flip_y_matrix()
        for i, world_pos in enumerate(world_points):
            if use_cam_matrices and camera_matrices and i < len(camera_matrices):
                mg = camera_matrices[i]
            else:
                mg = look_at_matrix(world_pos, target_pos)
            q, t, r_w2c = c2w_to_colmap_extrinsics(mg)
            # Precompute COLMAP-frame c2w and its inverse so the hot
            # sparse-point × camera projection loop avoids repeating the
            # two-Y-flip transform for every point.
            colmap_mg = _apply_world_flip_y(mg * flip_y)
            colmap_inv_mg = ~colmap_mg
            entries.append({
                "image_id": i + 1,
                "name": os.path.basename(_frame_image_path(i)),
                "q": q, "t": t, "r_w2c": r_w2c, "mg": mg,
                "colmap_mg": colmap_mg, "colmap_inv_mg": colmap_inv_mg,
                "obs": [],
            })
        return entries

    image_entries = _build_image_entries(use_cam_matrices=True)
    image_entries = sorted(image_entries, key=lambda e: e["name"])
    for idx, entry in enumerate(image_entries, start=1):
        entry["image_id"] = idx

    max_obs = max(2, _MAX_TRACK_OBS_PER_POINT)

    def _build_tracks(require_front_facing):
        tracks = {}
        for entry in image_entries:
            entry["obs"] = []
        for pid, (p3d, nrm) in enumerate(sparse_pts, start=1):
            tracks[pid] = []
            candidates = []
            for entry in image_entries:
                projected = _project_in_colmap_frame(
                    entry["colmap_mg"], entry["colmap_inv_mg"],
                    p3d, nrm, fx, fy, cx, cy,
                    require_front_facing=require_front_facing,
                )
                if projected is None:
                    continue
                u, v = projected
                if 0.0 <= u < width and 0.0 <= v < height:
                    candidates.append((entry, u, v))
            for entry, u, v in _cap_observations(candidates, max_obs):
                p2d_idx = len(entry["obs"])
                entry["obs"].append((u, v, pid))
                tracks[pid].append((entry["image_id"], p2d_idx))
        return tracks

    tracks = _build_tracks(require_front_facing=True)

    valid_points = [
        (pid, p3d, tracks[pid])
        for pid, (p3d, _nrm) in enumerate(sparse_pts, start=1)
        if len(tracks.get(pid, [])) >= 1
    ]

    # Retry without front-facing culling (some objects have flipped normals)
    fallback_no_facing = False
    if not valid_points:
        tracks = _build_tracks(require_front_facing=False)
        valid_points = [
            (pid, p3d, tracks[pid])
            for pid, (p3d, _nrm) in enumerate(sparse_pts, start=1)
            if len(tracks.get(pid, [])) >= 1
        ]
        fallback_no_facing = True

    # Retry with analytic look-at poses (evaluated matrices may be unusable)
    fallback_analytic = False
    if not valid_points and camera_matrices:
        image_entries = _build_image_entries(use_cam_matrices=False)
        image_entries = sorted(image_entries, key=lambda e: e["name"])
        for idx, entry in enumerate(image_entries, start=1):
            entry["image_id"] = idx
        tracks = _build_tracks(require_front_facing=False)
        valid_points = [
            (pid, p3d, tracks[pid])
            for pid, (p3d, _nrm) in enumerate(sparse_pts, start=1)
            if len(tracks.get(pid, [])) >= 1
        ]
        fallback_analytic = True

    # Final fallback: use interior volume points
    fallback_volume = False
    if not valid_points:
        vol_pts = generate_sparse_points_in_core_volume(
            target_obj, target_pos, SPARSE_POINT_COUNT, 0.35)
        if vol_pts:
            sparse_pts = vol_pts
            tracks = _build_tracks(require_front_facing=False)
            valid_points = [
                (pid, p3d, tracks[pid])
                for pid, (p3d, _nrm) in enumerate(sparse_pts, start=1)
                if len(tracks.get(pid, [])) >= 1
            ]
            fallback_volume = True

    # ------------------------------------------------------------------
    # Write images.txt
    # ------------------------------------------------------------------
    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}\n".format(len(world_points)))
        for entry in image_entries:
            qw, qx, qy, qz = entry["q"]
            tx, ty, tz = entry["t"]
            f.write("{} {} {} {} {} {} {} {} 1 {}\n".format(
                entry["image_id"],
                _clean_small(qw), _clean_small(qx),
                _clean_small(qy), _clean_small(qz),
                _clean_small(tx), _clean_small(ty), _clean_small(tz),
                entry["name"],
            ))
            if entry["obs"]:
                f.write(" ".join(
                    "{} {} {}".format(o[0], o[1], o[2]) for o in entry["obs"]) + "\n")
            else:
                f.write("\n")

    if not valid_points:
        obs_counts = [len(v) for v in tracks.values()]
        debug = [
            "COLMAP export debug",
            "sampled_points={}".format(len(sparse_pts)),
            "total_images={}".format(len(world_points)),
            "points_ge_1_obs={}".format(sum(1 for n in obs_counts if n >= 1)),
            "resolution={}x{}".format(int(width), int(height)),
            "fx={} fy={} cx={} cy={}".format(fx, fy, cx, cy),
            "fallback_no_facing={}".format(fallback_no_facing),
            "fallback_analytic={}".format(fallback_analytic),
            "fallback_volume={}".format(fallback_volume),
        ]
        report = os.path.join(output_dir, "colmap_debug.txt")
        with open(report, "w") as f:
            f.write("\n".join(debug) + "\n")
        raise ValueError(
            "No sparse point had >= 1 observation after visibility checks.  "
            "Try increasing CAMERA_COUNT, SPHERE_RADIUS, or SPARSE_POINT_COUNT.  "
            "Debug report: {}".format(report))

    # ------------------------------------------------------------------
    # Write points3D.txt
    # ------------------------------------------------------------------
    with open(points3d_txt, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: {}\n".format(len(valid_points)))
        for pid, p3d, track in valid_points:
            # Convert world → COLMAP frame (flip Y only, matching importer inverse)
            colmap_x = p3d.x
            colmap_y = -p3d.y
            colmap_z = p3d.z
            track_flat = " ".join(
                "{} {}".format(img_id, p2d) for img_id, p2d in track)
            f.write("{} {} {} {} 255 255 255 1.0 {}\n".format(
                pid, colmap_x, colmap_y, colmap_z, track_flat))

    return {
        "dir": output_dir,
        "cameras_txt": cameras_txt,
        "images_txt": images_txt,
        "points3d_txt": points3d_txt,
        "images_dir": images_dir,
        "points_count": len(valid_points),
        "intrinsics_source": intrinsics.get("source", "computed"),
        "model": intrinsics.get("model", "PINHOLE"),
    }


# ---------------------------------------------------------------------------
# Scene building helpers
# ---------------------------------------------------------------------------

def _make_track(op, descid):
    track = op.FindCTrack(descid)
    if track is None:
        track = c4d.CTrack(op, descid)
        op.InsertTrackSorted(track)
    return track


def _add_step_key(op, descid, time, value):
    track = _make_track(op, descid)
    curve = track.GetCurve()
    kd = curve.AddKey(time)
    if not kd:
        return
    key = kd["key"]
    key.SetValue(curve, float(value))
    key.SetInterpolation(curve, c4d.CINTERPOLATION_STEP)


def _create_target_tag(cam, target):
    tag = c4d.BaseTag(c4d.Ttargetexpression)
    if tag is None:
        return
    tag[c4d.TARGETEXPRESSIONTAG_LINK] = target
    cam.InsertTag(tag)


def _set_focus_distance(cam, target_pos):
    if cam is None or target_pos is None:
        return
    try:
        dist = float((target_pos - cam.GetAbsPos()).GetLength())
    except Exception:
        return
    for name in ["CAMERAOBJECT_TARGETDISTANCE", "CAMERAOBJECT_FOCUSDISTANCE",
                 "CAMERA_FOCUSDISTANCE", "CAMERAOBJECT_TARGETDIST"]:
        pid = getattr(c4d, name, None)
        if pid is None:
            continue
        try:
            cam[pid] = dist
            break
        except Exception:
            continue


def _camera_matrices_for_export(doc, render_cam, frame_count):
    if doc is None or render_cam is None or frame_count <= 0:
        return None
    current_time = doc.GetTime()
    out = []
    try:
        for frame in range(frame_count):
            doc.SetTime(c4d.BaseTime(frame, FPS))
            try:
                doc.ExecutePasses(None, True, True, True,
                                  getattr(c4d, "BUILDFLAGS_NONE", 0))
            except Exception:
                pass
            out.append(_copy_matrix(render_cam.GetMg()))
    finally:
        doc.SetTime(current_time)
        try:
            doc.ExecutePasses(None, True, True, True,
                              getattr(c4d, "BUILDFLAGS_NONE", 0))
        except Exception:
            pass
    return out if len(out) == frame_count else None


def _configure_render_settings(doc, render_cam, frame_count):
    rd = doc.GetActiveRenderData()
    if rd is None:
        return
    images_dir = _images_output_dir()
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    rd[c4d.RDATA_XRES]          = RESOLUTION_X
    rd[c4d.RDATA_YRES]          = RESOLUTION_Y
    rd[c4d.RDATA_FRAMERATE]     = FPS
    rd[c4d.RDATA_SAVEIMAGE]     = True
    rd[c4d.RDATA_PATH]          = _render_output_pattern()
    rd[c4d.RDATA_FORMAT]        = getattr(c4d, "FILTER_PNG", 1023671)
    rd[c4d.RDATA_FRAMESEQUENCE] = c4d.RDATA_FRAMESEQUENCE_ALLFRAMES
    rd[c4d.RDATA_FRAMEFROM]     = c4d.BaseTime(0, FPS)
    rd[c4d.RDATA_FRAMETO]       = c4d.BaseTime(max(0, frame_count - 1), FPS)

    if ENABLE_STRAIGHT_ALPHA:
        # Enable alpha channel
        for alpha_cid in ["RDATA_ALPHACHANNEL", "RDATA_ALPHA_CHANNEL"]:
            pid = getattr(c4d, alpha_cid, None)
            if pid is not None:
                try:
                    rd[pid] = True
                    break
                except Exception:
                    continue
        # Set straight (non-premultiplied) alpha
        for straight_cid in ["RDATA_STRAIGHTALPHA", "RDATA_STRAIGHT_ALPHA"]:
            pid = getattr(c4d, straight_cid, None)
            if pid is not None:
                try:
                    rd[pid] = True
                    break
                except Exception:
                    continue

    if render_cam is not None:
        if hasattr(c4d, "RDATA_CAMERA"):
            try:
                rd[c4d.RDATA_CAMERA] = render_cam
            except Exception:
                pass
        else:
            bd = doc.GetActiveBaseDraw()
            if bd is not None:
                bd.SetSceneCamera(render_cam)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    doc = c4d.documents.GetActiveDocument()
    if doc is None:
        c4d.gui.MessageDialog("C4D2GS Lite: No active document found.")
        return

    target_obj = doc.GetActiveObject()
    if target_obj is None:
        c4d.gui.MessageDialog(
            "C4D2GS Lite: No object selected.\n"
            "Please select the object you want to capture and run the script again.")
        return

    if not OUTPUT_PATH.strip():
        c4d.gui.MessageDialog(
            "C4D2GS Lite: OUTPUT_PATH is empty.\n"
            "Edit the OUTPUT_PATH parameter at the top of the script and run again.")
        return

    if CAMERA_COUNT < 1:
        c4d.gui.MessageDialog(
            "C4D2GS Lite: CAMERA_COUNT must be >= 1.\n"
            "Edit the CAMERA_COUNT parameter at the top of the script and run again.")
        return

    out_folder = _output_folder()
    if not out_folder:
        c4d.gui.MessageDialog("C4D2GS Lite: Could not resolve OUTPUT_PATH.")
        return

    target_pos = center_of_object(target_obj)

    world_pts       = []
    render_cam      = None
    camera_matrices = None

    doc.StartUndo()
    try:
        # Remove existing rig if requested
        if REPLACE_EXISTING_RIG:
            existing = find_object_by_name(doc.GetFirstObject(), _RIG_NAME)
            if existing is not None:
                doc.AddUndo(c4d.UNDOTYPE_DELETEOBJ, existing)
                existing.Remove()

        # Build rig null
        rig = c4d.BaseObject(c4d.Onull)
        rig.SetName(_RIG_NAME)
        rig.SetAbsPos(target_pos)
        doc.InsertObject(rig)
        doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, rig)

        # Target null (cameras look at this)
        target_null = c4d.BaseObject(c4d.Onull)
        target_null.SetName(_TARGET_NULL_NAME)
        target_null.InsertUnder(rig)
        target_null.SetAbsPos(target_pos)
        doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, target_null)

        # Generate viewpoint positions on the sphere (Fibonacci distribution)
        unit_pts  = fibonacci_sphere_points(CAMERA_COUNT)
        world_pts = [target_pos + p * SPHERE_RADIUS for p in unit_pts]

        # Static reference cameras (one per viewpoint)
        for i, wpos in enumerate(world_pts):
            cam = c4d.BaseObject(c4d.Ocamera)
            cam.SetName("GS_Cam_{:04d}".format(i))
            cam.InsertUnder(rig)
            cam.SetAbsPos(wpos)
            _create_target_tag(cam, target_null)
            _set_focus_distance(cam, target_pos)
            doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, cam)

        # Animated render camera
        render_cam = c4d.BaseObject(c4d.Ocamera)
        render_cam.SetName(_RENDER_CAM_NAME)
        render_cam.InsertUnder(rig)
        render_cam.SetAbsPos(world_pts[0])
        _create_target_tag(render_cam, target_null)
        _set_focus_distance(render_cam, target_pos)
        doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, render_cam)

        desc_x = c4d.DescID(
            c4d.DescLevel(c4d.ID_BASEOBJECT_REL_POSITION, c4d.DTYPE_VECTOR, 0),
            c4d.DescLevel(c4d.VECTOR_X, c4d.DTYPE_REAL, 0),
        )
        desc_y = c4d.DescID(
            c4d.DescLevel(c4d.ID_BASEOBJECT_REL_POSITION, c4d.DTYPE_VECTOR, 0),
            c4d.DescLevel(c4d.VECTOR_Y, c4d.DTYPE_REAL, 0),
        )
        desc_z = c4d.DescID(
            c4d.DescLevel(c4d.ID_BASEOBJECT_REL_POSITION, c4d.DTYPE_VECTOR, 0),
            c4d.DescLevel(c4d.VECTOR_Z, c4d.DTYPE_REAL, 0),
        )
        for frame, wpos in enumerate(world_pts):
            local_pos = wpos - target_pos
            t = c4d.BaseTime(frame, FPS)
            _add_step_key(render_cam, desc_x, t, local_pos.x)
            _add_step_key(render_cam, desc_y, t, local_pos.y)
            _add_step_key(render_cam, desc_z, t, local_pos.z)

        # Configure render settings
        _configure_render_settings(doc, render_cam, len(world_pts))

        # Collect evaluated camera matrices for accurate export
        camera_matrices = _camera_matrices_for_export(doc, render_cam, len(world_pts))

        doc.SetTime(c4d.BaseTime(0, FPS))
        c4d.EventAdd()

    finally:
        doc.EndUndo()

    # ------------------------------------------------------------------
    # Exports (outside undo block)
    # ------------------------------------------------------------------
    errors   = []
    messages = ["C4D2GS Lite — done.", ""]
    messages.append("Target : {}".format(target_obj.GetName()))
    messages.append("Cameras: {}  (Fibonacci sphere)".format(len(world_pts)))
    messages.append("Output : {}".format(out_folder))

    if EXPORT_CAMERA_POSES_JSON:
        try:
            pose_path = export_camera_poses_json(
                world_pts, target_pos, render_cam,
                camera_matrices=camera_matrices)
            messages.append("JSON   : {}".format(pose_path))
        except Exception as e:
            err = "JSON export failed: {}".format(e)
            errors.append(err)
            messages.append(err)

    if EXPORT_COLMAP_DATA:
        try:
            result = export_colmap_data(
                world_pts, target_pos, doc, target_obj,
                render_cam=render_cam,
                camera_matrices=camera_matrices)
            messages.append("COLMAP : {} ({} points, model={})".format(
                result["dir"], result["points_count"], result["model"]))
        except Exception as e:
            err = "COLMAP export failed: {}".format(e)
            errors.append(err)
            messages.append(err)

    if errors:
        messages.append("")
        messages.append("⚠ Some exports encountered errors (see above).")

    c4d.gui.MessageDialog("\n".join(messages))


if __name__ == "__main__":
    main()

"""if you read this you are a smart little rabbit and you should have a wonderful day! JT"""