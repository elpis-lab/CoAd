"""Analytic inner cells for yaw/translation tasks and a vertical hinged door.

TSR convention: Te = H(q) Delta Ts. At a cell center c, Te = H(c) Ts,
so the required displacement is Delta = inv(H(q)) H(c), not H(q)-H(c).
Lengths are metres; angles are radians. No collision claim follows from these
bounds alone: the fixed robot goal must also clear the cell's enclosure.
"""

from dataclasses import asdict, dataclass
import math
import itertools
from tqdm import tqdm
import numpy as np


@dataclass(frozen=True)
class TSRBounds:
    x: float
    y: float
    z: float
    yaw: float
    # Explicit numerical allowances for the IK solver, not cell dimensions.
    position_slack: float = 1e-4
    rotation_slack: float = 1e-3

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("TSR bounds must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)


def yaw_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def chord(radius, half_angle):
    """Maximum point displacement over an angular deviation <= half_angle."""
    if radius < 0 or half_angle < 0:
        raise ValueError("Negative radius or angular halfwidth")
    return 2 * radius * np.sin(min(half_angle, np.pi) / 2)


def microwave_radii(details):
    c = np.asarray(details["hinge_pos_body"])[:2]
    d = (
        np.asarray(details["door_origin_closed_body"])
        + np.asarray(details["handle_pos_door"])
    )[:2]
    rho = float(np.linalg.norm(d))
    return float(np.linalg.norm(c)) + rho, rho


def angular_translation(half_yaw, half_door=0.0, radii=None):
    if radii is None:
        if half_door:
            raise ValueError("Door bounds need hinge geometry")
        return 0.0
    body_radius, door_radius = radii
    return chord(body_radius, half_yaw) + chord(door_radius, half_door)


def calibrate_bounds(halfwidths, radii=None):
    """Choose task-defined TSR tolerances to retain an existing cell size.

    This is a compatibility choice, NOT a derivation of physical grasp quality
    from finger clearance. Collision feasibility is checked separately.
    """
    hx, hy, hz, u, v = halfwidths
    planar = np.hypot(hx, hy) + angular_translation(u, v, radii)
    return TSRBounds(
        float(planar + 1e-4),
        float(planar + 1e-4),
        float(hz + 1e-4),
        float(u + v + 1e-3),
    )


def inner_halfwidths(
    bounds, half_yaw=None, half_door=0.0, aspect=(1.0, 1.0), radii=None
):
    """Certified rectangle; equal aspect gives min(bx,by)/sqrt(2).

    With a door, reserve the chord displacement and combined angle budget.
    The rectangular generalization hx^2+hy^2 <= b_min^2 preserves the old
    anisotropic grids without giving up the yaw-independent guarantee.
    """
    u = bounds.yaw - bounds.rotation_slack - half_door if half_yaw is None else half_yaw
    if (
        min(u, half_door) < 0
        or u + half_door > bounds.yaw - bounds.rotation_slack + 1e-12
    ):
        raise ValueError("Cell angular widths exceed the TSR yaw budget")
    a = np.asarray(aspect, dtype=float)
    if a.shape != (2,) or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("Planar aspect must contain two positive finite values")
    budget = min(bounds.x, bounds.y) - bounds.position_slack
    budget -= angular_translation(u, half_door, radii)
    hz = bounds.z - bounds.position_slack
    if budget <= 0 or hz < -1e-12:
        raise ValueError("No positive inner cell fits these TSR bounds")
    hx, hy = budget * a / np.linalg.norm(a)
    return np.array([hx, hy, max(0.0, hz), u, half_door])


def reference_pose(center, details=None):
    """Object frame, or actual handle frame for a clockwise vertical hinge."""
    center = np.asarray(center, dtype=float)
    R = yaw_rotation(center[3])
    T = np.eye(4)
    T[:3, 3] = center[:3]
    T[:3, :3] = R
    if details is not None:
        Rd = yaw_rotation(-center[4])
        d = np.asarray(details["door_origin_closed_body"]) + np.asarray(
            details["handle_pos_door"]
        )
        T[:3, 3] += R @ (np.asarray(details["hinge_pos_body"]) + Rd @ d)
        T[:3, :3] = R @ Rd
    return T


def goal_fits_cell(bounds, key, actual_ee, offset, details=None):
    """Sufficient analytic TSR check including actual IK error for a whole cell."""
    box = np.asarray(key, dtype=float)
    half = (box[:, 1] - box[:, 0]) / 2
    if np.any(half < 0):
        raise ValueError("Reversed cell bounds")
    center = box.mean(axis=1)
    Hc = reference_pose(center, details)
    # E is the actual reference frame error after removing Ts.
    E = np.linalg.solve(Hc, actual_ee @ np.linalg.inv(offset))
    R = E[:3, :3]
    tilt = math.acos(float(np.clip(R[2, 2], -1, 1)))
    yaw_error = abs(math.atan2(R[1, 0], R[0, 0]))
    v = half[4] if len(half) == 5 else 0.0
    radii = microwave_radii(details) if details is not None else None
    planar = np.hypot(*half[:2]) + angular_translation(half[3], v, radii)
    return bool(
        planar + np.linalg.norm(E[:2, 3]) <= min(bounds.x, bounds.y) + 1e-12
        and half[2] + abs(E[2, 3]) <= bounds.z + 1e-12
        and half[3] + v + yaw_error <= bounds.yaw + 1e-12
        and tilt <= bounds.rotation_slack + 1e-12
    )


def tile_1d(intervals, width, eps=1e-12):
    """Contiguous regular bins; never round edges outward to five decimals."""
    bins = []
    for lo, hi in intervals:
        lo, hi = float(lo), float(hi)
        if not np.isfinite([lo, hi]).all() or hi < lo:
            raise ValueError("Invalid task interval")
        if hi == lo:
            bins.append([lo, hi])
            continue
        if not np.isfinite(width) or width <= 0:
            raise ValueError("A varying task dimension needs a positive cell width")
        count = int(math.ceil((hi - lo) / width))
        for i in range(count):
            a, b = lo + i * width, min(lo + (i + 1) * width, hi)
            if a < b:
                bins.append([a, b])
    return bins


def yaw_transform(theta):
    T = np.eye(4)
    T[:3, :3] = yaw_rotation(theta)
    return T


def make_tew_yaw_variants(tew_base, angles=(0.0, np.pi / 2, np.pi, 3 * np.pi / 2)):
    return [yaw_transform(th) @ tew_base for th in angles]


def valid_grasp_yaw_offsets(object_details, gripper_width, margin=0.0):
    sx, sy = (object_details["size"][0], object_details["size"][1])
    x_fits = sx + margin <= gripper_width
    y_fits = sy + margin <= gripper_width
    if not x_fits and (not y_fits):
        raise ValueError("Object does not fit gripper in either x or y.")
    offsets = []
    if x_fits:
        offsets += [np.pi / 2, 3 * np.pi / 2]
    if y_fits:
        offsets += [0.0, np.pi]
    return (tuple(offsets), x_fits, y_fits)


def translational_half_extents(del_geom_x, del_geom_y):
    """Validate historical geometric preferences, not certified TSR bounds."""
    if del_geom_x < 0 or del_geom_y < 0:
        raise ValueError(
            f"Negative grasp clearance: del_geom_x={del_geom_x}, del_geom_y={del_geom_y}. Check object size vs gripper width."
        )
    return (float(del_geom_x), float(del_geom_y))


def rmin_rmax_from_xy_interval(x_interval, y_interval, origin=(0.0, 0.0)):
    ox, oy = origin
    xmin, xmax = x_interval
    ymin, ymax = y_interval
    corners = np.array(
        [[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]], dtype=float
    )
    dists = np.linalg.norm(corners - np.array([ox, oy]), axis=1)
    rmax = float(np.max(dists))
    closest_x = np.clip(ox, xmin, xmax)
    closest_y = np.clip(oy, ymin, ymax)
    rmin = float(np.linalg.norm([closest_x - ox, closest_y - oy]))
    return (rmin, rmax)


def tile_in_reachable_annulus(tile, robot_pos, inner_rad, outer_rad):
    rmin, rmax = rmin_rmax_from_xy_interval(
        tile["x"], tile["y"], origin=(robot_pos[0], robot_pos[1])
    )
    return rmin <= outer_rad and rmax >= inner_rad


def tile_inside_intervals(tile, intervals):
    """Check whether an xy tile lies inside any valid xy region."""
    if intervals is None:
        return True
    intervals_array = np.asarray(intervals, dtype=float)
    if intervals_array.shape == (2, 2):
        regions = intervals_array[None, :, :]
    elif intervals_array.ndim == 3 and intervals_array.shape[1:] == (2, 2):
        regions = intervals_array
    else:
        raise ValueError(
            f"Expected intervals with shape (2, 2) or (N, 2, 2), got {intervals_array.shape}"
        )
    tile_xmin = min(tile["x"])
    tile_xmax = max(tile["x"])
    tile_ymin = min(tile["y"])
    tile_ymax = max(tile["y"])
    for region in regions:
        xmin, xmax = region[0]
        ymin, ymax = region[1]
        completely_inside = (
            tile_xmin >= xmin
            and tile_xmax <= xmax
            and (tile_ymin >= ymin)
            and (tile_ymax <= ymax)
        )
        if completely_inside:
            return True
    return False


def tile_center(tile):
    return {dim: 0.5 * (bounds[0] + bounds[1]) for (dim, bounds) in tile.items()}


def tile_center_inside_intervals(tile, intervals):
    """
    Check only whether the tile center lies inside the allowed XY intervals.

    intervals format:
        [[xmin, xmax], [ymin, ymax]]

    This does NOT check whether the full object footprint is inside.
    """
    if not intervals:
        return True
    xmin, xmax = intervals[0]
    ymin, ymax = intervals[1]
    center = tile_center(tile)
    cx = center["x"]
    cy = center["y"]
    return xmin <= cx <= xmax and ymin <= cy <= ymax


def create_tcr_set(env, batch_idx=None):
    env_details = env.env_details
    object_details = env.object_details
    grasp_details = env.grasp_details
    env_name = env.env_details["env_name"]
    if batch_idx is not None:
        tcr_intervals = grasp_details["tcr_intervals"][batch_idx]
    else:
        tcr_intervals = grasp_details["tcr_intervals"]
    variation = object_details["variation"]
    bin_widths = {}
    for bin_name, bin_intervals in tcr_intervals.items():
        if len(bin_intervals) == 1:
            bin_widths[bin_name] = 0
            continue
        bin_widths[bin_name] = bin_intervals[2] - bin_intervals[0]
    z_corrections = env.env_details.get("z_correction", [0])
    base_zmin = variation["z"][0][0]
    base_zmax = variation["z"][0][1]
    if batch_idx is not None:
        # Each stable support has its own object height above the surface.
        axis = {"xy": 2, "yz": 0, "zx": 1}[batch_idx]
        base_zmin = base_zmax = object_details["size"][axis] / 2

    variations_to_apply = variation.copy()
    variations_to_apply["z"] = [
        [base_zmin + dz, base_zmax + dz] for dz in z_corrections
    ]
    regions = {
        "x": variations_to_apply["x"],
        "y": variations_to_apply["y"],
        "z": variations_to_apply["z"],
        "yaw": variations_to_apply["yaw"],
    }
    if env_name == "microwave":
        regions["door"] = variations_to_apply["door"]
    print(f"regions: {regions}")
    print("TCR intervals:", tcr_intervals)
    print("Bin widths:", bin_widths)
    print("Regions:", regions)
    per_dim_bins, n_tiles = find_bins_per_dim(regions, bin_widths)
    valid_tiles = []
    robot_pos = env_details["robot_pos"]
    inner_rad = env_details["inner_rad"]
    outer_rad = env_details["outer_rad"]
    robot = env_details["robot"]
    env_name = env.env_details["env_name"]
    if env_name != "free":
        intervals = env.env_details["intervals"]
    else:
        intervals = None
    x_bins = per_dim_bins["x"]
    y_bins = per_dim_bins["y"]
    z_bins = per_dim_bins["z"]
    yaw_bins = per_dim_bins["yaw"]
    door_bins = per_dim_bins.get("door", [None])
    expand_count = len(z_bins) * len(yaw_bins) * len(door_bins)
    with tqdm(total=n_tiles, desc="Validating Task Set") as pbar:
        for x_bin, y_bin in itertools.product(x_bins, y_bins):
            xy_tile = {"x": x_bin, "y": y_bin}
            valid_xy = True
            if not tile_in_reachable_annulus(xy_tile, robot_pos, inner_rad, outer_rad):
                valid_xy = False
            if env_name in {"largeobj"}:
                inside = tile_center_inside_intervals(xy_tile, intervals)
            else:
                inside = tile_inside_intervals(xy_tile, intervals)
            if not inside:
                valid_xy = False
            if not valid_xy:
                pbar.update(expand_count)
                continue
            for yaw_bin, door_bin in itertools.product(yaw_bins, door_bins):
                if env_name == "microwave":
                    if robot == "panda":
                        outer_scale_val = 0.75
                    elif robot == "fetch":
                        outer_scale_val = 0.65
                    else:
                        raise NotImplementedError(
                            f"Unsupported robot for task validation: {robot} "
                        )
                    if not microwave_handle_filter(
                        x_bin,
                        y_bin,
                        yaw_bin,
                        door_bin,
                        env,
                        dot_min=0.0,
                        outer_rad_scale=outer_scale_val,
                    ):
                        pbar.update(len(z_bins))
                        continue
                for z_bin in z_bins:
                    tile = {"x": x_bin, "y": y_bin, "z": z_bin, "yaw": yaw_bin}
                    if door_bin is not None:
                        tile["door"] = door_bin
                    if door_bin is None:
                        TCR = (tuple(x_bin), tuple(y_bin), tuple(z_bin), tuple(yaw_bin))
                    else:
                        TCR = (
                            tuple(x_bin),
                            tuple(y_bin),
                            tuple(z_bin),
                            tuple(yaw_bin),
                            tuple(door_bin),
                        )
                    valid_tiles.append(TCR)
                pbar.update(len(z_bins))
    print(f"valid tiles: {len(valid_tiles)} / {n_tiles}")
    TCR_set = tiles_to_itsr_set(env, valid_tiles)
    return TCR_set


def yaw_rot_2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def bin_mid(bin_):
    return 0.5 * (float(bin_[0]) + float(bin_[1]))


def microwave_handle_pose_xy(x_bin, y_bin, yaw_bin, door_bin, env):
    x = bin_mid(x_bin)
    y = bin_mid(y_bin)
    yaw = bin_mid(yaw_bin)
    door_phi = bin_mid(door_bin)
    hinge_xy = np.array(env.object_details["hinge_pos_body"][:2], dtype=float)
    door_origin_xy = np.array(
        env.object_details["door_origin_closed_body"][:2], dtype=float
    )
    handle_pos_door_xy = np.array(
        env.object_details["handle_pos_door"][:2], dtype=float
    )
    R_body = yaw_rot_2d(yaw)
    R_door = yaw_rot_2d(-door_phi)
    handle_body_xy = hinge_xy + R_door @ (door_origin_xy + handle_pos_door_xy)
    handle_world_xy = np.array([x, y], dtype=float) + R_body @ handle_body_xy
    handle_normal_door = np.array([-1.0, 0.0], dtype=float)
    handle_normal_world = R_body @ R_door @ handle_normal_door
    handle_normal_world /= np.linalg.norm(handle_normal_world) + 1e-09
    return (handle_world_xy, handle_normal_world)


def microwave_handle_filter(
    x_bin, y_bin, yaw_bin, door_bin, env, dot_min=0.0, outer_rad_scale=0.75
):
    robot = env.env_details["robot"]
    if robot in {"panda", "fetch"}:
        handle_pos_xy, handle_normal_xy = microwave_handle_pose_xy(
            x_bin, y_bin, yaw_bin, door_bin, env
        )
        robot_xy = np.array(env.env_details["robot_pos"][:2], dtype=float)
        to_robot = robot_xy - handle_pos_xy
        dist = np.linalg.norm(to_robot)
        if dist < 1e-09:
            return True
        to_robot = to_robot / dist
        dot = np.dot(handle_normal_xy, to_robot)
        if dot >= dot_min:
            return True
        inner_rad = env.env_details["inner_rad"]
        outer_rad = env.env_details["outer_rad"]
        close_outer_rad = outer_rad_scale * outer_rad
        return inner_rad <= dist <= close_outer_rad
    else:
        raise ValueError(f"Microwave handle check not supported for robot: {robot}")


def tiles_to_itsr_set(env, valid_tiles):
    return {key: None for key in valid_tiles}


def find_bins_per_dim(regions, bin_widths):
    per_dim_bins = {
        dim: tile_1d(intervals, bin_widths[dim]) for (dim, intervals) in regions.items()
    }
    n_tiles = 1
    for dim, bins in per_dim_bins.items():
        print(f"{dim}: {len(bins)} bins")
        n_tiles *= len(bins)
    print(f"total tiles: {n_tiles}")
    return (per_dim_bins, n_tiles)
