import numpy as np
from tqdm import tqdm
import itertools

# TODO
# This part generates the TCRs for the workspace
# but is currently written as "Intersection of TSRs"
# This can be written as TCR directly with
# 1, theoretical bounds in SE2 derived from TSRs' bound
# 2, customized bounds directly defined in TCRs



def find_TSR_HMat(xyz, yaw):
    TSR_HMat = np.eye(4)
    TSR_HMat[0, 0] = np.cos(yaw)
    TSR_HMat[0, 1] = -np.sin(yaw)
    TSR_HMat[1, 0] = np.sin(yaw)
    TSR_HMat[1, 1] = np.cos(yaw)

    TSR_HMat[0, 3] = xyz[0]
    TSR_HMat[1, 3] = xyz[1]
    TSR_HMat[2, 3] = xyz[2]

    return TSR_HMat


def xy_half_extents(half_side):
    """Return separate x/y half extents.

    Older code passed a scalar `half_side`, which treated the translational
    footprint as square.  For rectangular objects, pass `(x_half, y_half)` to
    preserve different x/y coverage.
    """
    if isinstance(half_side, (list, tuple, np.ndarray)):
        if len(half_side) != 2:
            raise ValueError("half_side must be a scalar or a length-2 x/y extent")
        return float(half_side[0]), float(half_side[1])
    return float(half_side), float(half_side)


def find_TSR_Bounds(Bw, TSR_HMat1, yaw_1, yaw_2, yaw_buffer, half_side, grasp):
    tw1_0 = TSR_HMat1[:3, 3]
    x_half, y_half = xy_half_extents(half_side)

    if grasp == "top":
        B = np.array(
            [
                [tw1_0[0] - x_half, tw1_0[0] + x_half],
                [tw1_0[1] - y_half, tw1_0[1] + y_half],
                [tw1_0[2] + Bw[2, 0], tw1_0[2] + Bw[2, 1]],
                [Bw[3, 0], Bw[3, 1]],
                [Bw[4, 0], Bw[4, 1]],
                [yaw_2 - yaw_buffer, yaw_1 + yaw_buffer],
            ]
        )
    else:  # front
        B = np.array(
            [
                [tw1_0[0] - x_half, tw1_0[0] + x_half],
                [tw1_0[1] - y_half, tw1_0[1] + y_half],
                [tw1_0[2] + Bw[2, 0], tw1_0[2] + Bw[2, 1]],
                [yaw_2 - yaw_buffer, yaw_1 + yaw_buffer],
                [Bw[4, 0], Bw[4, 1]],
                [Bw[5, 0], Bw[5, 1]],
            ]
        )

    return B


def find_B0_intersection(B1_0, B2_0):
    # x
    xmin = max(B1_0[0, 0], B2_0[0, 0])
    xmax = min(B1_0[0, 1], B2_0[0, 1])
    if xmin <= xmax:
        xlim = [xmin, xmax]
    else:
        xlim = [None, None]
        return None

    ymin = max(B1_0[1, 0], B2_0[1, 0])
    ymax = min(B1_0[1, 1], B2_0[1, 1])
    if ymin <= ymax:
        ylim = [ymin, ymax]
    else:
        ylim = [None, None]
        return None

    zmin = max(B1_0[2, 0], B2_0[2, 0])
    zmax = min(B1_0[2, 1], B2_0[2, 1])
    if zmin <= zmax:
        zlim = [zmin, zmax]
    else:
        zlim = [None, None]
        return None

    rmin = max(B1_0[3, 0], B2_0[3, 0])
    rmax = min(B1_0[3, 1], B2_0[3, 1])
    if rmin <= rmax:
        rlim = [rmin, rmax]
    else:
        rlim = [None, None]
        return None

    pmin = max(B1_0[4, 0], B2_0[4, 0])
    pmax = min(B1_0[4, 1], B2_0[4, 1])
    if pmin <= pmax:
        plim = [pmin, pmax]
    else:
        plim = [None, None]
        return None

    yawmin = max(B1_0[5, 0], B2_0[5, 0])
    yawmax = min(B1_0[5, 1], B2_0[5, 1])
    if yawmin <= yawmax:
        yawlim = [yawmin, yawmax]
    else:
        yawlim = [None, None]
        return None

    B0_intersect = np.array([xlim, ylim, zlim, rlim, plim, yawlim])

    return B0_intersect


def merge_intervals(intervals, *, touch_as_overlap=True):
    # accept [lo, hi] lists/tuples; [] is treated as empty/no-op
    normed = list(extract_intervals(intervals))  # flatten/normalize
    if not normed:
        return []

    normed.sort()  # sort by (lo, hi)
    merged = [normed[0][:]]  # copy
    for a, b in normed[1:]:
        c, d = merged[-1]
        cond = (a <= d) if touch_as_overlap else (a < d)
        if cond:
            merged[-1][1] = max(d, b)  # extend hi
        else:
            merged.append([a, b])
    return merged


def extract_intervals(x):
    # yields [lo, hi] from arbitrarily nested lists/tuples
    if isinstance(x, (list, tuple)):
        if len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
            yield [x[0], x[1]]
        else:
            for child in x:
                yield from extract_intervals(child)


def find_intersection(s1, s2):
    # x
    xmin = max(s1[0], s2[0])
    xmax = min(s1[1], s2[1])
    if xmin > xmax:
        return None
    xlim = [xmin, xmax]

    return xlim


def rmin_rmax_from_rectangle_corners_2d(tw1, tw2, nominal_pose=None):
    x0, y0 = nominal_pose[0], nominal_pose[1]

    # robust min/max even if tw1/tw2 aren't ordered
    x_min = tw1[0] if tw1[0] < tw2[0] else tw2[0]
    x_max = tw2[0] if tw1[0] < tw2[0] else tw1[0]
    y_min = tw1[1] if tw1[1] < tw2[1] else tw2[1]
    y_max = tw2[1] if tw1[1] < tw2[1] else tw1[1]

    # r_min: distance from (x0,y0) to nearest point on rectangle
    nx = x0
    if nx < x_min:
        nx = x_min
    elif nx > x_max:
        nx = x_max

    ny = y0
    if ny < y_min:
        ny = y_min
    elif ny > y_max:
        ny = y_max

    dx = nx - x0
    dy = ny - y0
    r_min = np.hypot(dx, dy)  # 0 if inside rectangle

    # r_max: max distance from (x0,y0) to the rectangle corners
    dx1 = x_min - x0
    dx2 = x_max - x0
    dy1 = y_min - y0
    dy2 = y_max - y0

    # four corners: (dx1,dy1), (dx1,dy2), (dx2,dy1), (dx2,dy2)
    r2_1 = dx1 * dx1 + dy1 * dy1
    r2_2 = dx1 * dx1 + dy2 * dy2
    r2_3 = dx2 * dx2 + dy1 * dy1
    r2_4 = dx2 * dx2 + dy2 * dy2

    r_max = np.sqrt(max(r2_1, r2_2, r2_3, r2_4))

    return r_min, r_max



# Backwards-compatible alias for older imports/calls.
rmin_rmax_from_square_corners = rmin_rmax_from_rectangle_corners_2d


def rmin_rmax_from_box_corners(tw1, tw2, nominal_pose=(0.0, 0.0, 0.0)):
    x0, y0, z0 = nominal_pose

    # Robust min/max
    x_min = tw1[0] if tw1[0] < tw2[0] else tw2[0]
    x_max = tw2[0] if tw1[0] < tw2[0] else tw1[0]

    y_min = tw1[1] if tw1[1] < tw2[1] else tw2[1]
    y_max = tw2[1] if tw1[1] < tw2[1] else tw1[1]

    z_min = tw1[2] if tw1[2] < tw2[2] else tw2[2]
    z_max = tw2[2] if tw1[2] < tw2[2] else tw1[2]

    # ---- r_min ----
    nx = x0
    if nx < x_min:
        nx = x_min
    elif nx > x_max:
        nx = x_max

    ny = y0
    if ny < y_min:
        ny = y_min
    elif ny > y_max:
        ny = y_max

    nz = z0
    if nz < z_min:
        nz = z_min
    elif nz > z_max:
        nz = z_max

    dx = nx - x0
    dy = ny - y0
    dz = nz - z0

    r_min = np.sqrt(dx * dx + dy * dy + dz * dz)

    # ---- r_max ----
    dx1 = x_min - x0
    dx2 = x_max - x0
    dy1 = y_min - y0
    dy2 = y_max - y0
    dz1 = z_min - z0
    dz2 = z_max - z0

    # 8 corners
    r2_vals = [
        dx1 * dx1 + dy1 * dy1 + dz1 * dz1,
        dx1 * dx1 + dy1 * dy1 + dz2 * dz2,
        dx1 * dx1 + dy2 * dy2 + dz1 * dz1,
        dx1 * dx1 + dy2 * dy2 + dz2 * dz2,
        dx2 * dx2 + dy1 * dy1 + dz1 * dz1,
        dx2 * dx2 + dy1 * dy1 + dz2 * dz2,
        dx2 * dx2 + dy2 * dy2 + dz1 * dz1,
        dx2 * dx2 + dy2 * dy2 + dz2 * dz2,
    ]

    r_max = np.sqrt(max(r2_vals))

    return r_min, r_max


def Tz(theta):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:3, :3] = np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    return T


def Tx(theta):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:3, :3] = np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ],
        dtype=float,
    )
    return T


def make_Tew_yaw_variants(
    Tew_base, angles=(0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
):
    return [Tz(th) @ Tew_base for th in angles]


def make_Tew_x_variants(
    Tew_base, angles=(0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
):
    return [Tx(th) @ Tew_base for th in angles]


def is_box_object(object_details):
    return object_details.get("type", "box") == "box"


def valid_yaw_angles_for_box_grasp(object_size, gripper_width, margin=0.0):
    """Return yaw variants that can physically fit a rectangular box.

    Convention used here:
      * 0 and pi close across the object's y dimension.
      * pi/2 and 3pi/2 close across the object's x dimension.

    This keeps the old 4-fold symmetry for square/small objects, but rejects
    impossible 90-degree variants for long cuboids.
    """
    sx, sy = float(object_size[0]), float(object_size[1])
    allowed = []

    if sy + margin <= gripper_width:
        allowed.extend([0.0, np.pi])
    if sx + margin <= gripper_width:
        allowed.extend([np.pi / 2, 3 * np.pi / 2])

    if not allowed:
        raise ValueError(
            f"Object footprint {object_size[:2]} does not fit gripper width "
            f"{gripper_width} with margin {margin}."
        )

    return tuple(allowed)


def yaw_angles_for_object_grasp(object_details, gripper_width, margin=0.0):
    if is_box_object(object_details):
        return valid_yaw_angles_for_box_grasp(
            object_details["size"], gripper_width, margin
        )
    return (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)


def effective_xy_size_for_grasp_clearance(object_details, gripper_width, margin=0.0):
    """Return the x/y size to use when computing TSR translational clearance.

    For long cuboids, one footprint dimension may exceed the gripper opening,
    but the object can still be graspable if the perpendicular/narrow dimension
    fits. Since this generator stores one Bw per grasp strategy, not one Bw per
    yaw variant, we use the graspable/narrow dimension for both x/y clearance
    whenever only one orientation class is physically valid.

    This avoids immediately producing negative clearance for long-but-graspable
    cuboids while keeping the original behavior for square/small boxes.
    """
    if not is_box_object(object_details):
        return object_details["size"][0], object_details["size"][1]

    sx, sy = float(object_details["size"][0]), float(object_details["size"][1])
    x_fits = sx + margin <= gripper_width
    y_fits = sy + margin <= gripper_width

    if x_fits and y_fits:
        return sx, sy
    if y_fits:
        return sy, sy
    if x_fits:
        return sx, sx

    raise ValueError(
        f"Object footprint {object_details['size'][:2]} does not fit gripper width "
        f"{gripper_width} with margin {margin}."
    )

def valid_grasp_yaw_offsets(object_details, gripper_width, margin=0.0):
    sx, sy = object_details["size"][0], object_details["size"][1]

    x_fits = sx + margin <= gripper_width
    y_fits = sy + margin <= gripper_width

    if not x_fits and not y_fits:
        raise ValueError("Object does not fit gripper in either x or y.")

    offsets = []

    # Grasp across y dimension: object x can hang out
    if x_fits:
        offsets += [np.pi / 2, 3 * np.pi / 2]

    # Grasp across x dimension: object y can hang out
    if y_fits:
        offsets += [0.0, np.pi]

    return tuple(offsets), x_fits, y_fits

def translational_half_extents(del_geom_x, del_geom_y):
    """Separate x/y translational coverage for one rectangular TSR bin.

    The old code used one conservative scalar based on the smaller clearance:
        0.5 * sqrt(2) * min(del_geom_x, del_geom_y)
    That is safe but throws away useful coverage for long/narrow cuboids.
    """
    # print(del_geom_x)
    # print(del_geom_y)
    if del_geom_x < 0 or del_geom_y < 0:
        raise ValueError(
            f"Negative grasp clearance: del_geom_x={del_geom_x}, "
            f"del_geom_y={del_geom_y}. Check object size vs gripper width."
        )
    return (float(del_geom_x), float(del_geom_y))

def get_del_geoms(clearance_size_x, clearance_size_y, gripper_width):
    pass

def panda_TSR_parameters(object_details, yaw_buffer, alpha, grasp_strategy="top", min_contact_overlap=0.01):
    object_position = object_details["position"]
    object_size = object_details["size"]
    object_dist = object_details["dist"]
    TSR_params = {}
    # Top TSR params

    if grasp_strategy == "top":

        # Panda specifications
        #ee_z_offset = 0.02
        ee_z_offset = 0
        # ee_z_offset = 0
        s_f = 0.04

        Tew = np.eye(4)
        Tew[1, 1] = -1
        Tew[2, 2] = -1
        Tew[2, 3] = ee_z_offset + object_size[2] / 2
        # Tew[2, 3] = ee_z_offset + object_size[2] / 4

        clearance_size_x, clearance_size_y = float(object_details["size"][0]), float(object_details["size"][1])

        #yaw_angles = yaw_angles_for_object_grasp(object_details, gripper_width=2 * s_f)
        yaw_angles, x_fits, y_fits = valid_grasp_yaw_offsets(object_details, 2 * s_f)
        Tews = make_Tew_yaw_variants(Tew, yaw_angles)

        print(x_fits, y_fits)
        print(object_details)


        if y_fits and not x_fits:
            long_axis_slide = max(0.0, clearance_size_x / 2 - min_contact_overlap)
            del_geom_x = long_axis_slide
            del_geom_y = s_f - clearance_size_y / 2

        elif x_fits and not y_fits:
            long_axis_slide = max(0.0, clearance_size_y / 2 - min_contact_overlap)
            del_geom_y = long_axis_slide
            del_geom_x = s_f - clearance_size_x / 2

        elif x_fits and y_fits:
            del_geom_x = s_f - clearance_size_x / 2
            del_geom_y = s_f - clearance_size_y / 2

        else:
            raise ValueError("Object does not fit the gripper in either x or y.")

        print(del_geom_x)
        print(del_geom_y)

        Bw = np.array(
            [
                [-del_geom_x, del_geom_x],
                [-del_geom_y, del_geom_y],
                [0, 0],
                [0, 0],
                [0, 0],
                [0 - yaw_buffer, 0 + yaw_buffer],
            ]
        )


        # Approximating the intersection of all rotated TSR's with a conservative TSR
        print(del_geom_x)
        print(del_geom_y)
        half_side = translational_half_extents(del_geom_x, del_geom_y)

        object_dist_check = np.sign(np.array(object_dist))
        del_Bw = np.array(
            [
                (Bw[0, 1] - Bw[0, 0]) / 2,
                (Bw[1, 1] - Bw[1, 0]) / 2,
                (Bw[2, 1] - Bw[2, 0]) / 2,
                (Bw[5, 1] - Bw[5, 0]) / 2,
            ]
        )
        del_Bw = object_dist_check * del_Bw
        Tw2_w1 = alpha * del_Bw

        yaw_tw2_w1 = (
            np.array([alpha * xy_half_extents(half_side)[0], alpha * xy_half_extents(half_side)[1], Tw2_w1[2]])
            * object_dist_check[0:3]
        )

        #TSR_params["top"] = (Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1)
        TSR_params = (Tews, Bw, half_side, Tw2_w1, yaw_tw2_w1)

    else:

        # Front TSR params
        obj_offset = np.sqrt(object_size[0] ** 2 + object_size[1] ** 2) / 1
        l_f = 0.054 / 2
        ee_offset = l_f * 1.25

        # Tew = np.eye(4)
        # Tew[1, 1] = -1
        # Tew[2, 2] = -1
        # Tew[2, 3] = ee_z_offset + object_position[2]/2
        # Tew[2, 3] = ee_z_offset + object_size[2]/4

        """
        Tew = np.array([
            [ np.cos(np.pi/2), 0,  np.sin(np.pi/2), 0],
            [ 0, 1,  0, 0],
            [-np.sin(np.pi/2), 0,  np.cos(np.pi/2), 0],
            [0, 0, 0, 1]
        ])
        """
        Ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
        Rz_m90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        #R_new = Ry90 @ Rz_m90
        R_new = Ry90
        Tew = np.eye(4)
        Tew[:3, :3] = R_new

        # Tew[0, 3] = -1*(ee_offset)
        ee_offset_eeframe = np.array([0.0, 0.0, -ee_offset])
        Tew[:3, 3] = R_new @ ee_offset_eeframe

        #Tews = make_Tew_x_variants(Tew)
        yaw_angles = yaw_angles_for_object_grasp(object_details, gripper_width=2 * l_f)
        Tews = make_Tew_yaw_variants(Tew, yaw_angles)
        #print(f"Tews length: {len(Tews)}")

        clearance_size_x, clearance_size_y = effective_xy_size_for_grasp_clearance(
            object_details, gripper_width=2 * l_f
        )
        del_geom_x = l_f - (clearance_size_x / 2)
        del_geom_y = l_f - (clearance_size_y / 2)

        Bw = np.array(
            [
                [-del_geom_x, del_geom_x],
                [-del_geom_y, del_geom_y],
                [0, 0],
                [0 - yaw_buffer, 0 + yaw_buffer],
                [0, 0],
                [0, 0],
            ]
        )


        # Approximating the intersection of all rotated TSR's with a conservative TSR
        half_side = translational_half_extents(del_geom_x, del_geom_y)

        object_dist_check = np.sign(np.array(object_dist))
        del_Bw = np.array(
            [
                (Bw[0, 1] - Bw[0, 0]) / 2,
                (Bw[1, 1] - Bw[1, 0]) / 2,
                (Bw[2, 1] - Bw[2, 0]) / 2,
                (Bw[3, 1] - Bw[3, 0]) / 2,
            ]
        )
        del_Bw = object_dist_check * del_Bw
        Tw2_w1 = alpha * del_Bw

        yaw_tw2_w1 = (
            np.array([alpha * xy_half_extents(half_side)[0], alpha * xy_half_extents(half_side)[1], Tw2_w1[2]])
            * object_dist_check[0:3]
        )

        #TSR_params["front"] = (Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1)
        TSR_params = (Tews, Bw, half_side, Tw2_w1, yaw_tw2_w1)

    # return Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1
    return TSR_params


def fetch_TSR_parameters(object_details, yaw_buffer, alpha):
    object_position = object_details["position"]
    object_size = object_details["size"]
    object_dist = object_details["dist"]
    TSR_params = {}
    # Top TSR params

    # Panda specifications
    ee_z_offset = 0.02
    # ee_z_offset = 0
    s_f = 0.05

    Tew = np.eye(4)
    Tew[1, 1] = -1
    Tew[2, 2] = -1
    Tew[2, 3] = ee_z_offset + object_size[2] / 2
    # Tew[2, 3] = ee_z_offset + object_size[2] / 4

    yaw_angles = yaw_angles_for_object_grasp(object_details, gripper_width=2 * s_f)
    Tews = make_Tew_yaw_variants(Tew, yaw_angles)

    del_geom = s_f
    clearance_size_x, clearance_size_y = effective_xy_size_for_grasp_clearance(
        object_details, gripper_width=2 * s_f
    )
    del_geom_x = s_f - (clearance_size_x / 2)
    del_geom_y = s_f - (clearance_size_y / 2)

    Bw = np.array(
        [
            [-del_geom_x, del_geom_x],
            [-del_geom_y, del_geom_y],
            [0, 0],
            [0, 0],
            [0, 0],
            [0 - yaw_buffer, 0 + yaw_buffer],
        ]
    )

    # Approximating the intersection of all rotated TSR's with a conservative TSR
    half_side = translational_half_extents(del_geom_x, del_geom_y)

    object_dist_check = np.sign(np.array(object_dist))
    del_Bw = np.array(
        [
            (Bw[0, 1] - Bw[0, 0]) / 2,
            (Bw[1, 1] - Bw[1, 0]) / 2,
            (Bw[2, 1] - Bw[2, 0]) / 2,
            (Bw[5, 1] - Bw[5, 0]) / 2,
        ]
    )
    del_Bw = object_dist_check * del_Bw
    Tw2_w1 = alpha * del_Bw

    yaw_tw2_w1 = (
        np.array([alpha * xy_half_extents(half_side)[0], alpha * xy_half_extents(half_side)[1], Tw2_w1[2]])
        * object_dist_check[0:3]
    )

    # TSR_params["top"] = (Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1)
    TSR_params["top"] = (Tews, Bw, half_side, Tw2_w1, yaw_tw2_w1)

    # Front TSR params
    obj_offset = np.sqrt(object_size[0] ** 2 + object_size[1] ** 2) / 1
    l_f = 0.05 / 2
    ee_offset = l_f * 1.25

    # Tew = np.eye(4)
    # Tew[1, 1] = -1
    # Tew[2, 2] = -1
    # Tew[2, 3] = ee_z_offset + object_position[2]/2
    # Tew[2, 3] = ee_z_offset + object_size[2]/4

    """
    Tew = np.array([
        [ np.cos(np.pi/2), 0,  np.sin(np.pi/2), 0],
        [ 0, 1,  0, 0],
        [-np.sin(np.pi/2), 0,  np.cos(np.pi/2), 0],
        [0, 0, 0, 1]
    ])
    """
    Ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    Rx90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    Rxm90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    Rz_m90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
    # R_new = Ry90 @ Rz_m90
    R_new = Ry90 @ Rxm90
    Tew = np.eye(4)
    Tew[:3, :3] = R_new

    # Tew[0, 3] = -1*(ee_offset)
    ee_offset_eeframe = np.array([0.0, 0.0, -ee_offset])
    Tew[:3, 3] = R_new @ ee_offset_eeframe

    #Tews = make_Tew_x_variants(Tew)
    yaw_angles = yaw_angles_for_object_grasp(object_details, gripper_width=2 * l_f)
    Tews = make_Tew_yaw_variants(Tew, yaw_angles)
    #print(f"Tews length: {len(Tews)}")

    clearance_size_x, clearance_size_y = effective_xy_size_for_grasp_clearance(
        object_details, gripper_width=2 * l_f
    )
    del_geom_x = l_f - (clearance_size_x / 2)
    del_geom_y = l_f - (clearance_size_y / 2)

    Bw = np.array(
        [
            [-del_geom_x, del_geom_x],
            [-del_geom_y, del_geom_y],
            [0, 0],
            [0 - yaw_buffer, 0 + yaw_buffer],
            [0, 0],
            [0, 0],
        ]
    )

    # Approximating the intersection of all rotated TSR's with a conservative TSR
    half_side = translational_half_extents(del_geom_x, del_geom_y)

    object_dist_check = np.sign(np.array(object_dist))
    del_Bw = np.array(
        [
            (Bw[0, 1] - Bw[0, 0]) / 2,
            (Bw[1, 1] - Bw[1, 0]) / 2,
            (Bw[2, 1] - Bw[2, 0]) / 2,
            (Bw[3, 1] - Bw[3, 0]) / 2,
        ]
    )
    del_Bw = object_dist_check * del_Bw
    Tw2_w1 = alpha * del_Bw

    yaw_tw2_w1 = (
        np.array([alpha * xy_half_extents(half_side)[0], alpha * xy_half_extents(half_side)[1], Tw2_w1[2]])
        * object_dist_check[0:3]
    )

    # TSR_params["front"] = (Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1)
    TSR_params["front"] = (Tews, Bw, half_side, Tw2_w1, yaw_tw2_w1)

    # return Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1
    return TSR_params


def ur10_TSR_parameters(object_details, yaw_buffer, alpha):
    # object_position = object_details["position"]
    object_size = object_details["size"]
    object_dist = object_details["dist"]
    TSR_params = {}
    # Top TSR params

    # Panda specifications
    ee_z_offset = -0.04
    # ee_z_offset = 0
    s_f = 0.05

    Tew = np.eye(4)
    Tew[1, 1] = -1
    Tew[2, 2] = -1

    if object_details["type"] == "box":
        Tew[2, 3] = ee_z_offset + object_size[2] / 2
        del_geom = s_f
        clearance_size_x, clearance_size_y = effective_xy_size_for_grasp_clearance(
            object_details, gripper_width=2 * s_f
        )
        del_geom_x = s_f - (clearance_size_x / 2)
        del_geom_y = s_f - (clearance_size_y / 2)
    else: #cylinder
        Tew[2, 3] = ee_z_offset + object_size[1] / 2
        del_geom = s_f
        del_geom_x = s_f - (object_size[0])
        del_geom_y = s_f - (object_size[0])
    # Tew[2, 3] = ee_z_offset + object_size[2] / 4

    yaw_angles = yaw_angles_for_object_grasp(object_details, gripper_width=2 * s_f)
    Tews = make_Tew_yaw_variants(Tew, yaw_angles)

    Bw = np.array(
        [
            [-del_geom_x, del_geom_x],
            [-del_geom_y, del_geom_y],
            [0, 0],
            [0, 0],
            [0, 0],
            [0 - yaw_buffer, 0 + yaw_buffer],
        ]
    )

    # Approximating the intersection of all rotated TSR's with a conservative TSR
    half_side = translational_half_extents(del_geom_x, del_geom_y)

    object_dist_check = np.sign(np.array(object_dist))
    del_Bw = np.array(
        [
            (Bw[0, 1] - Bw[0, 0]) / 2,
            (Bw[1, 1] - Bw[1, 0]) / 2,
            (Bw[2, 1] - Bw[2, 0]) / 2,
            (Bw[5, 1] - Bw[5, 0]) / 2,
        ]
    )
    del_Bw = object_dist_check * del_Bw
    Tw2_w1 = alpha * del_Bw

    yaw_tw2_w1 = (
        np.array([alpha * xy_half_extents(half_side)[0], alpha * xy_half_extents(half_side)[1], Tw2_w1[2]])
        * object_dist_check[0:3]
    )

    # TSR_params["top"] = (Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1)
    TSR_params["top"] = (Tews, Bw, half_side, Tw2_w1, yaw_tw2_w1)

    # Front TSR params
    obj_offset = np.sqrt(object_size[0] ** 2 + object_size[1] ** 2) / 1
    l_f = 0.05 / 2
    ee_offset = l_f * -0.75

    # Tew = np.eye(4)
    # Tew[1, 1] = -1
    # Tew[2, 2] = -1
    # Tew[2, 3] = ee_z_offset + object_position[2]/2
    # Tew[2, 3] = ee_z_offset + object_size[2]/4

    """
    Tew = np.array([
        [ np.cos(np.pi/2), 0,  np.sin(np.pi/2), 0],
        [ 0, 1,  0, 0],
        [-np.sin(np.pi/2), 0,  np.cos(np.pi/2), 0],
        [0, 0, 0, 1]
    ])
    """
    Ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    Rz_m90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
    Rxm90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    # R_new = Ry90 @ Rz_m90
    R_new = Ry90 @ Rxm90
    Tew = np.eye(4)
    Tew[:3, :3] = R_new

    # Tew[0, 3] = -1*(ee_offset)
    ee_offset_eeframe = np.array([0.0, 0.0, -ee_offset])
    Tew[:3, 3] = R_new @ ee_offset_eeframe

    #Tews = make_Tew_x_variants(Tew)
    yaw_angles = yaw_angles_for_object_grasp(object_details, gripper_width=2 * l_f)
    Tews = make_Tew_yaw_variants(Tew, yaw_angles)

    clearance_size_x, clearance_size_y = effective_xy_size_for_grasp_clearance(
        object_details, gripper_width=2 * l_f
    )
    del_geom_x = l_f - (clearance_size_x / 2)
    del_geom_y = l_f - (clearance_size_y / 2)

    Bw = np.array(
        [
            [-del_geom_x, del_geom_x],
            [-del_geom_y, del_geom_y],
            [0, 0],
            [0 - yaw_buffer, 0 + yaw_buffer],
            [0, 0],
            [0, 0],
        ]
    )

    # Approximating the intersection of all rotated TSR's with a conservative TSR
    half_side = translational_half_extents(del_geom_x, del_geom_y)

    object_dist_check = np.sign(np.array(object_dist))
    del_Bw = np.array(
        [
            (Bw[0, 1] - Bw[0, 0]) / 2,
            (Bw[1, 1] - Bw[1, 0]) / 2,
            (Bw[2, 1] - Bw[2, 0]) / 2,
            (Bw[3, 1] - Bw[3, 0]) / 2,
        ]
    )
    del_Bw = object_dist_check * del_Bw
    Tw2_w1 = alpha * del_Bw

    yaw_tw2_w1 = (
        np.array([alpha * xy_half_extents(half_side)[0], alpha * xy_half_extents(half_side)[1], Tw2_w1[2]])
        * object_dist_check[0:3]
    )

    # TSR_params["front"] = (Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1)
    TSR_params["front"] = (Tews, Bw, half_side, Tw2_w1, yaw_tw2_w1)

    # return Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1
    return TSR_params

def rmin_rmax_from_xy_interval(x_interval, y_interval, origin=(0.0, 0.0)):
    ox, oy = origin

    xmin, xmax = x_interval
    ymin, ymax = y_interval

    corners = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymin],
        [xmax, ymax],
    ], dtype=float)

    dists = np.linalg.norm(corners - np.array([ox, oy]), axis=1)
    rmax = float(np.max(dists))

    # For rmin, closest point in rectangle to origin
    closest_x = np.clip(ox, xmin, xmax)
    closest_y = np.clip(oy, ymin, ymax)
    rmin = float(np.linalg.norm([closest_x - ox, closest_y - oy]))

    return rmin, rmax

def tile_in_reachable_annulus(tile, robot_pos, inner_rad, outer_rad):
    rmin, rmax = rmin_rmax_from_xy_interval(
        tile["x"],
        tile["y"],
        origin=(robot_pos[0], robot_pos[1]),
    )

    return (rmin <= outer_rad) and (rmax >= inner_rad)

def tile_inside_intervals(tile, intervals):

    if not intervals:
        return True

    xmin, xmax = intervals[0]
    ymin, ymax = intervals[1]

    return (
        min(tile["x"]) >= xmin and
        max(tile["x"]) <= xmax and
        min(tile["y"]) >= ymin and
        max(tile["y"]) <= ymax
    )

def tile_center(tile):
    return {
        dim: 0.5 * (bounds[0] + bounds[1])
        for dim, bounds in tile.items()
    }


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

def create_TCR_set(env, batch_idx=None):
    env_details = env.env_details
    object_details = env.object_details
    grasp_details = env.grasp_details

    env_name = env.env_details['env_name']

    # print(f"tcr_intervals first: {grasp_details['tcr_intervals']}")
    if batch_idx is not None:
        tcr_intervals = grasp_details['tcr_intervals'][batch_idx]
    else:
        tcr_intervals = grasp_details['tcr_intervals']
    # print(f"tcr_intervals: {tcr_intervals}")
    variation = object_details['variation']

    # print(env_details['intervals'])

    # print(f"tcr_intervals: {tcr_intervals}")

    bin_widths = {}
    for bin_name, bin_intervals in tcr_intervals.items():
        if len(bin_intervals)==1:
            bin_widths[bin_name] = 0
            continue
        bin_widths[bin_name] = bin_intervals[2] - bin_intervals[0]

    # print(f"bin_widths: {bin_widths}")
    
    z_corrections = env.env_details.get("z_correction", [0])

    base_zmin = variation["z"][0][0]
    base_zmax = variation["z"][0][1]

    # print(f"variation: {variation}")    
    # print(f"base_zmin: {base_zmin}")
    # print(f"base_zmax: {base_zmax}")    
    # print(f"z_corrections: {z_corrections}")

    variations_to_apply = variation.copy()

    variations_to_apply["z"] = [
        [base_zmin + dz, base_zmax + dz]
        for dz in z_corrections
    ]

    regions = {
        'x': variations_to_apply['x'],
        'y': variations_to_apply['y'],
        'z': variations_to_apply['z'],
        'yaw': variations_to_apply['yaw'],
    }

    if env_name == "microwave":
        regions['door'] = variations_to_apply['door']

    print(f"regions: {regions}")    
    
    # tiles = list(tile_region(regions, bin_widths))
    # tiles, n_tiles = tile_region(regions, bin_widths)

    per_dim_bins, n_tiles = find_bins_per_dim(regions, bin_widths)

    valid_tiles = []
    robot_pos = env_details['robot_pos']
    inner_rad = env_details['inner_rad']
    outer_rad = env_details['outer_rad']
    robot = env_details['robot']

    env_name = env.env_details['env_name']
    if env_name != "free":
        intervals = env.env_details['intervals']
    else:
        intervals = None

    x_bins = per_dim_bins["x"]
    y_bins = per_dim_bins["y"]
    z_bins = per_dim_bins["z"]
    yaw_bins = per_dim_bins["yaw"]
    door_bins = per_dim_bins.get("door", [None])

    xy_total = len(x_bins) * len(y_bins)
    expand_count = len(z_bins) * len(yaw_bins) * len(door_bins)

    with tqdm(total=n_tiles, desc="Validating Task Set") as pbar:
        for x_bin, y_bin in itertools.product(x_bins, y_bins):
            xy_tile = {
                "x": x_bin,
                "y": y_bin,
            }

            valid_xy = True

            if not tile_in_reachable_annulus(xy_tile, robot_pos, inner_rad, outer_rad):
                valid_xy = False

            if env_name in {"largeobj"}:
                inside = tile_center_inside_intervals(xy_tile, intervals)
            else:
                inside = tile_inside_intervals(xy_tile, intervals)

            if not inside:
                valid_xy = False

            # if valid_xy:
            #     for z_bin, yaw_bin, door_bin in itertools.product(z_bins, yaw_bins, door_bins):
            #         tile = {
            #             "x": x_bin,
            #             "y": y_bin,
            #             "z": z_bin,
            #             "yaw": yaw_bin,
            #         }

            #         if door_bin is not None:
            #             tile["door"] = door_bin

            #         valid_tiles.append(tile)
            # pbar.update(expand_count)
            
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
                        raise NotImplementedError(f"Unsupported robot for task validation: {robot} ")

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
                    tile = {
                        "x": x_bin,
                        "y": y_bin,
                        "z": z_bin,
                        "yaw": yaw_bin,
                    }

                    if door_bin is not None:
                        tile["door"] = door_bin


                    if door_bin is None:
                        TCR = (
                            tuple(x_bin),
                            tuple(y_bin),
                            tuple(z_bin),
                            tuple(yaw_bin)
                        )
                    else:
                        TCR = (
                            tuple(x_bin),
                            tuple(y_bin),
                            tuple(z_bin),
                            tuple(yaw_bin),
                            tuple(door_bin)
                        )
                    valid_tiles.append(TCR)

                    # valid_tiles.append(tile)

                pbar.update(len(z_bins))
            

    print(f"valid tiles: {len(valid_tiles)} / {n_tiles}")

    TCR_set = tiles_to_iTSR_set(env, valid_tiles)
    return TCR_set

def yaw_rot_2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s],
        [s,  c],
    ], dtype=float)


def bin_mid(bin_):
    return 0.5 * (float(bin_[0]) + float(bin_[1]))


def microwave_handle_pose_xy(x_bin, y_bin, yaw_bin, door_bin, env):
    x = bin_mid(x_bin)
    y = bin_mid(y_bin)
    yaw = bin_mid(yaw_bin)
    door_phi = bin_mid(door_bin)

    hinge_xy = np.array(
        env.object_details["hinge_pos_body"][:2],
        dtype=float,
    )

    door_origin_xy = np.array(
        env.object_details["door_origin_closed_body"][:2],
        dtype=float,
    )

    handle_pos_door_xy = np.array(
        env.object_details["handle_pos_door"][:2],
        dtype=float,
    )

    R_body = yaw_rot_2d(yaw)
    R_door = yaw_rot_2d(-door_phi)

    handle_body_xy = (
        hinge_xy
        + R_door @ (door_origin_xy + handle_pos_door_xy)
    )

    handle_world_xy = np.array([x, y], dtype=float) + R_body @ handle_body_xy

    # Your microwave front/outward direction in the CLOSED door frame.
    # Since the front is at negative body x, outward from inside microwave
    # through the door is local -x.
    handle_normal_door = np.array([-1.0, 0.0], dtype=float)

    handle_normal_world = R_body @ R_door @ handle_normal_door
    handle_normal_world /= np.linalg.norm(handle_normal_world) + 1e-9

    return handle_world_xy, handle_normal_world

def microwave_handle_filter(
    x_bin,
    y_bin, 
    yaw_bin, 
    door_bin, 
    env,
    dot_min=0.0,
    outer_rad_scale=0.75
):
    
    pass

    robot = env.env_details['robot']

    if robot in {"panda", "fetch"}:
        pass
        handle_pos_xy, handle_normal_xy = microwave_handle_pose_xy(
            x_bin,
            y_bin,
            yaw_bin,
            door_bin,
            env,
        )
        robot_xy = np.array(env.env_details['robot_pos'][:2], dtype=float)
        to_robot = robot_xy - handle_pos_xy
        dist = np.linalg.norm(to_robot)

        if dist < 1e-9:
            return True
        
        to_robot = to_robot / dist
        dot = np.dot(handle_normal_xy, to_robot)

        # Case 1: handle is facing robot
        if dot >= dot_min:
            return True
        # else:
        #     return False
        
        # Case 2: handle is not facing robot but might be reachable
        inner_rad = env.env_details["inner_rad"]
        outer_rad = env.env_details["outer_rad"]
        close_outer_rad = outer_rad_scale * outer_rad

        return inner_rad <= dist <= close_outer_rad
        
    else:
        raise ValueError(f"Microwave handle check not supported for robot: {robot}")



def tile_center_radius(tile, robot_pos):
    cx = 0.5 * (tile["x"][0] + tile["x"][1])
    cy = 0.5 * (tile["y"][0] + tile["y"][1])

    dx = cx - robot_pos[0]
    dy = cy - robot_pos[1]

    return np.sqrt(dx * dx + dy * dy), cx, cy

def tiles_to_iTSR_set(env, valid_tiles):
    return {key: None for key in valid_tiles}

def tile_1d(intervals, width, eps=1e-9):
    """
    intervals: [[lo, hi], [lo2, hi2], ...]
    width: bin width for this dimension
    """
    bins = []

    for lo, hi in intervals:
        lo = float(lo)
        hi = float(hi)

        # fixed dimension, e.g. z = 0.075
        if abs(hi - lo) < eps:
            bins.append([round(lo, 5), round(hi, 5)])
            continue

        if width <= eps:
            raise ValueError(f"Nonzero interval [{lo}, {hi}] needs positive width, got {width}")

        edges = np.arange(lo, hi, width).tolist()

        for a in edges:
            b = min(a + width, hi)
            bins.append([round(a, 5), round(b, 5)])

    return bins

def tile_region(regions, bin_widths):
    per_dim_bins = {
        dim: tile_1d(intervals, bin_widths[dim])
        for dim, intervals in regions.items()
    }

    n_tiles = 1
    for dim, bins in per_dim_bins.items():
        print(f"{dim}: {len(bins)} bins")
        n_tiles *= len(bins)

    print(f"total tiles: {n_tiles}")

    dims = list(per_dim_bins.keys())

    def generator():
        for combo in itertools.product(*(per_dim_bins[d] for d in dims)):
            yield dict(zip(dims, combo))

    return generator(), n_tiles

def find_bins_per_dim(regions, bin_widths):
    per_dim_bins = {
        dim: tile_1d(intervals, bin_widths[dim])
        for dim, intervals in regions.items()
    }

    n_tiles = 1
    for dim, bins in per_dim_bins.items():
        print(f"{dim}: {len(bins)} bins")
        n_tiles *= len(bins)

    print(f"total tiles: {n_tiles}")
    
    return per_dim_bins, n_tiles

def find_yaw_iTSR_set(object_details, problem_details, Tw2_w1):

    object_dist = object_details["dist"]
    object_position = object_details["position"]

    yaw_iTSR_set_all = []
    yaw_to_cover_all = []

    for problem in problem_details:
        curr_problem_details = problem_details[problem]

        half_side = curr_problem_details["half_side"]
        Bw = curr_problem_details["Bw"]
        yaw_buffer = curr_problem_details["yaw_buffer"]

        yaw_is_covered = False
        yaw_to_cover = [round(-object_dist[3], 5), round(object_dist[3], 5)]

        yaw_iTSR_set = {}
        yaw_covered = []
        yaw_key = []

        while yaw_is_covered is False:
            first_pos = [
                object_position[0] - object_dist[0],
                object_position[1] - object_dist[1],
                object_position[2] - object_dist[2],
            ]

            if yaw_iTSR_set == {}:
                yaw_1 = round(-object_dist[3], 5)
            else:
                prev_Tw2_0 = yaw_iTSR_set[yaw_key][2]
                yaw_1 = round(
                    np.arctan2(prev_Tw2_0[1, 0], prev_Tw2_0[0, 0]), 5
                )

            Tw1_0 = find_TSR_HMat(first_pos, yaw_1)

            tw2_0 = Tw1_0[:3, 3]  # Same position as Tw1
            yaw_2 = round(yaw_1 + Tw2_w1[3], 5)

            Tw2_0 = find_TSR_HMat(first_pos, yaw_2)

            B12_yaw_intersect = np.array(
                [
                    [tw2_0[0] - xy_half_extents(half_side)[0], tw2_0[0] + xy_half_extents(half_side)[0]],
                    [tw2_0[1] - xy_half_extents(half_side)[1], tw2_0[1] + xy_half_extents(half_side)[1]],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [yaw_2 - yaw_buffer, yaw_1 + yaw_buffer],
                ]
            )
            B12_yaw_intersect = find_TSR_Bounds(
                Bw, Tw1_0, yaw_1, yaw_2, yaw_buffer, half_side, grasp=problem
            )

            curr_yaw_intervals = [yaw_1, yaw_2]
            yaw_covered.append(curr_yaw_intervals)
            yaw_covered = merge_intervals(yaw_covered)

            yaw_cover_check = (
                find_intersection(yaw_covered[0], yaw_to_cover) == yaw_to_cover
            )
            yaw_is_covered = yaw_cover_check

            yaw_key = tuple(curr_yaw_intervals)
            yaw_iTSR_set[yaw_key] = [B12_yaw_intersect, Tw1_0, Tw2_0]

        yaw_iTSR_set_all.append(yaw_iTSR_set)
        yaw_to_cover_all.append(yaw_to_cover)
        # print(f"yaw key: {yaw_key}")
        # print(f"yaw itsr set: {yaw_iTSR_set[yaw_key]}")
    # return yaw_iTSR_set, yaw_to_cover
    return yaw_iTSR_set_all, yaw_to_cover_all


def find_iTSR_set(
    object_details,
    problem_details,
    yaw_tw2_w1_dict,
    yaw_iTSR_set,
    problem,
    robot_pos=[0.0, 0.0, 0.0],
):

    object_position = object_details["position"]
    object_dist = object_details["dist"]
    object_yaw = object_details["yaw"]

    iTSR_set_all = []
    iterno_all = []

    for grasp_ind, grasp_strategy in enumerate(problem_details):
        print(f"Generating bins for {grasp_strategy} grasping strategy...")
        curr_problem_details = problem_details[grasp_strategy]

        half_side = curr_problem_details["half_side"]
        Bw = curr_problem_details["Bw"]
        yaw_buffer = curr_problem_details["yaw_buffer"]
        alpha = curr_problem_details["alpha"]
        reachable_ws = curr_problem_details["reachable_ws"]
        robot_clearance = curr_problem_details["robot_clearance"]

        yaw_tw2_w1 = yaw_tw2_w1_dict[grasp_strategy]

        object_upper = [
            round(object_position[0] + object_dist[0], 5),
            round(object_position[1] + object_dist[1], 5),
            round(object_position[2] + object_dist[2], 5),
            round(object_yaw + object_dist[3], 5),
        ]
        object_lower = [
            round(object_position[0] - object_dist[0], 5),
            round(object_position[1] - object_dist[1], 5),
            round(object_position[2] - object_dist[2], 5),
            round(object_yaw - object_dist[3], 5),
        ]

        dist_to_cover = [
            [object_lower[0], object_upper[0]],
            [object_lower[1], object_upper[1]],
            [object_lower[2], object_upper[2]],
        ]

        dims = []
        for i in range(len(dist_to_cover)):
            if dist_to_cover[i][0] != dist_to_cover[i][1]:
                dims.append(1)
            else:
                dims.append(0)

        if dims[0] == 1 and dims[1] == 1:
            x_half, y_half = xy_half_extents(half_side)
            yaw_tw2_w1_x = np.array([alpha * x_half, 0, 0])
            yaw_tw2_w1_y = np.array([0, alpha * y_half, 0])

        x_covered = True
        iTSR_set = {}
        iterno = 0
        # print(yaw_iTSR_set[grasp_ind])
        for curr_yaw_interval in tqdm(yaw_iTSR_set[grasp_ind]):

            curr_yaw_iTSR = yaw_iTSR_set[grasp_ind][curr_yaw_interval][0]
            yaw_1 = curr_yaw_interval[0]
            yaw_2 = curr_yaw_interval[1]
            # print(f"yaw1: {yaw_1}")
            # print(f"yaw2: {yaw_2}")

            curr_yaw_iTSR_set = {}
            curr_yaw_dist_covered = [[], [], []]
            curr_yaw_is_covered = False

            while curr_yaw_is_covered is False:
                if curr_yaw_iTSR_set == {}:
                    Tw1_0 = np.eye(4)
                    Tw1_0[0, 3] = object_position[0] - object_dist[0]
                    Tw1_0[1, 3] = object_position[1] - object_dist[1]
                    Tw1_0[2, 3] = object_position[2] - object_dist[2]

                    Tw1_0 = find_TSR_HMat(
                        [
                            object_position[0] - object_dist[0],
                            object_position[1] - object_dist[1],
                            object_position[2] - object_dist[2],
                        ],
                        0,
                    )
                else:
                    if dims[0] == 1 and dims[1] == 1:
                        if x_covered:
                            Tw1_0 = x_pivot
                        else:
                            Tw1_0 = curr_yaw_iTSR_set[curr_yaw_key][3]
                    else:

                        Tw1_0 = curr_yaw_iTSR_set[curr_yaw_key][2]

                tw1_0 = Tw1_0[:3, 3]
                tw2_0 = tw1_0 + yaw_tw2_w1

                Tw2_0 = np.eye(4)
                Tw2_0[:3, 3] = tw2_0

                B1_0 = find_TSR_Bounds(
                    Bw,
                    Tw1_0,
                    yaw_1,
                    yaw_2,
                    yaw_buffer,
                    half_side,
                    grasp=grasp_strategy,
                )
                B2_0 = find_TSR_Bounds(
                    Bw,
                    Tw2_0,
                    yaw_1,
                    yaw_2,
                    yaw_buffer,
                    half_side,
                    grasp=grasp_strategy,
                )
                B12_intersect = find_B0_intersection(B1_0, B2_0)
                # print(f"B12 intersect: {B12_intersect}")

                if dims[0] == 1 and dims[1] == 1:
                    tw3_0 = tw1_0 + yaw_tw2_w1_x  # x translation
                    tw4_0 = tw1_0 + yaw_tw2_w1_y  # y translation

                    Tw3_0 = find_TSR_HMat(tw3_0, 0)
                    Tw4_0 = find_TSR_HMat(tw4_0, 0)

                    B3_0 = find_TSR_Bounds(
                        Bw,
                        Tw3_0,
                        yaw_1,
                        yaw_2,
                        yaw_buffer,
                        half_side,
                        grasp=grasp_strategy,
                    )
                    B4_0 = find_TSR_Bounds(
                        Bw,
                        Tw4_0,
                        yaw_1,
                        yaw_2,
                        yaw_buffer,
                        half_side,
                        grasp=grasp_strategy,
                    )

                    B34_intersect = find_B0_intersection(B3_0, B4_0)
                    B12_intersect = find_B0_intersection(
                        B34_intersect, B12_intersect
                    )

                    # print(f"B34 intersect: {B34_intersect}")
                    # print(f"B intersect: {B12_intersect}")
                    # return None
                    if x_covered:
                        x_pivot = Tw4_0
                        dist_covered_pivot = [[], [], []]
                        x_covered = False

                curr_intervals = [
                    [round(tw1_0[0], 5), round(tw2_0[0], 5)],
                    [round(tw1_0[1], 5), round(tw2_0[1], 5)],
                    [round(tw1_0[2], 5), round(tw2_0[2], 5)],
                    [round(yaw_1, 5), round(yaw_2, 5)],
                ]

                if dims[0] == 1 and dims[1] == 1:

                    dist_covered_pivot[0].append(curr_intervals[0])
                    dist_covered_pivot[1].append(curr_intervals[1])
                    dist_covered_pivot[2].append(curr_intervals[2])
                    # print(dist_covered_pivot)
                    dist_covered_pivot = [
                        merge_intervals(dist_covered_pivot[0]),
                        merge_intervals(dist_covered_pivot[1]),
                        merge_intervals(dist_covered_pivot[2]),
                    ]

                    x_covered = (
                        find_intersection(
                            dist_to_cover[0], dist_covered_pivot[0][0]
                        )
                        == dist_to_cover[0]
                    )

                    if x_covered:
                        curr_yaw_dist_covered[0].append(dist_covered_pivot[0])
                        curr_yaw_dist_covered[1].append(dist_covered_pivot[1])
                        curr_yaw_dist_covered[2].append(dist_covered_pivot[2])
                        # print(curr_yaw_dist_covered)
                        curr_yaw_dist_covered = [
                            merge_intervals(curr_yaw_dist_covered[0]),
                            merge_intervals(curr_yaw_dist_covered[1]),
                            merge_intervals(curr_yaw_dist_covered[2]),
                        ]
                        # print(curr_yaw_dist_covered)

                        curr_yaw_cover_check = [
                            find_intersection(
                                dist_to_cover[0], curr_yaw_dist_covered[0][0]
                            )
                            == dist_to_cover[0],
                            find_intersection(
                                dist_to_cover[1], curr_yaw_dist_covered[1][0]
                            )
                            == dist_to_cover[1],
                            find_intersection(
                                dist_to_cover[2], curr_yaw_dist_covered[2][0]
                            )
                            == dist_to_cover[2],
                        ]
                        curr_yaw_is_covered = all(curr_yaw_cover_check)
                    else:
                        curr_yaw_is_covered = False

                else:

                    curr_yaw_dist_covered[0].append(curr_intervals[0])
                    curr_yaw_dist_covered[1].append(curr_intervals[1])
                    curr_yaw_dist_covered[2].append(curr_intervals[2])

                    curr_yaw_dist_covered = [
                        merge_intervals(curr_yaw_dist_covered[0]),
                        merge_intervals(curr_yaw_dist_covered[1]),
                        merge_intervals(curr_yaw_dist_covered[2]),
                    ]

                    curr_yaw_cover_check = [
                        find_intersection(
                            dist_to_cover[0], curr_yaw_dist_covered[0][0]
                        )
                        == dist_to_cover[0],
                        find_intersection(
                            dist_to_cover[1], curr_yaw_dist_covered[1][0]
                        )
                        == dist_to_cover[1],
                        find_intersection(
                            dist_to_cover[2], curr_yaw_dist_covered[2][0]
                        )
                        == dist_to_cover[2],
                    ]

                    curr_yaw_is_covered = all(curr_yaw_cover_check)

                # print(f"Current yaw coverage check: {curr_yaw_is_covered}")

                key = tuple(tuple(row) for row in curr_intervals)
                curr_yaw_key = tuple(tuple(row) for row in curr_intervals[0:3])

                tw1_rvec = np.sqrt(
                    (tw1_0[0] - object_position[0]) ** 2
                    + (tw1_0[1] - object_position[1]) ** 2
                )

                #rmin, rmax = rmin_rmax_from_rectangle_corners_2d(
                #    tw1_0, tw2_0, nominal_pose=robot_pos
                # )

                if problem['robot'] == 'fetch':
                    #print("Using 2D check")
                    rmin, rmax = rmin_rmax_from_rectangle_corners_2d(
                        tw1_0, tw2_0, nominal_pose=robot_pos
                    )
                else:
                    rmin, rmax = rmin_rmax_from_box_corners(
                        tw1_0, tw2_0, nominal_pose=robot_pos
                    )
                in_sample_space = (rmin <= reachable_ws) and (
                    rmax >= robot_clearance
                )
                # print(f"rmin {rmin}, rmax {rmax}")

                if problem["name"] is None:
                    in_problem = True
                else:
                    problem_name = problem["name"]

                    if (
                        problem_name == "box"
                        or problem_name == "cage"
                        or problem_name == "table"
                    ):
                        xmin = problem["intervals"][0][0]
                        xmax = problem["intervals"][0][1]
                        ymin = problem["intervals"][1][0]
                        ymax = problem["intervals"][1][1]

                        if (
                            xmin <= tw1_0[0] <= xmax
                            and ymin <= tw1_0[1] <= ymax
                            and xmin <= tw2_0[0] <= xmax
                            and ymin <= tw2_0[1] <= ymax
                        ):
                            in_problem = True
                        else:
                            in_problem = False

                    elif problem_name == "shelf":
                        in_problem = False
                        for region in problem["intervals"]:
                            xmin = region[0][0]
                            xmax = region[0][1]
                            ymin = region[1][0]
                            ymax = region[1][1]

                            if (
                                xmin <= tw1_0[0] <= xmax
                                and ymin <= tw1_0[1] <= ymax
                                and xmin <= tw2_0[0] <= xmax
                                and ymin <= tw2_0[1] <= ymax
                            ):
                                in_problem = True
                                break

                # print(f"in_sample_space: {in_sample_space}, in_problem: {in_problem}")
                if dims[0] == 1 and dims[1] == 1:
                    curr_yaw_iTSR_set[curr_yaw_key] = [
                        B12_intersect,
                        Tw1_0,
                        Tw2_0,
                        Tw3_0,
                        Tw4_0,
                    ]
                    if (in_sample_space) and (in_problem):
                        iTSR_set[key] = [
                            B12_intersect,
                            Tw1_0,
                            Tw2_0,
                            Tw3_0,
                            Tw4_0,
                        ]

                else:
                    curr_yaw_iTSR_set[curr_yaw_key] = [
                        B12_intersect,
                        Tw1_0,
                        Tw2_0,
                    ]
                    if (in_sample_space) and (in_problem):
                        iTSR_set[key] = [B12_intersect, Tw1_0, Tw2_0]

                iterno += 1

            # print(f"Yaw interval covered: {yaw_1} to {yaw_2}")
        # print(f"Generated {len(iTSR_set)} bins.")
        iTSR_set_all.append(iTSR_set)
        iterno_all.append(iterno)

    # return iTSR_set, iterno
    return iTSR_set_all, iterno_all
