import numpy as np
from tqdm import tqdm  # type: ignore


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


def find_TSR_Bounds(Bw, TSR_HMat1, yaw_1, yaw_2, yaw_buffer, half_side, grasp):
    tw1_0 = TSR_HMat1[:3, 3]

    if grasp == "top":
        B = np.array(
            [
                [tw1_0[0] - half_side, tw1_0[0] + half_side],
                [tw1_0[1] - half_side, tw1_0[1] + half_side],
                [tw1_0[2] + Bw[2, 0], tw1_0[2] + Bw[2, 1]],
                [Bw[3, 0], Bw[3, 1]],
                [Bw[4, 0], Bw[4, 1]],
                [yaw_2 - yaw_buffer, yaw_1 + yaw_buffer],
            ]
        )
    else:  # front
        B = np.array(
            [
                [tw1_0[0] - half_side, tw1_0[0] + half_side],
                [tw1_0[1] - half_side, tw1_0[1] + half_side],
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
    if zmin <= zmax:
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


def rmin_rmax_from_square_corners(tw1, tw2, nominal_pose=[0, 0, 0]):
    x_min = tw1[0]
    y_min = tw1[1]
    x_max = tw2[0]
    y_max = tw2[1]

    if (x_min <= nominal_pose[0] <= x_max) and (
        y_min <= nominal_pose[1] <= y_max
    ):
        r_min = 0.0
    else:
        nearest_x = np.clip(nominal_pose[0], x_min, x_max)
        nearest_y = np.clip(nominal_pose[1], y_min, y_max)
        r_min = np.sqrt(nearest_x**2 + nearest_y**2)

    corners = np.array(
        [
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_min],
            [x_max, y_max],
        ]
    )
    nominal_pose_mat = np.array(
        [
            nominal_pose[0:2],
            nominal_pose[0:2],
            nominal_pose[0:2],
            nominal_pose[0:2],
        ]
    )

    r2 = (corners**2).sum(axis=1)
    r_max = np.sqrt(r2.max())

    return r_min, r_max


def panda_TSR_parameters(object_details, yaw_buffer, alpha):
    object_position = object_details["position"]
    object_size = object_details["size"]
    object_dist = object_details["dist"]
    TSR_params = {}
    # Top TSR params

    # Panda specifications
    ee_z_offset = 0.02
    # ee_z_offset = 0
    s_f = 0.04

    Tew = np.eye(4)
    Tew[1, 1] = -1
    Tew[2, 2] = -1
    Tew[2, 3] = ee_z_offset + object_size[2] / 2
    # Tew[2, 3] = ee_z_offset + object_size[2] / 4

    del_geom = s_f
    del_geom_x = s_f - (object_size[0] / 2)
    del_geom_y = s_f - (object_size[1] / 2)

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
    half_side = 0.5 * np.sqrt(2) * 0.5 * min(2 * del_geom_x, 2 * del_geom_y)

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
        np.array([alpha * half_side, alpha * half_side, Tw2_w1[2]])
        * object_dist_check[0:3]
    )

    TSR_params["top"] = (Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1)

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
    R_new = Ry90 @ Rz_m90
    # R_new = Ry90
    Tew = np.eye(4)
    Tew[:3, :3] = R_new

    # Tew[0, 3] = -1*(ee_offset)
    ee_offset_eeframe = np.array([0.0, 0.0, -ee_offset])
    Tew[:3, 3] = R_new @ ee_offset_eeframe

    del_geom_x = l_f - (object_size[0] / 2)
    del_geom_y = l_f - (object_size[1] / 2)

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
    half_side = 0.5 * np.sqrt(2) * 0.5 * min(2 * del_geom_x, 2 * del_geom_y)

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
        np.array([alpha * half_side, alpha * half_side, Tw2_w1[2]])
        * object_dist_check[0:3]
    )

    TSR_params["front"] = (Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1)

    # return Tew, Bw, half_side, Tw2_w1, yaw_tw2_w1
    return TSR_params


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
                    [tw2_0[0] - half_side, tw2_0[0] + half_side],
                    [tw2_0[1] - half_side, tw2_0[1] + half_side],
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
    problem=None,
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
            yaw_tw2_w1_x = np.array([alpha * half_side, 0, 0])
            yaw_tw2_w1_y = np.array([0, alpha * half_side, 0])

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

                rmin, rmax = rmin_rmax_from_square_corners(
                    tw1_0, tw2_0, nominal_pose=robot_pos
                )
                in_sample_space = (rmin <= reachable_ws) and (
                    rmax >= robot_clearance
                )

                if problem is None:
                    in_problem = True
                else:
                    problem_name = problem["name"]

                    if problem_name == "box":
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
