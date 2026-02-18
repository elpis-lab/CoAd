import numpy as np


def condense_dataset(model, data, f_prefix, object_details, obj_geom_list):
    """
    Takes a generated data structure and condenses it using a Prefix-Suffix strategy
    Outputs a collection of root paths and a collection of subpaths as references to the root paths
    """

    object_size = object_details["size"]
    robot_geoms = get_robot_collision_geom_ids(model)
    mujocoViewer = MujocoViewer(model, data, robot_geoms) if viewer else None
    ik_solver = PandaMinkIK(model, data, robot_geoms, obj_geom_list)

    # if viewer:
    #    mujocoViewer.open()

    # saved datastructures
    root_paths = {}
    prefix_map = {}

    roots_idx = []

    data, offsets, keys_arr = load_store(f_prefix, mmap_data=True)
    print(f"Number of bins: {len(keys_arr)}")
    indexer = SparseBoxGrid4D(keys_arr)
    plotter = Plotter(keys_arr, indexer)

    covered_key_indices = set()
    uncovered_key_indices = set(range(len(keys_arr)))
    always_invalid = set(np.where(offsets[1:] == offsets[:-1])[0])
    blacklist = set()

    recent_coverages = []

    exit_factor = 0.05
    rho = 0.01
    G_best_min = 25
    W_max = 20
    max_checked_bins = min(10000, int(0.15 * len(keys_arr)))
    max_suffix_checks = min(1000, int(0.05 * len(keys_arr)))

    fail_streak = 0
    FAIL_STREAK_MAX = 100
    A_accept_max = min(2000, max(100, int(0.005 * len(keys_arr))))

    while uncovered_key_indices:

        if len(uncovered_key_indices) > int(0.75 * len(keys_arr)):
            K_abs = 40
            fail_bridge_budget = 0
        elif len(uncovered_key_indices) > int(0.5 * len(keys_arr)):
            K_abs = 20
            fail_bridge_budget = 500
        else:
            K_abs = 5
            fail_bridge_budget = 1000

        covered_idx = []
        prefix_lengths = []
        covered_goals = []
        checked_bins = set()
        suffix_checks = 0
        # fail_bridge_budget = 1000
        # fail_bridge_budget = 0

        print("Sampling a new root path...")

        # Base pool: uncovered, excluding invalid and blacklisted
        pool = list(uncovered_key_indices - always_invalid - blacklist)

        if not pool:
            idx = random.choice(tuple(uncovered_key_indices))
        else:
            cands = [
                i
                for i in pool
                if any(
                    (n in uncovered_key_indices)
                    for n in indexer.adjacent_neighbors(keys_arr[i])
                )
            ]
            idx = random.choice(cands if cands else pool)

        """
        cands = [i for i in uncovered_key_indices
                if any(n not in covered_key_indices
                    for n in indexer.adjacent_neighbors(keys_arr[i]))]

        if not cands:
            idx = random.choice(tuple(uncovered_key_indices))
        else:
            idx = random.choice(cands)
        """

        path = get_path_by_index(data, offsets, idx)

        # if (path is None):
        if path is None or getattr(path, "size", 0) == 0:
            print("Empty candidate root. Skipping...")
            always_invalid.add(idx)
            fail_streak += 1
            continue

        # pathPrefix = PrefixBuilder(scene, robot, SV, len(path))
        # pathSuffix = SuffixBuilder(scene, robot, SV, len(path))

        pathPrefix = MjPrefix(
            model, data, robot_geoms, mujocoViewer, len(path)
        )
        pathSuffix = MjSuffix(
            model,
            data,
            robot_geoms,
            obj_geom_list,
            mujocoViewer,
            len(path),
        )

        # Uncovered neighbors of root
        # neighbor_idx = indexer.neighbors_by_box(keys_arr[idx], R_meters=0.1, R_yaw=0.3)
        neighbor_idx = indexer.adjacent_neighbors(keys_arr[idx])
        neighbor_idx = list(dict.fromkeys(neighbor_idx))

        # neighbor_idx = [n for n in neighbor_idx if n != idx and n not in covered_key_indices]
        neighbor_idx = [n for n in neighbor_idx if n != idx]
        neighbor_idx = [n for n in neighbor_idx if n not in checked_bins]
        # print("uncovered neighbor count:", len(neighbor_idx))

        cleared_queue = len(neighbor_idx) == 0

        while not cleared_queue:
            # print(f"Checking {len(neighbor_idx)} neighbors...")
            to_check_suffix = [
                n
                for n in neighbor_idx
                if n not in covered_key_indices and n not in checked_bins
            ]
            print(
                f"Layer: total={len(neighbor_idx)}, "
                f"suffix_checks={len(to_check_suffix)}, "
                f"checked_total={len(checked_bins)}"
            )

            next_layer = []
            covered_before = len(covered_idx)

            for i in range(len(neighbor_idx)):

                curr_bin = keys_arr[neighbor_idx[i]]
                prefix_len = pathPrefix.find_prefix_len(path, curr_bin)
                # print(f"prefix_len {prefix_len}")
                # pathPrefix.visualize_prefix(path[:prefix_len])

                n_bin_path = get_path_by_index(data, offsets, neighbor_idx[i])

                if n_bin_path.size == 0:
                    n_bin_goal = None
                else:
                    n_bin_goal = n_bin_path[-1]

                checked_bins.add(neighbor_idx[i])
                is_globally_covered = neighbor_idx[i] in covered_key_indices

                if n_bin_goal is not None:

                    if not is_globally_covered:

                        suffix_checks += 1

                        suffix_start = time.time()
                        # suffix = pathSuffix.ik_suffix(path, prefix_len, n_bin_goal)
                        suffix = pathSuffix.ik_suffix_single(
                            path, prefix_len, n_bin_goal, ik_solver
                        )
                        # print(f"Precollision check time: {time.time() - suffix_start}")
                        # print("Checking suffix for collisions")

                        if suffix is not None:
                            dense_suffix = densify_q_traj(
                                suffix, max_step=0.02
                            )
                            for q in dense_suffix:
                                set_panda_qpos(model, data, q)
                                in_contact, _ = robot_in_contact(
                                    model, data, robot_geoms
                                )
                                within_limits, _, _, _, _ = qpos_within_limits(
                                    model, q
                                )
                                # mujocoViewer.viewer.sync()
                                # input("Continue?")
                                if in_contact or not within_limits:
                                    suffix = None
                                    break

                    else:
                        suffix = None

                else:
                    # print("Empty neighbor bin")
                    suffix = None

                if (suffix is not None) and (not is_globally_covered):
                    # print(f"Suffix found. Time with collision checks: {time.time() - suffix_start}")
                    # print(f"Suffix time: {time.time() - suffix_start}")
                    # pathSuffix.visualize_suffix(suffix)

                    if viewer is True:
                        print("Visualizing entire path")
                        prefix = path[:prefix_len]
                        qpos_traj = np.vstack([prefix, suffix])
                        mujocoViewer.play_qpos_traj(qpos_traj)

                    covered_idx.append(neighbor_idx[i])
                    prefix_lengths.append(prefix_len)
                    covered_goals.append(n_bin_goal)

                if (suffix is not None) or is_globally_covered:
                    next_layer.extend(
                        indexer.adjacent_neighbors(keys_arr[neighbor_idx[i]])
                    )

                else:
                    if fail_bridge_budget > 0:
                        next_layer.extend(
                            indexer.adjacent_neighbors(
                                keys_arr[neighbor_idx[i]]
                            )
                        )
                        fail_bridge_budget -= 1

            print(f"Layer added. Covered: {len(covered_idx)-covered_before}")
            print(f"Current candidate coverage: {len(covered_idx)}")
            # plotter.plot_bin_and_neighbors_3d_shell(idx, covered_idx)

            if len(next_layer) > 0:
                cleared_queue = False

                # next_layer = [n for n in next_layer if n != idx and n not in covered_key_indices]
                next_layer = [n for n in next_layer if n != idx]
                next_layer = list(dict.fromkeys(next_layer))
                next_layer = [n for n in next_layer if n not in checked_bins]
                neighbor_idx = next_layer
            else:
                cleared_queue = True

            if (
                len(checked_bins) > max_checked_bins
                or suffix_checks > max_suffix_checks
            ):
                cleared_queue = True
                if (
                    len(covered_idx) == 0
                    and suffix_checks >= 0.8 * max_suffix_checks
                    and len(checked_bins) >= 0.8 * max_checked_bins
                ):
                    print(f"Blacklisting bin: {idx}")
                    blacklist.add(idx)
                break

        print(f"Candidate root covered {len(covered_idx)} paths")

        is_root = len(covered_idx) >= 1
        # is_root = len(covered_idx) >= max(K_abs, np.ceil(rho * len(checked_bins)))
        is_root = len(covered_idx) >= K_abs

        U = len(uncovered_key_indices)
        A_max = int(500 + 30 * np.sqrt(U))
        G_min = max(2, int(0.01 * len(keys_arr)))

        if is_root:

            # print(f"Root path added {idx} - covers {len(covered_idx)} paths")
            curr_root_idx = len(roots_idx)
            root_paths[curr_root_idx] = path
            tuple_idx = tuple(tuple(row) for row in keys_arr[idx])
            prefix_map[tuple_idx] = (f"{curr_root_idx}", path[-1].tolist())

            for i, i_cover in enumerate(covered_idx):
                tuple_idx = tuple(tuple(row) for row in keys_arr[i_cover])
                # prefix_map[tuple(keys_arr[i_cover])] = f"{curr_root_idx}-{prefix_lengths[i]}"
                prefix_map[tuple_idx] = (
                    f"{curr_root_idx}-{prefix_lengths[i]}",
                    covered_goals[i].tolist(),
                )

            newly = set(covered_idx)
            newly.add(idx)
            covered_key_indices |= newly
            uncovered_key_indices -= newly

            # covered_key_indices.update(covered_idx)
            roots_idx.append(idx)
            print(f"Root path added - covers {len(covered_idx)} paths.")
            print(f"Current number of root paths: {len(roots_idx)}")
            print(
                f"Current number of covered paths: {len(covered_key_indices)}"
            )

            recent_coverages.append(len(covered_idx))

            if len(recent_coverages) > W_max:
                recent_coverages.pop(0)

            fail_streak = 0

        else:
            print(f"Root path rejected {idx}")
            fail_streak += 1

        # Exit condition

        half_tail = len(uncovered_key_indices) <= 0.5 * len(keys_arr)
        if half_tail:
            print(
                f"Half tail reached. Adding remaining {len(uncovered_key_indices)} bins as root paths. Current number of root paths: {len(root_paths)}"
            )
            root_paths, prefix_map = add_remaining_bins(
                data,
                offsets,
                uncovered_key_indices,
                root_paths,
                prefix_map,
                keys_arr,
                roots_idx,
                covered_key_indices,
            )

            break

        tail = len(uncovered_key_indices) <= exit_factor * len(keys_arr)
        if tail:
            if len(recent_coverages) >= 5:
                W = min(W_max, max(5, len(recent_coverages)))
                window = recent_coverages[-W:]
                window_sorted = sorted(window)
                G_best = (
                    window_sorted[-2]
                    if len(window_sorted) >= 2
                    else window_sorted[-1]
                )

                if G_best <= G_best_min or fail_streak >= FAIL_STREAK_MAX:
                    print(
                        f"Exit condition reached (uncovered={len(uncovered_key_indices)}, G_best={G_best:.1f})."
                    )
                    root_paths, prefix_map = add_remaining_bins(
                        data,
                        offsets,
                        uncovered_key_indices,
                        root_paths,
                        prefix_map,
                        keys_arr,
                        roots_idx,
                        covered_key_indices,
                    )

                    break
        else:

            if fail_streak >= A_accept_max:
                print(
                    "No progress exit: promoting remaining uncovered bins to roots."
                )
                root_paths, prefix_map = add_remaining_bins(
                    data,
                    offsets,
                    uncovered_key_indices,
                    root_paths,
                    prefix_map,
                    keys_arr,
                    roots_idx,
                    covered_key_indices,
                )
                break

            if len(recent_coverages) >= 5:
                W = min(W_max, len(recent_coverages))
                window = recent_coverages[-W:]
                G_best = np.percentile(window, 90)

                if fail_streak >= A_max and G_best <= G_min:
                    print(
                        "No progress exit: promoting remaining uncovered bins to roots."
                    )
                    root_paths, prefix_map = add_remaining_bins(
                        data,
                        offsets,
                        uncovered_key_indices,
                        root_paths,
                        prefix_map,
                        keys_arr,
                        roots_idx,
                        covered_key_indices,
                    )
                    break

    # print(f"Number of root paths: {len(roots_idx)}")
    # print(f"Paths covered: {len(covered_key_indices)}/{len(keys_arr)}")
    print("Condensation complete.")
    print(f"Number of root paths: {len(root_paths)}")
    print(f"Paths covered: {len(prefix_map)}/{len(keys_arr)}")

    # print(f"Length of root_paths: {len(root_paths)}")
    # print(f"Length of prefix_map: {len(prefix_map)}")

    return root_paths, prefix_map


def add_remaining_bins(
    data,
    offsets,
    uncovered_key_indices,
    root_paths,
    prefix_map,
    keys_arr,
    roots_idx,
    covered_key_indices,
):
    for i, i_uncov in enumerate(uncovered_key_indices):
        leftover_path = get_path_by_index(data, offsets, i_uncov)
        if leftover_path.size == 0:
            leftover_goal = []
        else:
            leftover_goal = leftover_path[-1].tolist()
        curr_root_idx = len(root_paths)
        root_paths[curr_root_idx] = leftover_path
        tuple_idx = tuple(tuple(row) for row in keys_arr[i_uncov])
        # prefix_map[tuple_idx] = f"{curr_root_idx}"
        prefix_map[tuple_idx] = (f"{curr_root_idx}", leftover_goal)

    print(
        f"Adding {len(uncovered_key_indices)} unreachable bins to root paths"
    )
    newly = set(uncovered_key_indices)
    covered_key_indices |= newly
    roots_idx.extend(newly)
    uncovered_key_indices -= newly

    return root_paths, prefix_map
