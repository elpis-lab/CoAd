import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl

# ---- Use Times New Roman everywhere ----
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"  # makes math look like Times


def generate_fake_data(robots, envs, methods, n_problems=1000, seed=42):
    rng = np.random.default_rng(seed)

    base_mu = {
        "RRT-Connect": 1.50,
        "RRT*": 1.05,
        "TrajOpt": 1.20,
        "LOAD-Interpolation": 0.95,
        "LOAD-DMV": 0.90,
        "LOAD-TrajOpt": 0.80,
    }
    base_sigma = {
        "RRT-Connect": 0.25,
        "RRT*": 0.22,
        "TrajOpt": 0.18,
        "LOAD-Interpolation": 0.07,
        "LOAD-DMV": 0.04,
        "LOAD-TrajOpt": 0.05,
    }

    env_offset = {"Table": 0.00, "Cage": 0.10, "Shelf": 0.05}
    robot_offset = {"UR10": 0.00, "Fetch": 0.03}

    data = {}
    for r in robots:
        for e in envs:
            for m in methods:
                mu = base_mu[m] + env_offset[e] + robot_offset[r]
                sigma = base_sigma[m]
                samples = rng.normal(mu, sigma, n_problems)
                samples = np.clip(samples, 0.02, None)
                data[(r, e, m)] = samples
    return data


def plot_path_quality_boxplot(
    data, robots, envs, methods, save_name=None, fig_size=(16, 4)
):
    """Draw the boxplot of path quality"""
    sections = [(r, e) for r in robots for e in envs]
    n_methods = len(methods)

    gap = 2.5
    within = 1.0

    positions, box_data, method_ids = [], [], []
    section_centers, section_labels = [], []

    pos = 1.0
    for r, e in sections:
        start = pos
        for mi, m in enumerate(methods):
            positions.append(pos)
            box_data.append(np.asarray(data[(r, e, m)]))
            method_ids.append(mi)
            pos += within
        end = pos - within
        section_centers.append((start + end) / 2.0)
        section_labels.append(f"{r} - {e}")
        pos += gap

    # Text sizing rule:
    # previous "title" size becomes regular text size
    regular_size = 16
    title_size = int(regular_size * 1.2)  # smallest
    xlabel_ylabel_size = int(regular_size * 1.15)  # larger
    tick_size = regular_size
    legend_size = regular_size

    fig, ax = plt.subplots(figsize=fig_size)

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        zorder=3,
    )

    # Consistent method colors across sections
    colors = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()
        .get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])
    )
    method_colors = [colors[i % len(colors)] for i in range(n_methods)]

    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(method_colors[method_ids[i]])
        box.set_alpha(0.85)
        box.set_edgecolor("black")

    for med in bp["medians"]:
        med.set_linewidth(1.5)
        med.set_color("black")

    # Horizontal y-grid in background
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)

    # Labels/title
    ax.set_ylabel("Path Length (radians)", fontsize=xlabel_ylabel_size)
    # ax.set_xlabel("Robot and Environment", fontsize=xlabel_ylabel_size)
    # ax.set_title(
    #     "Path Cost across Robots and Environments",
    #     fontsize=title_size,
    # )

    ax.set_xticks(section_centers)
    ax.set_xticklabels(section_labels, fontsize=tick_size)
    ax.tick_params(axis="y", labelsize=tick_size)

    ax.set_xlim(min(positions) - 1.0, max(positions) + 1.0)

    # Light separators between sections
    for i in range(1, len(sections)):
        sep_x = (positions[i * n_methods - 1] + positions[i * n_methods]) / 2.0
        ax.axvline(sep_x, linewidth=1.0, color="0.85", zorder=1)

    # # ---- Bottom legend ----
    # handles = [
    #     Patch(
    #         facecolor=method_colors[i],
    #         edgecolor="black",
    #         label=methods[i],
    #         alpha=0.85,
    #     )
    #     for i in range(n_methods)
    # ]

    # fig.legend(
    #     handles=handles,
    #     labels=methods,
    #     loc="lower center",
    #     ncol=3,
    #     frameon=False,
    #     fontsize=legend_size,
    #     title="Methods",
    #     title_fontsize=legend_size,
    #     bbox_to_anchor=(0.5, -0.03),  # small negative y keeps it close
    # )

    # ---- Right-side legend ----
    handles = [
        Patch(
            facecolor=method_colors[i],
            edgecolor="black",
            label=methods[i],
            alpha=0.85,
        )
        for i in range(n_methods)
    ]

    # Place legend to the right of the axes
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # right side
        frameon=False,
        fontsize=legend_size,
        title="Methods",
        title_fontsize=legend_size,
    )

    # # Reserve ONLY a small bottom margin for the legend
    # fig.subplots_adjust(bottom=0.25)  # bottom legend
    fig.subplots_adjust(right=0.78)  # right legend

    if save_name:
        save_path = f"data/plots"
        os.makedirs(save_path, exist_ok=True)
        png = save_path + "/" + save_name + ".png"
        pdf = save_path + "/" + save_name + ".pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, dpi=300, bbox_inches="tight")
        print(f"[Saved] {png}")
        print(f"[Saved] {pdf}")

    plt.show()


def prepare_data(
    robots,
    envs,
    methods,
    metric="lengths",          # "lengths" or "times"
    only_success=False,
    time_in_ms=False,
    method_name_map=None
):
    default_method_name_map = {
        "RRT-Connect": ("rrtc", None),
        "Library": ("library", None),
        "LOAD-Interpolation": ("adaptations", "grr"),
        "LOAD-TrajOpt": ("adaptations", "opt"),
        "LOAD-DMP": ("adaptations", "dmp"),
    }
    if method_name_map is None:
        method_name_map = default_method_name_map

    data_out = {}

    for robot in robots:
        for env in envs:
            data_path = f"data/baseline_results_{robot}_{env}.npz"

            if not os.path.exists(data_path):
                print(f"[Warn] Missing: {data_path} (filling NaNs)")
                for m in methods:
                    data_out[(robot, env, m)] = np.array([np.nan], dtype=float)
                continue

            npz = np.load(data_path, allow_pickle=True)
            results = npz["results"].item()

            for m in methods:
                top_key, adapt_key = method_name_map[m]

                if top_key in ("rrtc", "library"):
                    values = np.asarray(results[top_key][metric], dtype=float)
                    if only_success:
                        success = np.asarray(results[top_key]["success"], dtype=bool)
                        values = values[success]

                elif top_key == "adaptations":
                    values = np.asarray(results["adaptations"][metric][adapt_key], dtype=float)
                    if only_success:
                        success = np.asarray(results["adaptations"]["success"][adapt_key], dtype=bool)
                        values = values[success]
                else:
                    raise ValueError(top_key)

                values = values[np.isfinite(values)]
                if metric == "times" and time_in_ms:
                    values = values * 1000.0

                # ensure non-empty so boxplot doesn't choke
                if values.size == 0:
                    values = np.array([np.nan], dtype=float)

                data_out[(robot, env, m)] = values

    return data_out

if __name__ == "__main__":
    robots = ["UR10", "Fetch"]
    envs = ["Table", "Cage", "Shelf"]
    methods = [
        "RRT-Connect",
        "Library",
        #"TrajOpt",
        "LOAD-Interpolation",
        "LOAD-DMP",
        "LOAD-TrajOpt",
    ]

    robots = ["panda", "fetch"]
    envs = ["table", "cage", "shelf"]

    # TODO replace this with real data
    #data = generate_fake_data(robots, envs, methods, n_problems=1000, seed=42)
    data = prepare_data(robots, envs, methods, only_success=False)

    plot_path_quality_boxplot(data, robots, envs, methods, save_name="quality")
