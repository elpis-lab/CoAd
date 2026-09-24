"""Create planning time and path-quality boxplots, each as PDF and PNG."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.common import LABELS, method_results, metrics
from experiments.reporting import parser, datasets


def plot(args):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    matplotlib.rcParams.update(
        {"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    if args.font:
        matplotlib.rcParams["font.serif"] = [args.font]
    experiments = datasets(args, "baseline")
    methods = [
        m
        for m in args.methods
        if any(m in method_results(r) for _, _, r in experiments)
    ]
    if not methods:
        raise ValueError("None of the selected methods have results")
    colors = {method: plt.get_cmap("tab10")(i % 10) for i, method in enumerate(methods)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for field, suffix, ylabel in [
        ("times", "time", "Planning time (ms)"),
        ("lengths", "quality", "Joint-space path length"),
    ]:
        fig, ax = plt.subplots(figsize=(max(7, 2.4 * len(experiments)), 4.8))
        centers, labels = [], []
        for i, (robot, env, results) in enumerate(experiments):
            values_by_method = method_results(results)
            center = i * (len(methods) + 2)
            centers.append(center + (len(methods) - 1) / 2)
            labels.append(
                f'{robot.capitalize()}\n{env.replace("allstable", "all stable").capitalize()}'
            )
            for j, method in enumerate(methods):
                if method not in values_by_method:
                    continue
                success, times, lengths = metrics(values_by_method[method])
                values = (times * 1000 if field == "times" else lengths)[success]
                values = values[np.isfinite(values)]
                if field == "times" and not args.linear_time:
                    values = values[values > 0]
                if not len(values):
                    continue
                box = ax.boxplot(
                    [values],
                    positions=[center + j],
                    widths=0.65,
                    patch_artist=True,
                    showfliers=args.show_outliers,
                )
                box["boxes"][0].set_facecolor(colors[method])
                for median in box["medians"]:
                    median.set_color("black")
        ax.set_xticks(centers, labels)
        ax.set_ylabel(ylabel)
        if field == "times" and not args.linear_time:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(
            handles=[Patch(facecolor=colors[m], label=LABELS[m]) for m in methods],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            ncol=min(5, len(methods)),
            frameon=False,
        )
        fig.tight_layout()
        for extension in ("pdf", "png"):
            path = args.output_dir / f"{args.prefix}_{suffix}.{extension}"
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved {path}")
        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--output-dir", type=Path, default=Path("plots"))
    cli.add_argument("--prefix", default="planning")
    cli.add_argument("--dpi", type=int, default=300)
    cli.add_argument(
        "--font", help="Optional installed serif font (e.g. Times New Roman)"
    )
    cli.add_argument("--show", action="store_true")
    cli.add_argument("--show-outliers", action="store_true")
    cli.add_argument("--linear-time", action="store_true")
    plot(cli.parse_args())
