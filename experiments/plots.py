import numpy as np
import matplotlib.pyplot as plt


def ecdf(x: np.ndarray):
    """Return x_sorted, y for empirical CDF (drops nan/inf)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    x.sort()
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def load_required(npz, name: str) -> np.ndarray:
    if name not in npz.files:
        raise KeyError(f"Missing '{name}' in npz. Found keys: {npz.files}")
    return npz[name]


def as_bool(arr: np.ndarray) -> np.ndarray:
    # handles saved bools or 0/1 floats
    return np.asarray(arr).astype(bool)


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def safe_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def load_bundle(path: str):
    d = np.load(path)

    bundle = {
        "planning_times": load_required(d, "planning_times"),
        "planning_lengths": load_required(d, "planning_lengths"),
        "planning_success": as_bool(load_required(d, "planning_success")),
        "adaptation_times": load_required(d, "adaptation_times"),
        "adaptation_lengths": load_required(d, "adaptation_lengths"),
        "adaptation_success": as_bool(load_required(d, "adaptation_success")),
    }
    return bundle


def main():
    path_grr = "data/benchmark_results_grr.npz"
    path_opt = "data/benchmark_results_opt.npz"

    grr = load_bundle(path_grr)
    opt = load_bundle(path_opt)

    # Planning metrics: take from GRR file (as you requested)
    plan_time_s = grr["planning_times"][grr["planning_success"]]
    plan_len_s = grr["planning_lengths"][grr["planning_success"]]
    plan_succ_rate = np.mean(grr["planning_success"])

    # Adaptation metrics (success-only) for each method
    grr_time_s = grr["adaptation_times"][grr["adaptation_success"]]
    grr_len_s = grr["adaptation_lengths"][grr["adaptation_success"]]
    grr_succ_rate = np.mean(grr["adaptation_success"])

    opt_time_s = opt["adaptation_times"][opt["adaptation_success"]]
    opt_len_s = opt["adaptation_lengths"][opt["adaptation_success"]]
    opt_succ_rate = np.mean(opt["adaptation_success"])

    # ---- Terminal report ----
    print("=== Success rates ===")
    print(f"Planning success rate: {plan_succ_rate:.3f} ({grr['planning_success'].sum()}/{len(grr['planning_success'])})")
    print(f"GRR success rate:      {grr_succ_rate:.3f} ({grr['adaptation_success'].sum()}/{len(grr['adaptation_success'])})")
    print(f"TrajOpt success rate:  {opt_succ_rate:.3f} ({opt['adaptation_success'].sum()}/{len(opt['adaptation_success'])})")

    print("\n=== Times (success-only) ===")
    print(f"Planning mean time:     {safe_mean(plan_time_s):.6f}")
    print(f"Planning median time:   {safe_median(plan_time_s):.6f}")
    print(f"GRR mean time:          {safe_mean(grr_time_s):.6f}")
    print(f"GRR median time:        {safe_median(grr_time_s):.6f}")
    print(f"TrajOpt mean time:      {safe_mean(opt_time_s):.6f}")
    print(f"TrajOpt median time:    {safe_median(opt_time_s):.6f}")

    print("\n=== Lengths (success-only) ===")
    print(f"Planning mean length:   {safe_mean(plan_len_s):.6f}")
    print(f"Planning median length: {safe_median(plan_len_s):.6f}")
    print(f"GRR mean length:        {safe_mean(grr_len_s):.6f}")
    print(f"GRR median length:      {safe_median(grr_len_s):.6f}")
    print(f"TrajOpt mean length:    {safe_mean(opt_len_s):.6f}")
    print(f"TrajOpt median length:  {safe_median(opt_len_s):.6f}")

    # ---- Plot ECDFs: Times ----
    x_plan, y_plan = ecdf(plan_time_s)
    x_grr, y_grr = ecdf(grr_time_s)
    x_opt, y_opt = ecdf(opt_time_s)

    plt.figure()
    if x_plan.size:
        plt.plot(x_plan, y_plan, label="RRTConnect Planning Time")
    if x_grr.size:
        plt.plot(x_grr, y_grr, label="GRR Adaptation Time")
    if x_opt.size:
        plt.plot(x_opt, y_opt, label="TrajOpt Adaptation Time")

    plt.xlabel("Time (s)")
    plt.ylabel("CDF")
    plt.xscale("log")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.title("ECDF: Solve Times")
    # Optional (often helpful for heavy tails):
    # plt.xscale("log")

    # ---- Plot ECDFs: Lengths ----
    x_planL, y_planL = ecdf(plan_len_s)
    x_grrL, y_grrL = ecdf(grr_len_s)
    x_optL, y_optL = ecdf(opt_len_s)

    plt.figure()
    if x_planL.size:
        plt.plot(x_planL, y_planL, label="RRTConnect Planning Length")
    if x_grrL.size:
        plt.plot(x_grrL, y_grrL, label="GRR Adaptation Length")
    if x_optL.size:
        plt.plot(x_optL, y_optL, label="TrajOpt Adaptation Length")

    plt.xlabel("Path length (joint-space)")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.title("ECDF: Path Lengths")

    plt.show()


if __name__ == "__main__":
    main()