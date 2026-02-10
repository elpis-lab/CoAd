import matplotlib.pyplot as plt # type: ignore
import numpy as np

def draw_bbox_wireframe(ax, xs, ys, zs):
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    z0, z1 = zs.min(), zs.max()

    corners = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ])

    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a,b in edges:
        ax.plot([corners[a,0], corners[b,0]],
                [corners[a,1], corners[b,1]],
                [corners[a,2], corners[b,2]],
                color="lightgray", linewidth=2)

class Plotter:
    def __init__(self, keys_arr, indexer):
        self.xs, self.ys, self.yaws = self.bin_centers_xyyaw(keys_arr)
        self.keys_arr = keys_arr
        self.indexer = indexer

    def _get_grid_indices(self, idx_list):
        ixs = []
        iys = []
        iyaws = []

        for i in idx_list:
            ix, iy, iz, iyaw = self.indexer.key_from_box(self.keys_arr[i])
            ixs.append(ix)
            iys.append(iy)
            iyaws.append(iyaw)

        return np.array(ixs), np.array(iys), np.array(iyaws)

    def plot_bin_and_neighbors_3d_shell(self, root_idx, covered_idx):

        all_idx = np.arange(len(self.xs))  # or len(keys_arr)

        xmin, xmax = self.xs[all_idx].min(), self.xs[all_idx].max()
        ymin, ymax = self.ys[all_idx].min(), self.ys[all_idx].max()
        yawmin, yawmax = self.yaws[all_idx].min(), self.yaws[all_idx].max()

        eps = 1e-9
        shell_mask = (
            (self.xs <= xmin + eps) | (self.xs >= xmax - eps) |
            (self.ys <= ymin + eps) | (self.ys >= ymax - eps) |
            (self.yaws <= yawmin + eps) | (self.yaws >= yawmax - eps)
        )

        shell_idx = np.where(shell_mask)[0]



        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        # covered indices (safe int dtype)
        cov = np.array(covered_idx, dtype=np.int64) if len(covered_idx) else np.empty((0,), dtype=np.int64)

        # bounding shell (grid boundary)
        ax.scatter(self.xs[shell_idx], self.ys[shell_idx], self.yaws[shell_idx],
                c="lightgray", s=40, alpha=0.4, depthshade=False, label="Bin bounds")

        # covered bins
        if cov.size:
            ax.scatter(self.xs[cov], self.ys[cov], self.yaws[cov],
                    c="blue", s=60, alpha=0.9, label="Covered")

        # root
        ax.scatter(self.xs[root_idx], self.ys[root_idx], self.yaws[root_idx],
                c="red", s=120, marker="x", label="Root")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("yaw")
        ax.legend()
        ax.view_init(elev=25, azim=35)
        plt.show()


    def bin_centers_xyyaw(self, keys_arr):
        xs = 0.5 * (keys_arr[:, 0, 0] + keys_arr[:, 0, 1])
        ys = 0.5 * (keys_arr[:, 1, 0] + keys_arr[:, 1, 1])
        yaws = 0.5 * (keys_arr[:, 3, 0] + keys_arr[:, 3, 1])
        return xs, ys, yaws

    def plot_bin_and_neighbors_3d(self, root_idx, neighbor_idx, show_all=True):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Optional: plot all bins lightly for context
        if show_all:
            ax.scatter(
                self.xs,
                self.ys,
                self.yaws,
                s=6,
                c="lightgray",
                alpha=0.3,
                label="All bins"
            )

        # Plot neighbors
        nb_x = self.xs[neighbor_idx]
        nb_y = self.ys[neighbor_idx]
        nb_yaw = self.yaws[neighbor_idx]

        ax.scatter(
            nb_x,
            nb_y,
            nb_yaw,
            s=30,
            c="blue",
            alpha=0.8,
            label="Covered neighbors"
        )

        # Plot root bin
        ax.scatter(
            self.xs[root_idx],
            self.ys[root_idx],
            self.yaws[root_idx],
            s=100,
            c="red",
            marker="x",
            linewidths=2,
            label="Root bin"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("yaw")

        ax.set_title(f"3D neighbors of bin {root_idx}")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_bin_and_neighbors(self, root_idx, neighbor_idx, show_all=True):
        plt.figure(figsize=(6, 6))

        # Optional: plot all bins lightly for context
        if show_all:
            plt.scatter(self.xs, self.ys, s=8, c="lightgray", alpha=0.4, label="All bins")

        # Plot neighbors
        nb_x = self.xs[neighbor_idx]
        nb_y = self.ys[neighbor_idx]
        plt.scatter(nb_x, nb_y, s=30, c="blue", alpha=0.8, label="Neighbors")

        # Plot root bin
        plt.scatter(self.xs[root_idx], self.ys[root_idx],
                    s=80, c="red", marker="x", linewidths=2,
                    label="Sampled bin")

        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title(f"Neighbors of bin {root_idx}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_root_and_covered(self, roots_idx, covered_idx, show_all=True):
        plt.figure(figsize=(6, 6))

        # Optional: plot all bins lightly for context
        if show_all:
            plt.scatter(self.xs, self.ys, s=8, c="lightgray", alpha=0.4, label="All bins")

        # Plot neighbors
        nb_x = self.xs[covered_idx]
        nb_y = self.ys[covered_idx]
        plt.scatter(nb_x, nb_y, s=30, c="black", alpha=0.8, label="Covered bins")

        # Plot root bin
        plt.scatter(self.xs[roots_idx], self.ys[roots_idx],
                    s=80, c="red", marker="x", linewidths=2,
                    label="Root bins")

        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title(f"Covered bins")
        plt.grid(True)
        plt.tight_layout()
        plt.show()