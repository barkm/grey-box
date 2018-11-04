import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm

from utils import DATA_DIR, load_numpy_data, observability_idx


def plot_colorbar(ax, vmin, vmax, label):
    m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array([vmin, vmax])
    m.set_clim(vmin, vmax)
    cbar = plt.colorbar(m, ax=ax)
    cbar.set_label(label)


def get_sensor_positions():
    # Load mesh coordinates
    coordinates = np.load(os.path.join(DATA_DIR, 'coordinates.npy'))

    # Get boundary limits
    min_x, min_y = np.min(coordinates, axis=0)
    max_x, max_y = np.max(coordinates, axis=0)

    nx = 20
    ny = 5

    # Compute indices of mesh vertices corresponding to sensors
    obs_idx = observability_idx(coordinates, min_x, min_y, max_x, max_y, nx, ny, 0.1)

    return coordinates[obs_idx]


def plot_contour(c):
    coordinates = np.load(os.path.join(DATA_DIR, 'coordinates.npy'))
    triangles = np.load(os.path.join(DATA_DIR, 'triangles.npy'))

    sensor_pos = get_sensor_positions()

    triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], triangles)

    w, h = plt.figaspect(3 * 1.5/5)
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(w, h), nrows=3)

    c_min = c.min(axis=(0, 1))
    c_max = c.max(axis=(0, 1))

    c_min -= 0.1 * c_min
    c_max += 0.1 * c_max

    for i in range(c.shape[0]):
        tri1 = ax1.tripcolor(triangulation, c[i, :, 0], cmap=cm.coolwarm, vmin=c_min[0], vmax=c_max[0])
        tri2 = ax2.tripcolor(triangulation, c[i, :, 1], cmap=cm.coolwarm, vmin=c_min[1], vmax=c_max[1])
        tri3 = ax3.tripcolor(triangulation, c[i, :, 2], cmap=cm.coolwarm, vmin=c_min[2], vmax=c_max[2])

        if i == 0:
            plot_colorbar(ax1, c_min[0], c_max[0], 'c_1')
            plot_colorbar(ax2, c_min[1], c_max[1], 'c_2')
            plot_colorbar(ax3, c_min[2], c_max[2], 'c_3')

            ax1.plot(sensor_pos[:, 0], sensor_pos[:, 1], 'ko', markersize=1)
            ax2.plot(sensor_pos[:, 0], sensor_pos[:, 1], 'ko', markersize=1)
            ax3.plot(sensor_pos[:, 0], sensor_pos[:, 1], 'ko', markersize=1)

            fig.tight_layout()

        plt.pause(0.001)

        tri1.remove()
        tri2.remove()
        tri3.remove()


if __name__ == '__main__':

    if len(sys.argv) == 2:
        datafile = sys.argv[1]
        c0, u, c, y = load_numpy_data(datafile)
        plot_contour(c)
    elif len(sys.argv) == 3:
        datafile1 = sys.argv[1]
        datafile2 = sys.argv[2]
        c01, u1, c1, y1 = load_numpy_data(datafile1)
        c02, u2, c2, y2 = load_numpy_data(datafile2)
        plot_contour(np.abs(c1 - c2))
