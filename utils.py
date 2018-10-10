import os
from itertools import product

import torch
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
RESULT_DIR = os.path.join(ROOT, 'result')


def progress_bar(progress=None, size=25, ndigits=2):
    if progress < 0 or progress > 1:
        raise ValueError('Progress has to be in [0, 1]')
    return '['+ int(size * progress) * '#' + (size - int(size * progress)) * \
           ' ' + '] {}%'.format(round(100*progress, ndigits))


def load_torch_data(datafile):
    """Load data as torch tensors"""
    data = np.load(datafile)
    c0 = torch.tensor(data['c0']).double()
    c = torch.tensor(data['c']).double()
    u = torch.tensor(data['u']).double()
    y = torch.tensor(data['y']).double()
    return c0, u, c, y


def load_numpy_data(datafile):
    """Load data as numpy arrays"""
    data = np.load(datafile)
    c0 = data['c0']
    c = data['c']
    u = data['u']
    y = data['y']
    return c0, u, c, y


def observability_idx(coordinates, min_x, min_y, max_x, max_y, nx, ny, threshold=None):
    """Compute indices of coordinates corresponding to sensors"""
    eps = 1e-3

    x_step = (max_x - min_x) / (nx + 1)
    y_step = (max_y - min_y) / (ny + 1)

    sensors_x = np.arange(min_x, max_x - x_step - eps, x_step) + x_step
    sensors_y = np.arange(min_y, max_y - y_step - eps, y_step) + y_step

    sensors = product(sensors_x, sensors_y)

    obs_idx = []
    for coord in sensors:
        # Compute distance to nearest coordinate
        diff = np.linalg.norm(coordinates - coord, axis=1)
        idx = np.argmin(diff)
        if threshold is not None:
            # Do not place sensor if coordinate is not near enough
            if diff[idx] > threshold:
                continue
        obs_idx.append(idx)

    return np.array(obs_idx)
