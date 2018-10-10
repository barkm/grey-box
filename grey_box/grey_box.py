import os

import torch
import numpy as np

from grey_box.white_box import WhiteBox
from grey_box.black_box import BlackBox

from utils import observability_idx, DATA_DIR


class ObservationModel(torch.nn.Module):
    def __init__(self, nx, ny):
        super(ObservationModel, self).__init__()

        # Load mesh coordinates
        coordinates = np.load(os.path.join(DATA_DIR, 'coordinates.npy'))

        min_x, min_y = np.min(coordinates, axis=0)
        max_x, max_y = np.max(coordinates, axis=0)

        # Compute indices of mesh vertices corresponding to sensors
        self.obs_idx = observability_idx(coordinates, min_x, min_y, max_x, max_y, nx, ny, 0.1)

    def forward(self, white_box_output):
        return white_box_output[:, self.obs_idx]


class GreyBox(torch.nn.Module):
    def __init__(self):
        super(GreyBox, self).__init__()
        # Create white-box model
        self.white_box = WhiteBox()

        # Create black-box model
        self.black_box = BlackBox(8)

        # Create observation model with a grid of 20 x 5 sensors
        self.observation_model = ObservationModel(20, 5)

    def forward(self, c0, w, u):
        n_time_steps = w.shape[0]

        c = []
        y = []
        c_prev = c0.unsqueeze(0)
        for i in range(n_time_steps):
            # Estimate reaction
            f_prev = self.black_box(c_prev)

            # Compute concentrations
            c_prev = self.white_box(c_prev, w[i:i+1], u[i:i+1], f_prev)

            # Measure concentration in sensors
            y_ = self.observation_model(c_prev)

            c.append(c_prev[0])
            y.append(y_[0])

        c = torch.stack(c)
        y = torch.stack(y)

        return c, y

