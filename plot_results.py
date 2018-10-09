import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import torch

from utils import RESULT_DIR
from grey_box.black_box import BlackBox


def plot_losses():
    losses_train_file = os.path.join(RESULT_DIR, 'losses_training.txt')
    losses_val_file = os.path.join(RESULT_DIR, 'losses_validation.txt')

    if os.path.exists(losses_train_file) and os.path.exists(losses_val_file):
        losses_train = np.loadtxt(losses_train_file)
        losses_val = np.loadtxt(losses_val_file)

        fig, ax = plt.subplots()

        ax.semilogy(losses_train)
        ax.semilogy(losses_val)

        ax.legend(['training loss', 'validation loss'])


def plot_black_box():
    black_box_file = os.path.join(RESULT_DIR, 'black_box.pt')

    if os.path.exists(black_box_file):

        black_box = BlackBox(8)
        black_box.load_state_dict(torch.load(black_box_file))

        c1 = np.linspace(0, 0.05, 20)
        c2 = np.linspace(0, 0.05, 20)

        C1, C2 = np.meshgrid(c1, c2)

        fhat = np.empty(list(C1.shape) + [3])

        for i, c1_ in enumerate(c1):
            for j, c2_ in enumerate(c2):
                inp = torch.tensor([[[c1_, c2_, 0]]]).double()
                fhat[i, j, :] = black_box(inp).detach().numpy()

        q = black_box.state_dict()['neural_network.3.weight'].detach().numpy()
        q_min_idx = np.argmin(np.abs(q))
        delta = q[q_min_idx]

        rhat = fhat[:, :, q_min_idx].transpose() * delta

        r = 2 * C1**2 * C2

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_wireframe(C1, C2, rhat)
        ax.plot_wireframe(C1, C2, r, color='red')

        ax.legend(['R_hat', 'R'])
        ax.set_xlabel('c_1')
        ax.set_ylabel('c_2')

        q_hat = np.abs(q) / np.min(np.abs(q))
        print('alpha_hat =', q_hat[0, 0])
        print('beta_hat =', q_hat[1, 0])
        print('gamma_hat =', q_hat[2, 0])


if __name__ == '__main__':

    plot_losses()
    plot_black_box()

    plt.show()