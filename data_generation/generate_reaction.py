import os

import torch_fenics
from fenics import *
from fenics_adjoint import *
import torch
import numpy as np
import tqdm

import utils


class Reaction(torch_fenics.FEniCSModule):
    def __init__(self):
        super().__init__()

        # Define time step
        self.dt = 0.2

        # Define diffusion constant
        self.diffusion_const = 0.02

        # Load mesh
        self.mesh = Mesh(os.path.join(utils.DATA_DIR, 'mesh.xml.gz'))

        # Create function spaces
        P1 = FiniteElement('P', triangle, 1)
        self.V = FunctionSpace(self.mesh, MixedElement([P1, P1, P1]))
        self.W = FunctionSpace(self.mesh, VectorElement('P', triangle, 1))

        # Create trial and test functions
        self.c1, self.c2, self.c3 = TrialFunctions(self.V)
        self.v1, self.v2, self.v3 = TestFunctions(self.V)

        # Create bilinear terms
        a1 = (self.c1 / self.dt) * self.v1 * dx + self.diffusion(self.c1, self.v1)
        a2 = (self.c2 / self.dt) * self.v2 * dx + self.diffusion(self.c2, self.v2)
        a3 = (self.c3 / self.dt) * self.v3 * dx + self.diffusion(self.c3, self.v3)
        self.a = a1 + a2 + a3

        # Create input domains
        self.chi1 = Expression('pow(x[0]-0.2,2)+pow(x[1]-0.6,2)<0.1*0.1 ? 1 : 0', degree=1)
        self.chi2 = Expression('pow(x[0]-0.2,2)+pow(x[1]-0.9,2)<0.1*0.1 ? 1 : 0', degree=1)
        self.chi3 = Constant(0)

    def advection(self, w, u, v):
        return inner(w, grad(u)) * v * dx

    def diffusion(self, u, v):
        return self.diffusion_const * inner(grad(u), grad(v)) * dx

    def solve(self, c_prev, w, u):
        c1_prev, c2_prev, c3_prev = split(c_prev)
        u1, u2, u3 = split(u)

        # Compute reaction rate
        r = 2 * c1_prev ** 2 * c2_prev

        # Compute reaction dynamics
        f1 = -r
        f2 = -2 * r
        f3 = 8 * r

        # Compute advection terms
        a1 = self.advection(w, self.c1, self.v1)
        a2 = self.advection(w, self.c2, self.v2)
        a3 = self.advection(w, self.c3, self.v3)
        a = self.a + a1 + a2 + a3

        # Compute input terms
        L1 = u1 * self.chi1 * self.v1 * dx
        L2 = u2 * self.chi2 * self.v2 * dx
        L3 = u3 * self.chi3 * self.v3 * dx

        # Compute reaction terms
        L1 += f1 * self.v1 * dx
        L2 += f2 * self.v2 * dx
        L3 += f3 * self.v3 * dx

        # Compute time difference terms
        L1 += (c1_prev / self.dt) * self.v1 * dx
        L2 += (c2_prev / self.dt) * self.v2 * dx
        L3 += (c3_prev / self.dt) * self.v3 * dx

        # Assemble linear term
        L = L1 + L2 + L3

        # Solve reacition-advection-diffusion equation
        c_next = Function(self.V)
        solve(a == L, c_next)

        return c_next

    def input_templates(self):
        return (
                Function(self.V),     # Previous concentrations
                Function(self.W),     # Velocity field
                Constant((0, 0, 0))   # Input signals
                )


class Observe(torch.nn.Module):
    def __init__(self, nx, ny, noise_std):
        super().__init__()

        self.noise_std = noise_std

        coordinates = np.load(os.path.join(utils.DATA_DIR, 'coordinates.npy'))

        min_x, min_y = np.min(coordinates, axis=0)
        max_x, max_y = np.max(coordinates, axis=0)

        self.obs_idx = utils.observability_idx(coordinates, min_x, min_y, max_x, max_y, nx, ny, 0.1)

    def forward(self, white_box_output):
        obs = white_box_output[:, self.obs_idx]
        return np.maximum(obs + self.noise_std * torch.randn(obs.shape, dtype=torch.float64), 0)


def random_signal(min_t, max_t, min_val, max_val, n):
    sig = []
    i = 0
    while i < n:
        t = np.random.randint(min_t, max_t)
        c = (max_val - min_val) * np.random.rand() + min_val
        j = 0
        while i < n and j < t:
            sig.append(c)
            i += 1
            j += 1
    return np.array(sig)


def generate_reaction():
    # Create reaction model
    reaction = Reaction()

    # Create observation model
    observe = Observe(20, 5, 0)

    # Load velocity field
    w = np.load(os.path.join(utils.DATA_DIR, 'flow.npy'))

    # Compute input signals
    u1 = random_signal(5, 10, 5, 10, 50)
    u2 = random_signal(5, 10, 5, 10, 50)
    u3 = np.zeros(u1.shape)
    u = np.vstack((u1, u2, u3)).transpose()

    # Create initial condition
    c0 = reaction.numpy_input_templates()[0]
    c_prev = np.array([c0])

    # Simulate
    c = []
    y = []
    for i in tqdm.tqdm(range(50), desc='Simulating reaction'):
        c_prev = reaction(c_prev, w[i:i+1], u[i:i+1])
        c_prev = np.maximum(c_prev, 0)
        y_ = observe(c_prev)

        c.append(c_prev[0].numpy())
        y.append(y_[0].numpy())

    return c0, u, c, y
