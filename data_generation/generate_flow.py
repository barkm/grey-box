import os

import torch_fenics
from fenics import *
from fenics_adjoint import *
import numpy as np
import tqdm

import utils


class Flow(torch_fenics.FEniCSModule):
    def __init__(self):
        super().__init__()

        # Load mesh
        self.mesh = Mesh(os.path.join(utils.DATA_DIR, 'mesh.xml.gz'))

        # Define function spaces
        WH = VectorElement('P', self.mesh.ufl_cell(), 1)
        QH = FiniteElement('P', self.mesh.ufl_cell(), 1)
        VH = WH * QH

        V = FunctionSpace(self.mesh, VH)
        W, Q = V.split()
        self.W_collapsed = V.sub(0).collapse()

        # Create trial and test functions
        self.v, self.q = TestFunctions(V)
        self.u = Function(V)
        self.w, self.p = split(self.u)

        # Define boundary markers
        right = 'near(x[0], 5)'
        left = 'near(x[0], 0)'
        walls = 'near(x[1], 0) || near(x[1], 1.5)'
        cylinder = 'on_boundary && x[0] > 0.1 && x[0] < 2 && x[1] > 0.1 && x[1] < 1.4'

        # Define inflow profile
        uin = Expression(('4*(x[1]*(y_max-x[1]))/(y_max*y_max)', '0.'), degree=2, y_max=1.5)

        # Define boundary conditions
        bc_walls = DirichletBC(W, Constant((0, 0)), walls)
        bc_cylinder = DirichletBC(W, Constant((0, 0)), cylinder)
        bc_right = DirichletBC(Q, Constant(0), right)
        bc_left = DirichletBC(W, uin, left)

        self.bc = [bc_walls, bc_cylinder, bc_right, bc_left]

        # Define step size
        self.dt = 0.2

        # Define viscosity
        self.nu = 0.001

        # Define stabilization factor
        self.d = 0.05 * CellDiameter(self.mesh)


    def solve(self, w_prev):
        # Use Crank-Nicholson method
        wbar = 0.5 * (self.w + w_prev)

        # Navier-Stokes residual
        r_navier_stokes = (inner((self.w - w_prev) / self.dt + grad(self.p) + grad(wbar) * wbar, self.v) +
                           self.nu * inner(grad(wbar), grad(self.v))) * dx

        # Incompressibility residual
        r_div = div(wbar) * self.q * dx

        # Stabilization residual
        r_stab = self.d * (inner(grad(self.p) + grad(wbar) * wbar, grad(self.q) + grad(wbar) * self.v)
                           + inner(div(wbar), div(self.v))) * dx

        # Solve for the velocity and pressure
        solve(r_navier_stokes + r_div + r_stab == 0, self.u, self.bc)

        # Workaround in order to convert the solution to a FEniCS Function
        w_next = project(self.w, self.W_collapsed)
        return w_next

    def input_templates(self):
        # Input template of the velocity at the previous time step
        return Function(self.W_collapsed)


def generate_flow():
    # Create flow model
    flow = Flow()

    # Create initial condition
    w_prev = flow.numpy_input_templates()[0]
    w_prev = np.array([w_prev])

    # Simulate
    w = []
    for i in tqdm.tqdm(range(100), desc='Simulating flow'):
        w_prev = flow(w_prev)
        w.append(w_prev[0].numpy())

    # Save data_generation
    np.save(os.path.join(utils.DATA_DIR, 'flow.npy'), w[50:])



