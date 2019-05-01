import os

from fenics import *
from fenics_adjoint import *
import torch_fenics
import torch

import utils


class ReactionModel(torch_fenics.FEniCSModule):
    def __init__(self):
        super(ReactionModel, self).__init__()

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

    def solve(self, c_prev, w, u, f):
        c1_prev, c2_prev, c3_prev = split(c_prev)
        u1, u2, u3 = split(u)
        f1, f2, f3 = split(f)

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
        return (Function(self.V),     # Previous concentrations
                Function(self.W),     # Velocity field
                Constant((0, 0, 0)),  # Input signals
                Function(self.V),     # Estimated reaction dynamics
                )
                


class WhiteBox(torch.nn.Module):
    def __init__(self):
        super(WhiteBox, self).__init__()
        self.reaction_model = ReactionModel()
        self.threshold = torch.nn.ReLU()

    def forward(self, c_prev, w, u, f):
        # Run the FEniCS model
        c_next = self.reaction_model(c_prev, w, u, f)

        # Remove numerical artifacts
        return self.threshold(c_next)


