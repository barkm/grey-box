import os

from fenics import *
import mshr

import numpy as np

from utils import DATA_DIR


def generate_mesh():
    # Define domain
    x_min = 0; x_max = 5
    y_min = 0; y_max = 1.5

    xc = 0.7; yc = 0.5 * (y_min + y_max)
    r = 0.2

    circle = mshr.Circle(Point(xc, yc), r)
    domain = mshr.Rectangle(Point(x_min, y_min), Point(x_max, y_max)) - circle

    # Create resolution
    mesh_res = 64

    # Create mesh
    mesh = mshr.generate_mesh(domain, mesh_res)

    # Save mesh
    File(os.path.join(DATA_DIR, 'mesh.xml.gz')) << mesh

    # Save dof coordinates
    V = FunctionSpace(mesh, 'P', 1)
    dof_coordinates = V.tabulate_dof_coordinates()
    np.save(os.path.join(DATA_DIR, 'coordinates.npy'), dof_coordinates)

    # Save dof triangles
    vertex2dof = vertex_to_dof_map(V)
    vertex2dof_vec = np.vectorize(lambda v: vertex2dof[v])
    dof_triangles = vertex2dof_vec(mesh.cells())
    np.save(os.path.join(DATA_DIR, 'triangles.npy'), dof_triangles)
