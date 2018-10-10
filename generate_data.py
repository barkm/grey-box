import os

import numpy as np

from data_generation.generate_mesh import generate_mesh
from data_generation.generate_flow import generate_flow
from data_generation.generate_reaction import generate_reaction

from utils import DATA_DIR

if __name__ == '__main__':
    # Generate the mesh
    generate_mesh()

    # Simulate the flow in the channel
    w = generate_flow()

    # Simulate the reaction system to generate training, validation and test data
    c0_train, u_train, c_train, y_train = generate_reaction('Generating training data')
    c0_val, u_val, c_val, y_val = generate_reaction('Generating validation data')
    c0_test, u_test, c_test, y_test = generate_reaction('Generating test data')

    with open(os.path.join(DATA_DIR, 'training.npz'), 'wb') as f:
        np.savez(f, c0=c0_train, u=u_train, c=c_train, y=y_train)

    with open(os.path.join(DATA_DIR, 'validation.npz'), 'wb') as f:
        np.savez(f, c0=c0_val, u=u_val, c=c_val, y=y_val)

    with open(os.path.join(DATA_DIR, 'test.npz'), 'wb') as f:
        np.savez(f, c0=c0_test, u=u_test, c=c_test, y=y_test)

