import os

from grey_box.grey_box import GreyBox

from utils import DATA_DIR, RESULT_DIR, load_torch_data

import torch
import numpy as np


if __name__ == '__main__':
    # Load test data
    c0_test, u_test, c_test, y_test = load_torch_data(os.path.join(DATA_DIR, 'test.npz'))
    w = np.load(os.path.join(DATA_DIR, 'flow.npy'))

    # Restore grey-box model
    grey_box = GreyBox()
    grey_box.black_box.load_state_dict(torch.load(os.path.join(RESULT_DIR, 'best_black_box.pt')))

    # Mean square error loss
    loss = torch.nn.MSELoss()

    # Test model on test data
    c_hat_test, y_hat_test = grey_box(c0_test, w, u_test)
    loss_test = loss(y_hat_test, y_test)

    print('Loss on test data:', loss_test.detach().numpy())

    # Save model prediction on test data
    with open(os.path.join(RESULT_DIR, 'prediction.npz'), 'wb') as f:
        np.savez(f,
                 c0=c0_test.detach().numpy(),
                 u=u_test.detach().numpy(),
                 c=c_hat_test.detach().numpy(),
                 y=y_hat_test.detach().numpy())
