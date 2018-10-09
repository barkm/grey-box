import os

import torch
import numpy as np

from grey_box.grey_box import GreyBox

from utils import DATA_DIR, RESULT_DIR, progress_bar, load_torch_data


def save_losses(save_path, losses):
    with open(save_path, 'a') as f:
        for loss in losses:
            f.write(str(loss) + '\n')


if __name__ == '__main__':
    c0_train, u_train, c_train, y_train = load_torch_data(os.path.join(DATA_DIR, 'training.npz'))
    c0_val, u_val, c_val, y_val = load_torch_data(os.path.join(DATA_DIR, 'validation.npz'))

    w = np.load(os.path.join(DATA_DIR, 'flow.npy'))

    grey_box = GreyBox()

    optimizer = torch.optim.LBFGS(grey_box.black_box.parameters(), lr=0.1, max_iter=1)

    loss = torch.nn.MSELoss()

    os.makedirs(RESULT_DIR, exist_ok=True)

    losses_train = []
    losses_val = []
    min_val_loss = None
    for i in range(1000):
        def closure():
            optimizer.zero_grad()

            c_hat_train, y_hat_train = grey_box(c0_train, w, u_train)
            loss_train = loss(y_hat_train, y_train)

            c_hat_val, y_hat_val = grey_box(c0_val, w, u_val)
            loss_val = loss(y_hat_val, y_val)

            print('\rTraining grey-box', progress_bar((i+1) / 1000),
                  'training loss: {}, validation loss: {}'.format(loss_train, loss_val),
                  ' '*10, end='')

            losses_train.append(loss_train.detach().numpy())
            losses_val.append(loss_val.detach().numpy())

            if (i+1) % 1 == 0:
                global min_val_loss
                if min_val_loss is None or loss_val < min_val_loss:
                    print('Saved!', loss_val)
                    torch.save(grey_box.black_box.state_dict(),
                               os.path.join(RESULT_DIR, 'black_box.pt'))
                    min_val_loss = loss_val

                save_losses(os.path.join(RESULT_DIR, 'losses_training.txt'), losses_train)
                save_losses(os.path.join(RESULT_DIR, 'losses_validation.txt'), losses_val)

                del losses_train[:]
                del losses_val[:]

            loss_train.backward()

            return loss_train

        optimizer.step(closure)

