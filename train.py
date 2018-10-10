import os

import torch
import numpy as np

from grey_box.grey_box import GreyBox

from utils import DATA_DIR, RESULT_DIR, progress_bar, load_torch_data


def save_losses(save_path, losses, overwrite=False):
    write_mode = 'w' if overwrite else 'a'
    with open(save_path, write_mode) as f:
        for loss in losses:
            f.write(str(loss) + '\n')


if __name__ == '__main__':
    # Load data
    c0_train, u_train, c_train, y_train = load_torch_data(os.path.join(DATA_DIR, 'training.npz'))
    c0_val, u_val, c_val, y_val = load_torch_data(os.path.join(DATA_DIR, 'validation.npz'))
    w = np.load(os.path.join(DATA_DIR, 'flow.npy'))

    # Initialise grey-box
    grey_box = GreyBox()

    # Create L-BFGS optimizer
    optimizer = torch.optim.LBFGS(grey_box.black_box.parameters(), lr=0.05, max_iter=1)

    # Use mean square error loss
    loss = torch.nn.MSELoss()

    if os.path.exists(RESULT_DIR):
        # Restore black box parameters
        grey_box.black_box.load_state_dict(torch.load(os.path.join(RESULT_DIR, 'black_box.pt')))
    else:
        os.makedirs(RESULT_DIR)

    # Number of times to run the training loop
    n_iterations = 1000

    losses_train = []
    losses_val = []
    min_val_loss = None
    for i in range(n_iterations):
        # Define training step
        def closure():
            optimizer.zero_grad()

            # Run training data
            c_hat_train, y_hat_train = grey_box(c0_train, w, u_train)
            loss_train = loss(y_hat_train, y_train)

            # Run validation data
            c_hat_val, y_hat_val = grey_box(c0_val, w, u_val)
            loss_val = loss(y_hat_val, y_val)

            print('\rTraining grey-box', progress_bar((i+1) / n_iterations),
                  'training loss: {}, validation loss: {}'.format(loss_train, loss_val),
                  ' '*10, end='')

            # Log losses
            losses_train.append(loss_train.detach().numpy())
            losses_val.append(loss_val.detach().numpy())

            # Save model with best performance on validation data
            global min_val_loss
            if min_val_loss is None or loss_val < min_val_loss:
                torch.save(grey_box.black_box.state_dict(), os.path.join(RESULT_DIR, 'best_black_box.pt'))
                min_val_loss = loss_val

            # Save losses and current black-box parameters to disk
            if (i+1) % 10 == 0:
                torch.save(grey_box.black_box.state_dict(), os.path.join(RESULT_DIR, 'black_box.pt'))

                save_losses(os.path.join(RESULT_DIR, 'losses_training.txt'), losses_train)
                save_losses(os.path.join(RESULT_DIR, 'losses_validation.txt'), losses_val)

                del losses_train[:]
                del losses_val[:]

            # Compute gradients
            loss_train.backward()

            return loss_train

        # Update parameters
        optimizer.step(closure)

    print('')
