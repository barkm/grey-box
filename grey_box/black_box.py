import torch


class BlackBox(torch.nn.Module):
    def __init__(self, hidden_neurons):
        super(BlackBox, self).__init__()

        # Create mulit-layer perceptron network
        self.neural_network = torch.nn.Sequential(torch.nn.Linear(2, hidden_neurons, bias=True),
                                                  torch.nn.Sigmoid(),
                                                  torch.nn.Linear(hidden_neurons, 1, bias=True),
                                                  torch.nn.Linear(1, 3, bias=False)
                                                  ).double()

    def forward(self, c_prev):
        # Use c_1 and c_2 as input to network
        return self.neural_network(c_prev[:, :, :2])
