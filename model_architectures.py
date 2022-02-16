from torch import flatten
import torch.nn as nn
import torch.optim as optim


class nvidia_arch(nn.Module):
    """
    1- nvidia doesn't mention any activation function in their paper so i took the liberty to use ReLU.
    2- nvidia uses 200x66 resolution image but to make things just a little faster and more memory
    efficient i will be using images of size 185x60
    3- hyperparameters are also chosen by me because they are not mentioned
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5,5), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5,5), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5,5), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3))
        )

        # fully connected dense layers
        self.linear = nn.Sequential(
            # input to the fully connected layer will be a (17x1) image with 64 channels
            nn.Linear(in_features=17 * 1 * 64, out_features=100),
            nn.Linear(in_features=100, out_features=50),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = flatten(x)
        x = self.linear(x)
        return x

class nvidiaNeuralNet:
    def __init__(self):
        self.architecture = nvidia_arch()
        self.optimizer = optim.Adam(self.architecture.parameters(), lr=0.001)
        self.epochs = 10



    def train(self, trainset):
        for epoch in range(self.epochs):
            for data in trainset:

