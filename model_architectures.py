import torch
import torch.nn as nn

# slightly modified nvidia architecture
class nvidia_arch(nn.Module):
    """
    1- nvidia doesn't mention any activation function in their paper so i took the liberty to use ReLU.
    2- nvidia uses 200x66 resolution image but to make things just a little faster and more memory.
    efficient we will be using images of size 120x60 but this might change in the future.
    3- hyper parameters are also chosen by me because they are not mentioned.
    4- network will also do additional convolutions to understand information from minimap and speedometer.
    """

    def __str__(self):
        return "this is nvidia architecture"

    def __init__(self):
        super().__init__()
        # conv1 will look at the road, conv2 will look at the minimap and conv3 will look at the speedometer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(5,5), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5,5), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5,5), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1)),
            nn.Flatten()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=18, kernel_size=(5, 5), stride=(2, 2)),
            nn.Flatten()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5), stride=(2, 2)),
            nn.Flatten()
        )

        # fully connected dense layers
        self.linear = nn.Sequential(
            # input to the fully connected layer will be a (17x1) image with 64 channels
            nn.Linear(in_features=1454, out_features=100, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=6, bias=False)
        )

    # x1 is road x2 is minimap and x3 is the speedometer
    def forward(self, x1, x2, x3):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x = torch.concat((x1,x2,x3), dim=1)
        x = self.linear(x)
        return x

