import torch
from torch import nn
import torch.nn.functional as f

### Variables ###
kern_sz = (3, 5)
stride = (1, 2)
padding = 0


class SimpleNet(nn.Module):
    """
    Defintion of a CNN with 7 convolutional layers followed by one Fully Connected Layer
    to map the activations from the 7th Conv-Layer to the 55 classes.
    """

    def __init__(self):
        super().__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kern_sz, stride=stride, padding=padding,bias=True),
            nn.ReLU()
        )

        self.convblock6 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        self.convblock7 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        # self.linear = nn.Linear(256 * 13 * 5, 55)
        self.linear = nn.Linear(256 * 13 * 3, 55)

    def forward(self, x):
        """
        defines the path of data through the net
        :param x (input data):
        :return x (tensor of lenght 55 for the 55 classes:
        """
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.linear(x.view(x.size(0), -1))

        return x
