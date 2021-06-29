import torch
from torch import nn
import torch.nn.functional as f

### Variables ###
kern_sz = (1, 40)
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
            nn.Conv2d(11, 16, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        self.convblock4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        self.convblock5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        # self.linear = nn.Linear(256 * 1 * 43, 5) # bei kernel size (1,5)
        self.linear = nn.Linear(256 * 1 * 10, 5) # bei kernel size (1,40)

    def forward(self, x):
        """
        defines the path of data through the net
        :param x (input data):
        :return x (tensor of lenght 5 for the 5 classes:
        """
        # print("Input Shape: ", x.shape)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        # print("Later Shape: ", x.shape)
        # print("Size: ", x.size(0))
        # print("view: ", x.view(x.size(0), -1).shape)
        # print("After view Shape: ", x.shape)

        x = self.linear(x.view(x.size(0), -1))

        return x