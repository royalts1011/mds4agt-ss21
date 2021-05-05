import torch
from torch import nn
import torch.nn.functional as f

### Variables ###
kern_sz = (3, 3)
stride = (1, 2)
padding = 0


class SimpleNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=kern_sz, stride=stride, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=kern_sz, stride=stride, padding=padding, bias=True)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=kern_sz, stride=stride, padding=padding, bias=True)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=kern_sz, stride=stride, padding=padding, bias=True)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=kern_sz, stride=stride, padding=padding,bias=True)
        #self.conv6 = nn.Conv1d(32, 64, kernel_size=kern_sz, stride=stride, padding=0, bias=True)
        self.linear = nn.Linear(64 * 17 * 24, 55)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        #x = f.relu(self.conv6(x))
        #print(x.shape)
        x = self.linear(x.view(x.size(0), -1))

        return x


class SimpleNet1D(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(27, 32, kernel_size=3, stride=stride, padding=0, bias=True)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=stride, padding=0, bias=True)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=stride, padding=0, bias=True)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=stride, padding=0, bias=True)
        #self.conv5 = nn.Conv1d(256, 512, kernel_size=3, stride=stride, padding=0,bias=True)
        #self.conv6 = nn.Conv1d(256, 512, kernel_size=kern_sz, stride=stride, padding=0, bias=True)
        self.linear = nn.Linear(256 * 49, 55)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        #x = f.relu(self.conv5(x))
        #x = f.relu(self.conv6(x))
        #print(x.shape)
        x = self.linear(x.view(x.size(0), -1))

        return x