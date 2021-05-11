import torch
from torch import nn
import torch.nn.functional as f

### Variables ###
kern_sz = (3, 3)
stride = (1, 2)
padding = 0

stride1D = 1

class SimpleNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(4),
            nn.ReLU()
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kern_sz, stride=stride, padding=padding,bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.convblock6 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.convblock7 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.linear = nn.Linear(256 * 13 * 5, 55)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        #print(x.shape)
        x = self.linear(x.view(x.size(0), -1))

        return x


class SimpleNet1D(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(27, 32, kernel_size=3, stride=stride1D, padding=0, bias=True)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=stride1D, padding=0, bias=True)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=stride1D, padding=0, bias=True)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=stride1D, padding=0, bias=True)
        #self.conv5 = nn.Conv1d(256, 512, kernel_size=3, stride=stride1D, padding=0,bias=True)
        #self.conv6 = nn.Conv1d(256, 512, kernel_size=kern_sz, stride=stride,1D padding=0, bias=True)
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