import torch
from torch import nn
import torch.nn.functional as f

### Variables ###
kern_sz = 3
stride = 1
padding = 0

stride1D = 1

class SimpleNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.ciao = nn.Sequential(
            nn.Conv1d(3, 4, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=kern_sz, stride=stride, padding=padding,bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=kern_sz, stride=stride, padding=padding, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.linear = nn.Linear(256 * 13 * 5, 55)

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        x1 = self.ciao(x1)
        x2 = self.ciao(x2)
        x3 = self.ciao(x3)
        x4 = self.ciao(x4)
        x5 = self.ciao(x5)
        x6 = self.ciao(x6)
        x7 = self.ciao(x7)
        x8 = self.ciao(x8)
        x9 = self.ciao(x9)

        x = torch.cat(x1,x2,x3,x4,x5,x6,x7,x8,x9, dim=1)
        #print(x.shape8)
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