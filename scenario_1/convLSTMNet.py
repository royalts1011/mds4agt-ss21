import torch
from torch import nn
import torch.nn.functional as f

### Variables ###
kern_sz = (3, 3)
stride = (1, 2)
padding = 0

class convLSTMNET(nn.Module):

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


        self.lstmblock = nn.Sequential(
            nn.LSTM(input_size=4, hidden_size=3, num_layers=1, batch_first=True, dropout=0.0)
        )

        self.linear = nn.Linear(22344, 55)
        # self.linear = nn.Linear(32 * 98, 55)

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        lstm_input = x.view(len(x), -1, 4)
        x, hs = self.lstmblock(lstm_input)
        #print(x.shape)
        x = self.linear(x.reshape(x.size(0), -1))

        return x