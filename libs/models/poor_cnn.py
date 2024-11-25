import torch.nn as nn
import torch
from collections import OrderedDict

class PoorPerformingCNN(nn.Module):
    def __init__(self):
        super().__init__() #changed as well
        ##############################
        ###     CHANGE THIS CODE   ###
        ##############################  
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #4 input channels not 5 cause of number of output chanels in previous layer
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        #what we need is 8x8x8 (calculation below)
        #and also 10 output channels cause there is 10 cifar-10 classes
        self.fc1 = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        #cifar 3x32x32
        #conv1 4x32x32
        #pool 4x16x16
        x = self.pool(self.relu1(self.conv1(x)))
        #conv2 8x16x16
        #pool 8x8x8
        x = self.pool(self.relu2(self.conv2(x)))

        x = x.view(-1, 8 * 8 * 8)
        x = self.fc1(x)
        return x