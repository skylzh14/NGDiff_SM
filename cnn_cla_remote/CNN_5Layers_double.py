import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def weight(name=None):
    initial = torch.rand(size=[1])
    return initial

branch1 = weight('branch1')+1
branch2 = weight('branch2')+1

branch_1=branch1.to(device)
branch_2 = branch2.to(device)

#branch_1=branch1.to(device='cuda')
#branch_2 = branch2.to(device='cuda')


class CNN(nn.Module):  #pytorch
    def __init__(self, BAND, classes):
        super(CNN, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=BAND,
                out_channels=18,
                kernel_size=5,
                padding=2,
                stride=1
            ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
            )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(
              in_channels=18,
              out_channels=36,
              kernel_size=3,
              padding=1,
              stride=1
            ),
              nn.ReLU(),
              #nn.MaxPool2d(kernel_size=2)
            )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(
            in_channels = 36,
            out_channels = 72,
            kernel_size = 3,
            padding = 1,
            stride = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
            )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(
            in_channels=300,
            out_channels =128,
            kernel_size = 3,
            padding = 1,
            stride = 1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
            )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels= 256 + 72,
                out_channels=512,
                kernel_size= 3,
                padding=1,
                stride=1
            ),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=1,
                padding=0,
                stride=1
            ),
            nn.ReLU()
        )
        self.batch_norm = nn.Sequential(
            nn.BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.full_connection = nn.Sequential(
             #nn.ReLU(),
             nn.Dropout(p=0.5),
             #nn.Linear(100, classes) #patch = 4
             nn.Linear(4096, classes)
             #nn.Linear(32, 5) #patch=3
        )
    def forward(self, x):
        x1 = x[:,:9,:,:]
        x2 = x[:,9:,:,:]
        #F = nn.AdaptiveAvgPool2d((1,1))
        #x2 = F(x2)
        x1_1 = self.conv1_1(x1)
        #print(x1_1.shape)
        x2_1 = self.conv2_1(x1_1)
        #print(x2_1.shape)
        x3_1 = self.conv3_1(x2_1)
        #print(x3_1.shape)
        x1_2 = self.conv1_2(x2)
        #print(x1_2.shape)
        x2_2 = self.conv2_2(x1_2)
        #print(x2_2.shape)
        x4 = torch.concat([branch_1*x3_1, branch_2*x2_2],dim=1)
        #print(x4.shape)
        x4 = self.conv4(x4)
        #print(x4.shape)
        x4 = self.conv5(x4)
        #print(x4.shape)
        x5 = self.batch_norm(x4)
        #print(x5.shape)
        x6=x5.contiguous().view(x5.size(0), -1)
        #print(x6.shape)
        #x = x.view((x.size(0), -1))
        return self.full_connection(x6)