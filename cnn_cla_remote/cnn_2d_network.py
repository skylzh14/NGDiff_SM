import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN_2D(nn.Module):
    def __init__(self,channel):
        super(CNN_2D, self).__init__()

        self.channel = channel

        self.CNN_1 = nn.Sequential(
            nn.BatchNorm2d(self.channel),
            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)),
            nn.LeakyReLU(),

            # nn.BatchNorm2d(128),
            # nn.Conv2d(128, 128, kernel_size=(1, 1)),
            # nn.LeakyReLU(),

            # nn.BatchNorm2d(128),
            # nn.Conv2d(128, 128, kernel_size=(1, 1)),
            # nn.LeakyReLU()
        )

        self.CNN_2 = nn.Sequential(
            # nn.BatchNorm2d(128),
            # nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            # nn.LeakyReLU(),
            #
            # nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=5//2, groups=128),
            # nn.LeakyReLU(),

            # #第二层
            # nn.BatchNorm2d(128),
            # nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            # nn.LeakyReLU(),
            #
            # nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=5 // 2, groups=128),
            # nn.LeakyReLU(),

            #第三层
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=5//2, groups=64),
            nn.LeakyReLU()
        )

        self.fc = nn.Linear(64, 3)
    def forward(self, x):
        out = self.CNN_1(x)
        out = self.CNN_2(out)
        out = F.adaptive_avg_pool2d(out, 1).squeeze()
        out = self.fc(out)
        return out