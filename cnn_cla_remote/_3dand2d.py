import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN_3dand2d(nn.Module):
    def __init__(self,channel):
        super(CNN_3dand2d, self).__init__()
        self.channel = channel
        self.spa3d = nn.Conv3d(1, 8, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.spe3d = nn.Conv3d(8, 16, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.spa_bn = nn.BatchNorm3d(8)
        self.spe_bn = nn.BatchNorm3d(16)
        self.relu = nn.ReLU()
        self.fla = nn.Flatten(1, 2)
        self.all2d = nn.Conv2d(self.channel * 16, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(364, 3)

    def forward(self, x):
        out = torch.unsqueeze(x, dim=1)

        out = self.spa3d(out)
        out = self.spa_bn(out)
        out = self.relu(out)

        out = self.spe3d(out)
        out = self.spe_bn(out)
        out = self.relu(out)

        out = self.fla(out)
        out = self.all2d(out)
        out = torch.cat((out, x), dim=1)

        out = F.adaptive_avg_pool2d(out, 1).squeeze()
        out = self.fc(out)
        return out

# model = CNN_3dand2d(channel=9).to(device)
# summary(model, input_size=(1, 9, 13, 13))