import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class TwoLayerConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(TwoLayerConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class gc_net(nn.Module):
    def __init__(self, ori_bands, bands, groups, groups_width):
        super(gc_net, self).__init__()
        # ori 分支的残差块
        self.residual_block = ResidualBlock(ori_bands, 128, kernel_size=3, padding=1)

        # ori 和 dif 分支的两层卷积
        self.conv_ori = TwoLayerConv(128, 64, kernel_size=3, padding=1)
        self.conv_dif = TwoLayerConv(bands, 64, kernel_size=3, padding=1)

        # 全连接层
        self.fc = nn.Linear(128, 3)

    def forward(self, x, mask):
        # 分别获取两个输入
        ori = x[:, :9, :, :]
        dif = x[:, 9:, :, :]

        # ori 分支经过残差块
        ori = self.residual_block(ori)

        # ori 和 dif 分别经过相同的两层卷积
        ori = self.conv_ori(ori)
        dif = self.conv_dif(dif)

        #根据位置选取不同的特征
        # out = torch.where(mask, ori, dif)
        # 合并 ori 和 dif 输出
        out = torch.concat((ori, dif), dim=1)

        # 全局平均池化和全连接层
        out = F.adaptive_avg_pool2d(out, 1).squeeze()
        out = self.fc(out)
        return out#, ori_embed, dif_embed

# 示例用法：
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = gc_net(ori_bands=9, bands=9, groups=4, groups_width=8).to(device)
# summary(model, input_size=(18, 13, 13))
