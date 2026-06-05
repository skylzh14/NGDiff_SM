import torch
import torch.nn as nn
import torch.nn.functional as F
from cross_attention import CrossAttention
from torchsummary import summary


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for Channel Attention"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # 全局平均池化后调整维度
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y


class PyramidConvWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 9], reduction=16):
        super(PyramidConvWithAttention, self).__init__()
        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=ks // 2)
            for ks in kernel_sizes
        ])
        # 批归一化
        self.bn = nn.BatchNorm2d(out_channels * len(kernel_sizes))
        # 注意力机制
        self.se = SEBlock(out_channels * len(kernel_sizes), reduction)
        # 残差连接的1x1卷积匹配通道数
        self.res_conv = nn.Conv2d(in_channels, out_channels * len(kernel_sizes), kernel_size=1)

    def forward(self, x):
        # 多尺度卷积特征提取
        features = [conv(x) for conv in self.convs]
        combined = torch.cat(features, dim=1)  # 拼接不同尺度的特征
        combined = self.bn(combined)

        # 加入注意力机制
        combined_1 = self.se(combined)

        # 残差连接
        residual = self.res_conv(x)
        combined_1 += residual
        combined_1 += combined

        return F.relu(combined_1)

class Res2(nn.Module):

    def __init__(self, in_channels, inter_channels, kernel_size, padding=0):
        super(Res2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        return X

class Res(nn.Module):
    def __init__(self, in_channels, kernel_size, padding, groups_s):
        super(Res, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s//2)
        self.bn1_1 = nn.BatchNorm2d(in_channels)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s//2)
        self.bn1_2 = nn.BatchNorm2d(in_channels)

        self.conv2_1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s)
        self.bn2_1 = nn.BatchNorm2d(in_channels)
        self.conv2_2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s)
        self.bn2_2 = nn.BatchNorm2d(in_channels)

        self.conv3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s*2)
        self.bn3_1 = nn.BatchNorm2d(in_channels)
        self.conv3_2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s*2)
        self.bn3_2 = nn.BatchNorm2d(in_channels)

        self.res2 = Res2(in_channels, 32, kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        Y1 = F.relu(self.bn1_1(self.conv1_1(X)))
        Y1 = self.bn1_2(self.conv1_2(Y1))

        Y2 = F.relu(self.bn2_1(self.conv2_1(X)))
        Y2 = self.bn2_2(self.conv2_2(Y2))

        Y3 = F.relu(self.bn3_1(self.conv3_1(X)))
        Y3 = self.bn3_2(self.conv3_2(Y3))

        Z = self.res2(X)
        return F.relu(X + Y1 + Y2 + Y3 + Z)

class extract(nn.Module):
    def __init__(self,bands):
        super(extract, self).__init__()
        self.bands = bands
        self.CNN_1 = nn.Sequential(
            nn.BatchNorm2d(self.bands),
            nn.Conv2d(self.bands, 128, kernel_size=(1, 1)),
            nn.LeakyReLU(),

            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=5 // 2, groups=64),
            nn.LeakyReLU()
        )
    def forward(self,x):
        x = self.CNN_1(x)
        return x

class group_conv(nn.Module):
    def __init__(self,bands, groups, groups_width):
        super(group_conv, self).__init__()
        # self.cnn_1 = extract(bands)
        # self.pyramid = PyramidConvWithAttention(in_channels=64, out_channels=32, kernel_sizes=[3, 5, 7])
        self.conv_1 = nn.Conv2d(64, groups * groups_width, (1, 1), groups=groups)
        self.bn_1 = nn.BatchNorm2d(groups * groups_width)
        self.res0 = Res(groups * groups_width, (1, 1), (0, 0), groups_s=groups)
        self.conv_1x1 = nn.Conv2d(32*4, 32, kernel_size=1)
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        # x = self.cnn_1(x)
        # x_pyramid = self.pyramid(x)
        x_group = F.relu(self.bn_1(self.conv_1(x)))
        x_group_res = self.res0(x_group)
        x = F.relu(x_group + x_group_res)
        # x = torch.concat((x_group, x_pyramid), dim=1)
        # x = F.relu(self.conv_1x1(x))
        out = F.adaptive_avg_pool2d(x, 1).squeeze()
        out = self.fc(out)
        out = F.softmax(out, dim=-1)
        return x, out#x:feature,32  out:softmax


class Re_channel(nn.Module):
    def __init__(self,bands):
        super(Re_channel, self).__init__()
        self.bands = bands
        self.CNN_1 = nn.Sequential(
            nn.BatchNorm2d(self.bands),
            nn.Conv2d(self.bands, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(),
        )
    def forward(self,x):
        x = self.CNN_1(x)
        return x

class gc_net(nn.Module):
    def __init__(self, ori_bands, bands, groups, groups_width, num_class):
        super(gc_net, self).__init__()
        self.num_class = num_class
        self.re_cha1 = Re_channel(ori_bands)
        self.re_cha2 = Re_channel(bands)
        self.re_cha3 = nn.Sequential(
            # nn.Conv2d(128, 32, kernel_size=(1, 1)),
            nn.Conv2d(64, 32, kernel_size=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.gc = group_conv(256, groups, groups_width)
        self.gc1 = group_conv(128, groups, groups_width)
        self.gc2 = group_conv(128, groups, groups_width)
        self.gc_all = group_conv(309, groups, groups_width)
        self.fc = nn.Linear(32, self.num_class)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pyramid = PyramidConvWithAttention(in_channels=64, out_channels=32, kernel_sizes=[3, 5, 7])
        self.pyramid1 = PyramidConvWithAttention(in_channels=64, out_channels=32, kernel_sizes=[3, 5, 7])
        self.fc_dif = nn.Linear(64, 32*3)
        self.fc_ori = nn.Linear(64, 32 * 3)
        self.dropout1 = nn.Dropout(0.3)
        self.fc_dif_embed = nn.Linear(32 * 3, 32)
        self.fc_ori_embed = nn.Linear(32 * 3, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Sequential(
            nn.Linear(32, 32 // 3, False),
            nn.ReLU(),
            nn.Linear(32 // 3, 32, False),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32, 32 // 3, False),
            nn.ReLU(),
            nn.Linear(32 // 3, 32, False),
            nn.Sigmoid()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32*2, 32*2 // 3, False),
            nn.ReLU(),
            nn.Linear(32*2 // 3, 32*2, False),
            nn.Sigmoid()
        )
        self.cross_atten = CrossAttention(num_classes=num_class)#有默认值
        self.patch_embed_ori = nn.Conv2d(64, 96, kernel_size=1)  # ori_feature -> embed_dim
        self.patch_embed_dif = nn.Conv2d(64, 96, kernel_size=1)  # dif_feature -> embed_dim
    def forward(self, x, mask):
        #直接输入gc网络
        # feature, out = self.gc_all(x)
        #分别获取两个输入
        ori = x[:, :9, :, :]
        dif = x[:, 9:, :, :]
        #调整两个输入为同一纬度
        ori = self.re_cha1(ori)
        dif = self.re_cha2(dif)

        #单跑ori
        # feature, out = self.gc1(ori)
        #单跑dif
        # feature, out = self.gc2(dif)
        #处理成同一纬度，再gc
        # ori_fea, ori_out = self.gc1(ori)
        # dif_fea, dif_out = self.gc2(dif)
        # out = torch.concat((ori, dif), dim=1)
        # feature, out = self.gc(out)
        #分别跑再结合
        # ori, out_ori = self.gc1(ori) #ori:原始数据的特征(128,32,13,13),out_ori:(128,3)
        # ori_embed = F.adaptive_avg_pool2d(ori, 1).squeeze()#(128,32)
        #pyramid
        ori_py = self.pyramid1(ori)
        ori_embed = F.adaptive_avg_pool2d(ori, 1).squeeze()  # (128,96)
        ori_py = F.adaptive_avg_pool2d(ori_py, 1).squeeze()  # (128,96)
        ori_embed = self.fc_ori(ori_embed)
        # ori_embed = self.dropout1(ori_embed)
        ori_embed = ori_embed + ori_py
        ori_embed = self.fc_ori_embed(ori_embed)
        ori_embed = self.dropout2(ori_embed)

        #pyramid_用于cross注意力
        # ori_py = self.pyramid1(ori)
        # ori_embed = self.patch_embed_ori(ori)
        # ori_embed = ori_embed + ori_py#网络的最后输出

        # 注意力
        b, c = ori_embed.shape
        # fc = self.fc1(ori_embed)  # 两次卷积，最后输出通道与输入通道一致
        # fc = fc.view(b, c, 1, 1)
        # Y_ori = torch.multiply(ori, fc)
        # ori_embed = F.adaptive_avg_pool2d(Y_ori, 1).squeeze()#(128,32)
        # dif, out_dif = self.gc2(dif)

        # dif = self.re_cha3(dif)#dif:(128,32,13,13)
        # dif_embed = F.adaptive_avg_pool2d(dif, 1).squeeze()  # (128,32)

        # #pyramid
        dif_py = self.pyramid(dif)
        dif_embed = F.adaptive_avg_pool2d(dif, 1).squeeze()  # (128,96)
        dif_py = F.adaptive_avg_pool2d(dif_py, 1).squeeze()  # (128,96)
        dif_embed = self.fc_dif(dif_embed)
        dif_embed = self.dropout1(dif_embed)
        dif_embed = dif_embed + dif_py
        dif_embed = self.fc_dif_embed(dif_embed)
        dif_embed = self.dropout2(dif_embed)

        # pyramid_用于cross注意力
        # dif_py = self.pyramid1(dif)
        # dif_embed = self.patch_embed_dif(dif)
        # dif_embed = dif_embed + dif_py  # 网络的最后输出

        #cross_attention
        # out = self.cross_atten(ori_embed, dif_embed)
        # ori_embed = F.adaptive_avg_pool2d(ori_embed, 1).squeeze()
        # dif_embed = F.adaptive_avg_pool2d(dif_embed, 1).squeeze()
        #注意力
        # fc = self.fc2(dif_embed)  # 两次卷积，最后输出通道与输入通道一致
        # fc = fc.view(b, c, 1, 1)
        # Y_dif = torch.multiply(dif, fc)
        # dif_embed = F.adaptive_avg_pool2d(Y_dif, 1).squeeze()  # (128,32)
        #联合注意力
        # out = torch.concat((ori, dif), dim=1)
        # out = F.adaptive_avg_pool2d(out, 1).squeeze()
        #::::::::::::::::::::::::::::::::::::::::::::::::
        out = ori_embed + dif_embed
        # out = torch.concat((ori_embed, dif_embed), dim=1)
        fc = self.fc2(out)  # 两次卷积，最后输出通道与输入通道一致
        # fc = fc.view(b, 2*c, 1, 1)
        Y_out = torch.multiply(out, fc)

        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # Y_embed = F.adaptive_avg_pool2d(Y_out, 1).squeeze()  # (128,64)
        # ori_embed = Y_embed[:, :32]
        # dif_embed = Y_embed[:, 32:]
        # 根据位置选取不同的特征
        # mask = torch.unsqueeze(mask, -1).expand(-1, 32)
        # out = torch.where(mask, ori_embed, dif_embed)
        # out = torch.where(mask, dif_embed, ori_embed)  #xi'an

        # out = torch.concat((ori, dif), dim=1)
        # out = F.adaptive_avg_pool2d(out, 1).squeeze()
        # out = torch.concat((ori_embed, dif_embed), dim=1)
        # out = ori_embed + dif_embed

        out = self.fc(out)#out
        return out, ori_embed, dif_embed
        #置信度
        # ori_fea, ori_out = self.gc1(ori)
        # dif_fea, dif_out = self.gc2(dif)
        # ori_2, _ = torch.topk(ori_out, 2, dim=1)
        # dif_2, _ = torch.topk(ori_out, 2, dim=1)
        # alpha_ori = ori_2[:, 0] - ori_2[:, 1]
        # alpha_dif = dif_2[:, 0] - dif_2[:, 1]
        # alpha_ori = alpha_ori[:, None, None, None]
        # alpha_dif = alpha_dif[:, None, None, None]
        # out = torch.concat((alpha_ori*ori_fea, alpha_dif*dif_fea), dim=1)
        # out = F.adaptive_avg_pool2d(out, 1).squeeze()
        # out = self.fc(out)
        # out = F.softmax(out,dim=-1)
        # return ori_out, dif_out, out
        # return out


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = group_conv(bands=9, groups=4, groups_width=8).to(device)
# summary(model, input_size=(9, 13, 13))