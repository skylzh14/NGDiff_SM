import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, weight_predictor_channels):
        super(PyramidConvolution, self).__init__()

        # 初始化不同尺度的卷积核
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, k, padding=k // 2) for k in kernel_sizes
        ])

        # 初始化权重预测网络
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(in_channels, weight_predictor_channels, 1),  # 1x1卷积用于降维或特征提取
            nn.ReLU(inplace=True),
            nn.Conv2d(weight_predictor_channels, len(kernel_sizes), 1)  # 输出与卷积核数量相同的权重图
        )

    def forward(self, x):
        # 通过不同尺度的卷积核得到特征图
        features = [conv(x) for conv in self.convs]

        # 通过权重预测网络得到权重图
        weights = torch.softmax(self.weight_predictor(x), dim=1)  # 使用softmax确保权重和为1
        weights = weights.view(-1, len(self.convs), 1, 1)  # 调整权重图的形状以匹配特征图

        # 对特征图进行加权求和
        weighted_sum = sum(weights[:, i, :, :] * features[i] for i in range(len(self.convs)))

        return weighted_sum


# 示例使用
if __name__ == "__main__":
    # 假设输入为 (batch_size, in_channels, height, width) 的特征图
    input_tensor = torch.randn(1, 3, 64, 64)  # 例如，batch_size=1, in_channels=3, height=64, width=64

    # 定义金字塔卷积的参数
    in_channels = 3
    out_channels = 16
    kernel_sizes = [3, 5, 7]  # 使用3x3, 5x5, 7x7的卷积核
    weight_predictor_channels = 8  # 权重预测网络的中间通道数

    # 实例化金字塔卷积
    pyconv = PyramidConvolution(in_channels, out_channels, kernel_sizes, weight_predictor_channels)

    # 通过金字塔卷积得到输出
    output_tensor = pyconv(input_tensor)

    # 打印输出张量的形状
    print(f"Output tensor shape: {output_tensor.shape}")