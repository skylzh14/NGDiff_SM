import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, input_channels=96, embed_dim=256, num_heads=8, num_classes=10):
        super(CrossAttention, self).__init__()

        # Patch embedding: 将输入的特征图转换为固定维度的特征
        self.patch_embed_ori = nn.Conv2d(input_channels, embed_dim, kernel_size=1)  # ori_feature -> embed_dim
        self.patch_embed_dif = nn.Conv2d(input_channels, embed_dim, kernel_size=1)  # dif_feature -> embed_dim

        # 线性层用于计算 Q, K, V
        self.query_ori = nn.Linear(embed_dim, embed_dim)
        self.key_ori = nn.Linear(embed_dim, embed_dim)
        self.value_ori = nn.Linear(embed_dim, embed_dim)

        self.query_dif = nn.Linear(embed_dim, embed_dim)
        self.key_dif = nn.Linear(embed_dim, embed_dim)
        self.value_dif = nn.Linear(embed_dim, embed_dim)

        # 输出卷积层，将处理后的特征图恢复为原始输入通道数
        self.output_conv = nn.Conv2d(embed_dim*2, input_channels, kernel_size=1)

        # 分类层: 全局平均池化 + 全连接层
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling, output size: [B, C, 1, 1]
        self.fc = nn.Linear(input_channels, num_classes)  # 最终的全连接层用于分类

    def forward(self, ori_feature, dif_feature):
        batch_size = ori_feature.size(0)

        # Patch embedding: 将特征图展平到 (B, embed_dim, H, W)
        ori_embed = self.patch_embed_ori(ori_feature)  # (B, embed_dim, 13, 13)
        dif_embed = self.patch_embed_dif(dif_feature)  # (B, embed_dim, 13, 13)

        # 展平空间维度到 (B, 169, embed_dim)
        ori_embed = ori_embed.flatten(2).transpose(1, 2)  # (B, 169, embed_dim)
        dif_embed = dif_embed.flatten(2).transpose(1, 2)  # (B, 169, embed_dim)

        # 计算 Q, K, V
        Q_ori = self.query_ori(ori_embed)  # (B, 169, embed_dim)
        K_ori = self.key_ori(ori_embed)  # (B, 169, embed_dim)
        V_ori = self.value_ori(ori_embed)  # (B, 169, embed_dim)

        Q_dif = self.query_dif(dif_embed)  # (B, 169, embed_dim)
        K_dif = self.key_dif(dif_embed)  # (B, 169, embed_dim)
        V_dif = self.value_dif(dif_embed)  # (B, 169, embed_dim)

        # 跨模态注意力计算
        # 计算 ori 和 dif 之间的注意力权重
        attention_scores = torch.matmul(Q_ori, K_dif.transpose(1, 2)) / (Q_ori.size(-1) ** 0.5)  # (B, 169, 169)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, 169, 169)

        # 使用注意力权重加权求和得到输出
        attention_output = torch.matmul(attention_weights, V_dif)  # (B, 169, embed_dim)

        # 同理，计算 dif 和 ori 之间的注意力权重
        attention_scores_dif = torch.matmul(Q_dif, K_ori.transpose(1, 2)) / (Q_dif.size(-1) ** 0.5)  # (B, 169, 169)
        attention_weights_dif = F.softmax(attention_scores_dif, dim=-1)  # (B, 169, 169)

        # 使用注意力权重加权求和得到输出
        attention_output_dif = torch.matmul(attention_weights_dif, V_ori)  # (B, 169, embed_dim)

        # 跨模态融合：拼接 ori 和 dif 的注意力输出
        combined_output = attention_output + attention_output_dif  # (B, 169, embed_dim)
        combined_output = torch.concat((attention_output, attention_output_dif), dim=-1)

        # 恢复空间维度: 将处理后的输出转换回 (B, embed_dim, H, W)
        combined_output = combined_output.transpose(1, 2).reshape(batch_size, -1, 13, 13)  # (B, embed_dim, 13, 13)

        # 使用卷积层映射回原始通道数
        output = self.output_conv(combined_output)  # (B, input_channels, 13, 13)

        # 全局平均池化 (B, input_channels, 1, 1)
        pooled_output = self.global_pool(output)  # (B, input_channels, 1, 1)
        pooled_output = pooled_output.view(batch_size, -1)  # 展平为 (B, input_channels)

        # 分类输出
        logits = self.fc(pooled_output)  # (B, num_classes)

        return logits


# 示例使用
if __name__ == "__main__":
    # 生成随机输入特征
    ori_feature = torch.randn(128, 96, 13, 13)  # ori_feature 的大小 [B, C, H, W]
    dif_feature = torch.randn(128, 96, 13, 13)  # dif_feature 的大小 [B, C, H, W]

    num_classes = 10  # 假设我们有10个类别
    model = CrossAttention(num_classes=num_classes)
    output = model(ori_feature, dif_feature)
    print("Output shape:", output.shape)  # 应该是 [B, num_classes] (128, 10)
