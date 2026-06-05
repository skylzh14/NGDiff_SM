import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # 用于打印模型摘要
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Simple3DCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        #顺序cbr
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(4704, 512)  # 修改此处以匹配输入尺寸
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = torch.unsqueeze(x, dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool3d(out, 1).squeeze()  # 输出尺寸：[batch_size, out_channels]
        # out = out.view(out.size(0), -1)  # Flatten the tensor
        # out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# class Simple3DCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(Simple3DCNN, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 5, 5))
#         self.bn1 = nn.BatchNorm3d(8)
#         self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 5, 5))
#         self.bn2 = nn.BatchNorm3d(16)
#         self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))
#         self.bn3 = nn.BatchNorm3d(32)
#         self.conv4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3))
#         self.bn4 = nn.BatchNorm3d(64)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(in_features=64, out_features=256)
#         self.dropout1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(in_features=256, out_features=128)
#         self.dropout2 = nn.Dropout(0.4)
#         self.output = nn.Linear(in_features=128, out_features=16)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.bn1(x)
#         x = F.relu(self.conv2(x))
#         x = self.bn2(x)
#         x = F.relu(self.conv3(x))
#         x = self.bn3(x)
#         x = F.relu(self.conv4(x))
#         x = self.bn4(x)
#         x = self.flatten(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = F.softmax(self.output(x), dim=1)
#         return x

# class Simple3DCNN(nn.Module):
#     def __init__(self, in_channels, out_channels, num_classes=10):
#         super(Simple3DCNN, self).__init__()
#
#         self.layer1 = nn.Sequential(
#
#             nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU()
#         )
#
#         # 第一个卷积层
#         self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#
#         # 第二个卷积层
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#
#         # 残差连接的1x1卷积层，匹配维度
#         self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
#         self.bn_res = nn.BatchNorm3d(out_channels)
#
#         # 全连接层，将特征图映射到3维
#         self.fc = nn.Linear(out_channels, num_classes)
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         # 原始输入
#         residual = self.residual_conv(x)
#         residual = self.bn_res(residual)
#
#         out = self.layer1(residual)
#
#         # 第一个卷积层
#         out = self.conv1(out)
#         out = self.bn1(out)
#         out = F.relu(out)
#
#         # 第二个卷积层
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         # 残差连接
#         out += residual
#         out = F.relu(out)
#
#         # 全局平均池化，减少特征图的维度
#         out = F.adaptive_avg_pool3d(out, 1).squeeze()  # 输出尺寸：[batch_size, out_channels]
#
#         # 全连接层，映射到3维
#         out = self.dropout(out)
#         out = self.fc(out)
#         out = F.softmax(out, dim=1)
#
#         return out



# 检查模型摘要，假设输入数据的尺寸为
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Simple3DCNN(num_classes=10).to(device)#in_channels=1, out_channels=64,
# summary(model, input_size=(1, 9, 13, 13))

# # 训练和测试代码示例
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练函数
# def train(model, train_loader, criterion, optimizer, num_epochs=5):
#     model.train()
#     for epoch in range(num_epochs):
#         for i, (inputs, labels) in enumerate(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             if (i+1) % 10 == 0:
#                 print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
#
# # 假设 train_loader 是 DataLoader 类实例，包含训练数据
# # train_loader = ...
#
# # 训练模型
# # train(model, train_loader, criterion, optimizer, num_epochs=5)
#
# # 验证函数
# def validate(model, test_loader, criterion):
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         print(f'Accuracy of the model on the test data: {100 * correct / total:.2f}%')
#
# # 假设 test_loader 是 DataLoader 类实例，包含测试数据
# # test_loader = ...
#
# # 验证模型
# # validate(model, test_loader, criterion)
