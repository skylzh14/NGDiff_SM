import torch
import torch.nn as nn
import utils
import time
import numpy as np
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import matplotlib.pyplot as plt
import spectral as spy
import torch.nn.functional as F
import hdf5storage as hdf5
from utils import device

#混淆矩阵各个数据集的类别
def target_name(data_name):
    if data_name == "512_512":
        return ['water', 'grass', 'building']

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def compution_accuracy(y_pred_test, y_test, data_name):
    # 计算 oa,aa,kappa,EachClassAccuracy,混淆矩阵(这些仅test数据)
    #存数据
    res = {}
    class_num = np.max(y_test) + 1
    classification = classification_report(y_test, y_pred_test,
                                           labels=list(range(class_num)), digits=4, target_names=target_name(data_name), zero_division=1)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    res['classification'] = str(classification)
    res['oa'] = oa * 100
    res['confusion'] = str(confusion)
    res['each_acc'] = str(each_acc * 100)
    res['aa'] = aa * 100
    res['kappa'] = kappa * 100
    return res

def predict(model, test_loader, data_name, ratio=5, keep_probability=False):
    count = 0
    model.eval()
    y_pred_test = []
    y_test = []
    y_probability = []

    with torch.no_grad():  # 禁用梯度计算，节省内存
        for inputs, labels, mask in test_loader:
            # inputs = inputs.to(device)
            # mask = mask.to(device)
            # labels = labels.to(device)  # 将 labels 也移到 GPU

            # 获取模型输出
            outputs, _, _ = model(inputs, mask)
            # outputs = model(inputs, mask)
            # 在GPU上获取每个样本的预测标签
            outputs_arg = torch.argmax(outputs, dim=1)

            if count == 0:
                y_pred_test = outputs_arg
                y_test = labels
                y_probability = outputs
                count = 1
            else:
                y_pred_test = torch.cat((y_pred_test, outputs_arg))  # 在GPU上拼接预测值
                y_test = torch.cat((y_test, labels))  # 在GPU上拼接真实标签
                y_probability = torch.cat((y_probability, outputs))  # 在GPU上拼接预测概率

    # 当你需要将它们转换为 NumPy 格式时再转移到CPU
    y_pred_test = y_pred_test.cpu().numpy()
    y_test = y_test.cpu().numpy()
    if keep_probability == True:
        y_probability = y_probability.cpu().numpy()
        hdf5.savemat(f"./{data_name}/y_probabilityps={ratio}p.mat", {'probability': y_probability})

    return y_pred_test, y_test

    # 保存和输出

# 计算余弦距离的函数

def inter_class_loss_optimized(features, labels):
    """
    类间损失的矩阵化实现，优化效率。
    :param features: Tensor, shape (batch_size, feature_dim)，输入特征
    :param labels: Tensor, shape (batch_size, )，类别标签
    :return: 类间损失值
    """
    unique_labels = labels.unique()  # 获取所有类别
    centers = []

    # 计算每个类别的特征中心
    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        center = class_features.mean(dim=0)
        centers.append(center)

    centers = torch.stack(centers)  # (num_classes, feature_dim)

    # 计算类中心之间的距离矩阵
    dist_matrix = torch.cdist(centers, centers, p=2)  # (num_classes, num_classes)
    dist_matrix = dist_matrix + torch.eye(dist_matrix.size(0)).to(dist_matrix.device) * 1e6  # 避免自身对角元素

    # 类间损失：对非对角元素求倒数
    loss = (1.0 / dist_matrix).sum() / (dist_matrix.size(0) * (dist_matrix.size(0) - 1))
    return loss

# 定义分支独特性约束损失
class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()

    def forward(self, f1, f2):
        """
        计算正交性约束损失
        :param f1: 分支1的特征张量，形状 (batch_size, feature_dim)
        :param f2: 分支2的特征张量，形状 (batch_size, feature_dim)
        :return: 正交性约束损失
        """
        # 归一化特征向量
        f1_norm = F.normalize(f1, p=2, dim=1)
        f2_norm = F.normalize(f2, p=2, dim=1)

        # 计算特征之间的点积（即余弦相似度）
        similarity_matrix = torch.mm(f1_norm, f2_norm.t())  # shape: (batch_size, batch_size)

        # 只计算非对角线部分（避免自相似性）
        batch_size = f1.size(0)
        identity_matrix = torch.eye(batch_size, device=f1.device)
        orthogonality_loss = torch.mean((similarity_matrix - identity_matrix) ** 2)

        return orthogonality_loss

def mutual_information_loss(features1, features2, bins=256):
    """
    计算两个特征张量之间的互信息。
    :param features1: 第一个分支的特征 (batch_size, feature_dim)
    :param features2: 第二个分支的特征 (batch_size, feature_dim)
    :param bins: 直方图分箱数量，用于近似概率分布
    :return: 互信息损失值
    """
    batch_size, feature_dim = features1.size()

    # 将特征归一化到 [0, 1]
    f1 = (features1 - features1.min()) / (features1.max() - features1.min() + 1e-6)
    f2 = (features2 - features2.min()) / (features2.max() - features2.min() + 1e-6)

    # 计算联合直方图
    joint_hist = torch.histc(torch.cat((f1, f2), dim=1).view(-1), bins=bins)
    joint_prob = joint_hist / joint_hist.sum()

    # 计算边缘直方图
    f1_hist = torch.histc(f1.view(-1), bins=bins)
    f2_hist = torch.histc(f2.view(-1), bins=bins)

    f1_prob = f1_hist / f1_hist.sum()
    f2_prob = f2_hist / f2_hist.sum()

    # 防止数值溢出：加稳定项 1e-6
    joint_prob = joint_prob + 1e-6
    f1_prob = f1_prob + 1e-6
    f2_prob = f2_prob + 1e-6

    # 互信息公式：MI = Σ p(x,y) * log(p(x,y) / (p(x) * p(y)))
    mi = joint_prob * (torch.log(joint_prob + 1e-6) - torch.log(f1_prob.unsqueeze(1) + 1e-6) - torch.log(f2_prob.unsqueeze(0) + 1e-6))
    return -torch.sum(mi)  # 最小化互信息

def cosine_distance(m, n):
    # 计算向量的余弦相似度
    cos_sim = F.cosine_similarity(m, n, dim=-1)
    # 余弦距离为 1 - 余弦相似度
    return 1 - cos_sim

# 计算 L1 和 L2 损失
def calculate_loss(f_ui, g_vi, f_uj, g_vj):
    # 计算相似性损失 L1
    d_fui_gvi = cosine_distance(f_ui, g_vi)
    d_fuj_gvj = cosine_distance(f_uj, g_vj)
    L1 = 0.5 * (d_fui_gvi + d_fuj_gvj)

    # 计算差异性损失 L2
    d_fui_fuj = cosine_distance(f_ui, f_uj)
    d_gvi_gvj = cosine_distance(g_vi, g_vj)
    L2 = -0.5 * (d_fui_fuj + d_gvi_gvj)

    # 总损失 L_total
    L_total = L1 + L2
    return L_total.mean()

def caculate_similarity_and_distance(ori_vec, dif_vec):
    # # 矩阵乘法
    # ori_mm = torch.mm(ori_vec, ori_vec.T)  # [32, 32]
    # dif_mm = torch.mm(dif_vec, dif_vec.T)  # [32, 32]
    # ori_dif = torch.mm(ori_vec, dif_vec.T)  # [32, 32]
    #
    # # 计算 L2 范数
    # ori_norm = torch.norm(ori_vec, dim=1, keepdim=True)  # [32, 1]
    # dif_norm = torch.norm(dif_vec, dim=1, keepdim=True)  # [32, 1]
    #
    # # 计算余弦相似度矩阵
    # ori_mm = 1 - (ori_mm / (ori_norm * ori_norm.T ) )
    # dif_mm = 1 - (dif_mm / (dif_norm * dif_norm.T ) )
    #
    # # 创建掩码来排除对角线元素
    # mask = ~torch.eye(ori_vec.size(0), dtype=torch.bool)  # [32, 32]
    #
    # # 计算非对角线的平均距离
    # avg_distance_ori = ori_mm[mask].mean()
    # avg_distance_dif = dif_mm[mask].mean()
    # avg_distance = (-1) * (avg_distance_ori + avg_distance_dif) / 2
    #
    # # 计算 ori_vec 和 dif_vec 间的余弦相似度（只考虑对角线元素）
    # similarity = 1 - (ori_dif / (ori_norm * dif_norm.T + 1e-8)).diag().mean()
    #
    # return avg_distance + similarity
    # distance1 = cosine_distance(ori_vec, ori_vec)
    # distance2 = cosine_distance(dif_vec, dif_vec)
    # # 创建掩码来排除对角线元素
    # mask = ~torch.eye(ori_vec.size(0), dtype=torch.bool)  # [32, 32]
    # # 计算非对角线的平均距离
    # avg_distance = ((-0.5) * (distance1 + distance2))[mask].mean()
    # # distance = 0.5 * (distance1 + distance2)
    # # 计算两个输入对应样本之间的相似性  距离小
    # similarity = cosine_distance(ori_vec, dif_vec).diag().mean()
    # return avg_distance + similarity
    # 计算同一batch内各向量的距离  距离大  上三角与下三角相同
    alpha = 0.8
    distance1 = 1 - torch.mm(ori_vec, ori_vec.T) / (torch.norm(ori_vec, dim=1).unsqueeze(1) * torch.norm(ori_vec, dim=1).unsqueeze(0))
    distance2 = 1 - torch.mm(dif_vec, dif_vec.T) / (
                torch.norm(dif_vec, dim=1).unsqueeze(1) * torch.norm(dif_vec, dim=1).unsqueeze(0))
    # 创建掩码来排除对角线元素
    mask = ~torch.eye(distance1.size(0), dtype=torch.bool)
    distance = ((-0.25) * (distance1 + distance2))
    distance = distance[mask]  #torch.Size([992])
    distance = distance.mean()
    #计算两个输入对应样本之间的相似性  距离小
    # similarity = 1 - torch.mm(ori_vec, dif_vec.T) / (torch.norm(ori_vec, dim=1).unsqueeze(1) * torch.norm(dif_vec, dim=1).unsqueeze(0))
    # similarity = similarity.diag()
    # similarity = similarity.mean()#torch.Size([32])
    similarity = F.cosine_similarity(ori_vec, dif_vec)
    similarity = 1 - torch.mean(similarity)
    return (1 - alpha) * distance + alpha *similarity
    # return similarity

    # cosine_similarity是一个方阵，其中cosine_similarity[i, j]是X[i]和X[j]的余弦相似度

# 训练函数
def train(model, train_loader, test_loader, val_loader, criterion, optimizer, scheduler, num_epochs, data_name, patch_size):
    time_start = time.time()  # 记录开始时间
    best_loss = 9999
    total_loss = 0
    epoch_avg_loss = utils.AvgrageMeter()
    patience_counter = 0
    patience = 15
    orthogonality_loss_fn = OrthogonalityLoss()
    alpha = 0.3
    beta = 0.1
    theata = 0.1
    for epoch in range(num_epochs):
        model.train()
        epoch_avg_loss.reset()
        for i, (inputs, labels, mask) in enumerate(train_loader):
            # inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            #置信度
            # ori_o, dif_o, outputs = model(inputs)
            # loss1 = criterion(ori_o, labels)
            # loss2 = criterion(dif_o, labels)
            # loss = criterion(outputs, labels)
            # loss = loss + 0.3*(loss1 + loss2)
            #加入相似性和距离损失
            outputs, ori_vec, dif_vec = model(inputs, mask)
            loss1 = criterion(outputs, labels)
            loss2 = caculate_similarity_and_distance(ori_vec, dif_vec)
            # loss3 = orthogonality_loss_fn(ori_vec, dif_vec)
            loss4 = inter_class_loss_optimized(ori_vec, labels)
            loss5 = inter_class_loss_optimized(dif_vec, labels)
            # loss2 = mutual_information_loss(ori_vec, dif_vec)
            loss = loss1 + alpha * loss2 + beta * loss4 + theata * loss5
            #普通损失
            # outputs = model(inputs, mask)
            # loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_avg_loss.update(loss.item(), inputs.shape[0])
        # 在每个 epoch 结束后更新学习率
        # scheduler.step()
        model.eval()  # 切换到评估模式
        val_loss = 0.0
        with torch.no_grad():  # 禁用梯度计算
            for inputs, labels, mask in val_loader:
                # output = model(inputs, mask)
                # loss = criterion(output, labels)
                outputs, ori_vec, dif_vec = model(inputs, mask)
                loss_1 = criterion(outputs, labels)
                loss_2 = caculate_similarity_and_distance(ori_vec, dif_vec)
                loss_3 = orthogonality_loss_fn(ori_vec, dif_vec)
                loss_4 = inter_class_loss_optimized(ori_vec, labels)
                loss_5 = inter_class_loss_optimized(dif_vec, labels)
                # loss_2 = mutual_information_loss(ori_vec, dif_vec)
                loss = loss_1 + alpha * loss_2 #+ beta * loss_4 + theata * loss_5
                val_loss += loss.item()
        val_loss /= len(val_loader)  # 计算平均验证损失
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "model\\best_model.pt")
            # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            # 早停逻辑
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break
        if (epoch+1) % 5 == 0:
            tra_pred, tra_label = predict(model, train_loader, data_name)
            val_pred, val_label = predict(model, val_loader, data_name)

            res_tra_accuracy = compution_accuracy(tra_pred, tra_label, data_name)
            res_accuracy = compution_accuracy(val_pred, val_label, data_name)
            current_lr = optimizer.param_groups[0]['lr']
            print(
                '[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [tra:] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [val:] [loss: %.5f] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [lr: %.5f]' % (
                epoch + 1,
                epoch_avg_loss.get_avg(),
                total_loss / (epoch + 1),
                res_tra_accuracy['oa'], res_tra_accuracy['aa'], res_tra_accuracy['kappa'],
                val_loss, res_accuracy['oa'], res_accuracy['aa'], res_accuracy['kappa'], current_lr))


    print('Finished Training')
    time_end = time.time()
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(f'训练时长：{time_sum}')
    # path = 'model' + '.pth'
    # torch.save(model, path)


    #test
    print("测试...")
    strat_test = time.time()
    model.load_state_dict(torch.load("model\\best_model.pt"))
    y_pred_test, y_test = predict(model, test_loader, data_name)

    #计算精度
    res_accuracy = compution_accuracy(y_pred_test, y_test, data_name)
    with open(f'.\\{data_name}\\testdata.txt', 'a') as file:
        file.write(f"ps={patch_size}{res_accuracy}\n")
    end_test = time.time()
    test_time = end_test - strat_test
    print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (
    epoch + 1, res_accuracy['oa'], res_accuracy['aa'], res_accuracy['kappa'], str(y_test.shape)))
    print(f'测试时长：{test_time}')
    print("finished training")
    return True
