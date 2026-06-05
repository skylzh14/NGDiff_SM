import numpy as np
import scipy.io as sio
import hdf5storage as hdf5
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time, json
import os, sys
from utils import device
""" Training dataset"""


class DataSetIter(torch.utils.data.Dataset):
    def __init__(self, _base_img, _base_labels, _index2pos, _margin, _patch_size, _append_dim, mask) -> None:
        self.base_img = _base_img  # 全量数据包括margin (145+2margin * 145+2margin * spe)
        self.mask = mask #边界区域掩码，全部(512*512)
        self.base_labels = _base_labels  # 全量数据无margin (145 * 145)
        self.index2pos = _index2pos  # 训练数据 index -> (x, y) 对应margin后base_img的中心点坐标
        self.size = len(_index2pos)

        self.margin = _margin
        self.patch_size = _patch_size
        self.append_dim = _append_dim  # False

    def __getitem__(self, index):
        start_x, start_y = self.index2pos[index]
        patch = self.base_img[start_x:start_x + 2 * self.margin + 1, start_y:start_y + 2 * self.margin + 1, :]
        if self.append_dim:
            patch = np.expand_dims(patch, 0)  # [channel=1, h, w, spe]
            patch = patch.transpose((0, 3, 1, 2))  # [c, spe, h, w]
        else:
            patch = patch.transpose((2, 0, 1))  # [spe, h, w]
        label = self.base_labels[start_x, start_y] - 1
        mask_sample = self.mask[start_x, start_y]
        # mask_sample = np.expand_dims(mask_sample, -1)

        # print(index, patch.shape, start_x, start_y, label)
        return torch.FloatTensor(patch).to(device), torch.LongTensor(label.reshape(-1))[0].to(device), torch.tensor(mask_sample.reshape(-1))[0].to(device)

    def __len__(self):
        return self.size


class HSIDataLoader(object):
    def __init__(self, label, ratio, TR, TE, val, patch_size, batch_size, bianjie, j) -> None:

        # self.data = pol_data  # 原始读入X数据 shape=(h,w,c),diffusion feature
        # self.dif_data = dif_data   #diffusion_feature
        self.labels = label  # 原始读入Y数据 shape=(h,w,1)
        self.ratio = ratio
        self.TR = TR  # 标记训练数据
        self.TE = TE  # 标记测试数据
        self.val = val  #验证数据
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.no_patch = j
        self.spectracl_size = 300
        self.pca_num = 300

        self.if_numpy = False  # False
        self.append_dim = False  # False ：不添加新的维度，True：添加一个新的维度，来进行3d的torch卷积训练
        self.norm_type = 'max_min'# 'none', 'max_min', 'mean_var'

        self.mask = bianjie



    def _padding(self, X, margin=2):
        # pading with zeros
        w, h, c = X.shape
        new_x, new_h, new_c = w + margin * 2, h + margin * 2, c
        returnX = np.zeros((new_x, new_h, new_c))
        start_x, start_y = margin, margin
        returnX[start_x:start_x + w, start_y:start_y + h, :] = X
        return returnX

    def get_valid_num(self, y):
        tempy = y.reshape(-1)
        validy = tempy[tempy > 0]
        print('valid y shape is ', validy.shape)
        return validy.shape[0]

    def get_train_test_num(self, TR, TE, val):  # 统计label>0的数据量
        train_num, test_num, val_num = TR[TR > 0].reshape(-1).size, TE[TE > 0].reshape(-1).size, val[val > 0].reshape(-1).size
        print("train_num=%s, test_num=%s, val_num=%s" % (train_num, test_num, val_num))
        return train_num, test_num, val_num

    def get_train_test_patches(self, X, y, TR, TE, val):
        h, w, c = X.shape
        # 给 X 做 padding
        windowSize = self.patch_size  # 13
        margin = int((windowSize - 1) / 2)  # padding数，6
        zeroPaddedX = self._padding(X, margin=margin)  # 补0

        # 确定train和test的数据量
        train_num, test_num, val_num = self.get_train_test_num(TR, TE, val)
        trainX_index2pos = {}  # 位置
        testX_index2pos = {}
        valX_index2pos = {}
        all_index2pos = {}

        patchIndex = 0
        trainIndex = 0
        testIndex = 0
        valIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                start_x, start_y = r - margin, c - margin
                tempy = y[start_x, start_y]
                temp_tr = TR[start_x, start_y]
                temp_te = TE[start_x, start_y]
                temp_val = val[start_x, start_y]
                if temp_tr > 0 and temp_te > 0:
                    print("here", temp_tr, temp_te, r, c)
                    raise Exception("data error, find sample in trainset as well as testset.")
                if temp_tr > 0 and temp_val > 0:
                    print("here", temp_tr, temp_val, r, c)
                    raise Exception("data error, find sample in trainset as well as valset.")
                if temp_te > 0 and temp_val > 0:
                    print("here", temp_te, temp_val, r, c)
                    raise Exception("data error, find sample in testset as well as valset.")

                if temp_tr > 0:  # train data
                    trainX_index2pos[trainIndex] = [start_x, start_y]
                    trainIndex += 1
                elif temp_te > 0:
                    testX_index2pos[testIndex] = [start_x, start_y]
                    testIndex += 1
                elif temp_val > 0:
                    valX_index2pos[valIndex] = [start_x, start_y]
                    valIndex += 1
                all_index2pos[patchIndex] = [start_x, start_y]
                patchIndex = patchIndex + 1
        return zeroPaddedX, y, trainX_index2pos, testX_index2pos, valX_index2pos, all_index2pos, margin, self.patch_size

    def applyPCA(self, X, numComponents=30):  # PCA numComponents：900
        newX = np.reshape(X, (-1, X.shape[2]))  # 将最后一个维度展开成一列 (h*w,X.shape[2])
        pca = PCA(n_components=numComponents, whiten=True)  # 降维，并将数据白化，即各个维度之间不相关。
        newX = pca.fit_transform(newX)  # 拟合降维,
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))  # 再reshape为h*w*900
        return newX

    def mean_var_norm(self, data):
        print("use mean_var norm...")
        h, w, c = data.shape
        data = data.reshape(h * w, c)
        data = StandardScaler().fit_transform(data)
        data = data.reshape(h, w, c)
        return data

    def data_preprocessing(self, data):
        '''
        1. normalization
        2. pca
        3. spectral filter
        data: [h, w, spectral]
        '''
        if self.norm_type == 'max_min':  # 执行
            norm_data = np.zeros(data.shape)  # 创建全0矩阵
            for i in range(data.shape[2]):  # 遍历每个光谱维度，归一化
                input_max = np.max(data[:, :, i])
                input_min = np.min(data[:, :, i])
                norm_data[:, :, i] = (data[:, :, i] - input_min) / (input_max - input_min)
        elif self.norm_type == 'mean_var':
            norm_data = self.mean_var_norm(data)
        else:
            norm_data = data

        if data.shape[2] > 57:
            if self.pca_num > 0:
                print('before pca')
                pca_data = self.applyPCA(norm_data, int(self.pca_num))  # norm_data归一化后的数据，900
                norm_data = pca_data  # h*w*900
                print('after pca')
        if self.spectracl_size > 0:  # 按照给定的spectral size截取数据
            norm_data = norm_data[:, :, :self.spectracl_size]
        return norm_data

    def generate_numpy_dataset(self):
        # 2. 数据预处理 主要是norm化
        norm_data = self.data_preprocessing(self.data)

        print(
            '[data preprocessing done.] data shape data=%s, label=%s' % (str(norm_data.shape), str(self.labels.shape)))

        # 3. reshape & filter
        h, w, c = norm_data.shape
        norm_data = norm_data.reshape((h * w, c))
        norm_label = self.labels.reshape((h * w))
        TR_reshape = self.TR.reshape((h * w))
        TE_reshape = self.TE.reshape((h * w))
        TrainX = norm_data[TR_reshape > 0]
        TrainY = norm_label[TR_reshape > 0]
        TestX = norm_data[TE_reshape > 0]
        TestY = norm_label[TE_reshape > 0]
        train_test_data = norm_data[norm_label > 0]
        train_test_label = norm_label[norm_label > 0]

        print('------[data] split data to train, test------')
        print("X_train shape : %s" % str(TrainX.shape))
        print("Y_train shape : %s" % str(TrainY.shape))
        print("X_test shape : %s" % str(TestX.shape))
        print("Y_test shape : %s" % str(TestY.shape))

        return TrainX, TrainY, TestX, TestY, norm_data



    def prepare_data(self):  # 准备数据
        # 2. 数据预处理 主要是norm化
        # norm_data = self.data_preprocessing(self.data)
        # norm_dif_data = self.data_preprocessing(self.dif_data)
        # print(
        #     '[data preprocessing done.] data shape data=%s, label=%s' % (str(norm_data.shape), str(self.labels.shape)))
        # print(
        #     '[ori_data preprocessing done.] data shape ori_data=%s, label=%s' % (
        #     str(norm_dif_data.shape), str(self.labels.shape)))
        #
        # #将处理后的原始数据和diffusion数据结合起来
        # norm_data = np.concatenate((norm_data, norm_dif_data), axis=2)
        # #保存合并后的数据，减少时间消耗
        # # hdf5.savemat('./900_700/ori_9_dif300.mat', {'norm_data': norm_data})
        # # hdf5.savemat('./900_700/ene_54_dif300.mat', {'norm_data': norm_data})
        # # hdf5.savemat('./1300_1200/ori_9_dif300.mat', {'norm_data': norm_data})
        # # hdf5.savemat('./512_512/ori_9_dif300_63.mat', {'norm_data': norm_data})
        # # hdf5.savemat('./212_387/ori_9_dif300.mat', {'norm_data': norm_data})
        # # hdf5.savemat('./512_512/ene_54_dif300.mat', {'norm_data': norm_data})
        # hdf5.savemat('./1400_1200/ori_9_dif300.mat', {'norm_data': norm_data})
        # print('meiwenti')
        #上述pca降维完成后且保存后，直接load保存下来的数据
        # norm_data = hdf5.loadmat('./1400_1200/ori_9_dif300.mat')['norm_data']
        # norm_data = hdf5.loadmat('./900_700/ori_9_dif300.mat')['norm_data']
        # norm_data = hdf5.loadmat('./900_700/ene_54_dif300.mat')['norm_data']
        # norm_data = hdf5.loadmat('./1300_1200/ori_9_dif300.mat')['norm_data']
        # norm_data = hdf5.loadmat('./900_1024/ori_9_dif300.mat')['norm_data']
        # norm_data = hdf5.loadmat('1300_1200/pattch/D1.mat')['diffusion']
        # norm_data = hdf5.loadmat('./512_512/ori_9_dif300.mat')['norm_data']
        norm_data = hdf5.loadmat(f"F:/pycharm/PyCharm_projects/postGraduate/sky/cnn_cla/1300_1200/pattch/D{self.no_patch}.mat")['diffusion']
        # norm_data = hdf5.loadmat('./512_512/ene_54_dif300.mat')['norm_data']
        # norm_data = hdf5.loadmat('./212_387/ori_9_dif300.mat')['norm_data']
        # 3. 获取patch 并形成batch型数据
        '''
        base_img : padding后的图像
        labels : self.labels
        train_index2pos : 训练集标签对应的位置 list
        test_index2pos : 测试集标签对应的位置 list
        all_index2pos : 所有标签对应的位置 list
        margin : 边缘补0数，6
        patch_size : 块大小，13
        '''
        base_img, labels, train_index2pos, test_index2pos, val_index2pos, all_index2pos, margin, patch_size \
            = self.get_train_test_patches(norm_data, self.labels, self.TR, self.TE, self.val)

        print('------[data] split data to train, test, val------')
        print("train len: %s" % len(train_index2pos))
        print("test len : %s" % len(test_index2pos))
        print("val len : %s" % len(val_index2pos))
        print("all len: %s" % len(all_index2pos))

        trainset = DataSetIter(base_img, labels, train_index2pos, margin, patch_size,
                               self.append_dim, self.mask)  # append_dim:True
        unlabelset = DataSetIter(base_img, labels, test_index2pos, margin, patch_size, self.append_dim, self.mask)  # 训练集里标签为0的
        testset = DataSetIter(base_img, labels, test_index2pos, margin, patch_size, self.append_dim, self.mask)
        valset = DataSetIter(base_img, labels, val_index2pos, margin, patch_size, self.append_dim, self.mask)
        allset = DataSetIter(base_img, labels, all_index2pos, margin, patch_size, self.append_dim, self.mask)
        return trainset, unlabelset, testset, valset, allset

    def generate_torch_dataset(self):
        # 0. 判断是否使用numpy数据集
        if self.if_numpy:  # False
            return self.generate_numpy_dataset()

        trainset, unlabelset, testset, valset, allset = self.prepare_data()


        multi = 1
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   drop_last=False
                                                   )
        unlabel_loader = torch.utils.data.DataLoader(dataset=unlabelset,
                                                     batch_size=int(self.batch_size * multi),
                                                     shuffle=False,
                                                     num_workers=0,
                                                     drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  drop_last=False
                                                  )
        val_loader = torch.utils.data.DataLoader(dataset=valset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  drop_last=False
                                                  )
        all_loader = torch.utils.data.DataLoader(dataset=allset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 drop_last=False
                                                 )

        return train_loader, unlabel_loader, test_loader, val_loader, all_loader

