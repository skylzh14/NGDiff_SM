import split_data as sd
import data_preprocess as dp
import hdf5storage as hdf5
import cnn_3d_network as cnn_3d
import cnn_2d_network as cnn_2d
import _3dand2d as _32
import groupconv as gc
import new_net as nene
import trainer
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from trainer import Draw_Classification_Map
import MI_SSL_CNN as mi
import CNN_5Layers_double as c5
import paint_label as pl
from utils import device


dataname = "1300_1200"
for i in [2, 3, 4, 5]:
# for j in [1, 2, 3, 4]:#分块
    ratio = 5
    patch_size = 13
    batch_size = 128#32
    lr = 0.001
    num_class = 5
    in_channels = 1
    out_channels = 64
    epochs = 60#100
    #分数据集
    train_set, test_set, val_set, label= sd.generate_TR_TE(dataname, ratio, j)
    # data = hdf5.loadmat(f'E:/PyCharm_projects/ww/new_polsardata/512_512_XIAN/{dataname}_9.mat')['feature']
    # data = hdf5.loadmat(f'E:/matlab_projects/sky/step1_polsar hybrid edge-line detector/{dataname}_54.mat')['aa']#能量图数据
    # data = data.transpose(1, 2, 0)
    # data = hdf5.loadmat(f'E:/PyCharm_projects/ww/new_polsardata/512_512_XIAN/{dataname}_57.mat')['feature']#多特征数据
    # dif_data = np.load('F:\\pycharm\\PyCharm_projects\\postGraduate\\sky\\spectraldiff_diffusion\\codes\\save_feature\\1400_1200_fisher_diffusiont=1\\t1_0_full.pkl.npy')
    # data_xiagao = hdf5.loadmat(f'spank1.mat')['span']
    # data_xiagao = np.expand_dims(data_xiagao, axis=2)
    # data = np.concatenate((data, data_xiagao), axis=2)
    # bianjie = hdf5.loadmat(f".//r_map//r_map_{dataname}.mat")['r_map']
    bianjie = np.random.randint(1, 4, size=[1300, 1200])
    bianjie = (bianjie == 2)
    # mask_bianjie = torch.tensor(bianjie == 1)
    # mask_bianjie = mask_bianjie.unsqueeze(-1)
    # h, w, c = data.shape
    c = 9
    dataloader = dp.HSIDataLoader(label, ratio, train_set, test_set, val_set, patch_size, batch_size, bianjie, j)
    train_loader, unlabel_loader, test_loader, val_loader, all_loader = dataloader.generate_torch_dataset()
    #训练

    # model = cnn_3d.Simple3DCNN(num_class)#in_channels, out_channels,
    # model = cnn_2d.CNN_2D(300)
    # model = _32.CNN_3dand2d(300)
    # model = gc.group_conv(300, 4, 8)
    model = gc.gc_net(c, 300, 4, 8, num_class)
    # model = nene.gc_net(c, 300, 4, 8)
    # model = mi.MI_SSL(9, 300)
    # model = c5.CNN(9, 3)
    model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # 定义学习率调整策略为余弦退火
    scheduler = CosineAnnealingLR(optimizer, T_max=5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    trainer.train(model, train_loader, test_loader, val_loader, criterion, optimizer, scheduler, epochs, dataname, patch_size=patch_size)
    print("预测全图标签。。。。")
    y_pred_all, y_all = trainer.predict(model, all_loader, dataname, ratio=patch_size, keep_probability=True)
    all_accuracy = trainer.compution_accuracy(y_pred_all, y_all, dataname)
    h, w = label.shape
    y_pred_all = y_pred_all.reshape((h, w))
    hdf5.savemat(f'pre_y_{dataname}ps={i}patch={j}.mat', {'pre': y_pred_all})
    print("存储完成，结束。。。")
# print("开始画图。。。。")
# # Draw_Classification_Map(y_pred_all, ".\\map\\map")
# pl.fun(y_pred_all, label, ".\\map_1400\\1400_1200_atten特征+simi+inter损失")#mapmaskdifyunzhi_dif仅做两层卷积
# print("画图结束。。。。")

