import split_data as sd
import data_preprocess as dp
import hdf5storage as hdf5
import groupconv as gc
import trainer
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from trainer import Draw_Classification_Map
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
    bianjie = np.random.randint(1, 4, size=[1300, 1200])
    bianjie = (bianjie == 2)
    # h, w, c = data.shape
    c = 9
    dataloader = dp.HSIDataLoader(label, ratio, train_set, test_set, val_set, patch_size, batch_size, bianjie, j)
    train_loader, unlabel_loader, test_loader, val_loader, all_loader = dataloader.generate_torch_dataset()
    #训练
    model = gc.gc_net(c, 300, 4, 8, num_class)
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

