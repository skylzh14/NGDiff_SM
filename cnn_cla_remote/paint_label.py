#!/usr/bin/env python
# coding: utf-8
## Imports

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage as hdf5


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 1.0 / dpi,
                        ground_truth.shape[0] * 1.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        #print(x_list.shape)
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255. #黑色
            # y[index] = np.array([255, 255, 255]) / 255.
        if item == 1:
            y[index] = np.array([255, 0, 0]) / 255.    #红色
        if item == 2:
            y[index] = np.array([0, 255, 0]) / 255.    #绿色
        if item == 3:
            y[index] = np.array([153, 51, 250]) / 255.  #湖紫色
        if item == 4:
            y[index] = np.array([255, 255, 0]) / 255.  #黄色
        if item == 5:
            y[index] = np.array([0, 255, 255]) / 255.  #青色
        if item == 6:
            y[index] = np.array([255, 0, 255]) / 255.   #深红色
        if item == 7:
            y[index] = np.array([60, 179, 113]) / 255.  # 青绿色
        if item == 8:
            y[index] = np.array([255, 239, 213]) / 255.
        if item == 9:
            y[index] = np.array([139, 139, 0]) / 255.
        if item == 10:
            y[index] = np.array([178, 48, 96]) / 255.
        if item == 11:
            y[index] = np.array([156, 102, 31]) / 255.
        if item == 12:
            y[index] = np.array([124, 252, 0]) / 255.
        if item == 13:
            y[index] = np.array([221, 160, 221]) / 255.
        if item == 14:
            y[index] = np.array([0, 100, 0]) / 255.
        if item == 15:
            y[index] = np.array([70, 130, 180]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 17:
            y[index] = np.array([255, 255, 255]) / 255.  # 白色为背景色
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.  # 黑色
    return y
def fun(y_pre, gt_hsi,path):
    y_pre = (y_pre + 1).ravel()  # 拉成一维
    y_pre = y_pre.flatten()
    print(y_pre.shape)
    y_pre = list_to_colormap(y_pre)
    print(y_pre.shape)
    y_pre = np.reshape(y_pre, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    print(y_pre.shape)
    #path='C:/Users/极化SAR小组/Desktop/First_Work_Results/CVCNN/1400_1200/pic'
    classification_map(y_pre, gt_hsi, 600,
                   path + '.png')
    print('------Get classification maps successful-------')

'''
##生成全图
y_re = hdf5.loadmat("\label_pre.mat")['pre']

#y_re = y_re+1
gt_hs = hdf5.loadmat("label.mat")['label']
'''
