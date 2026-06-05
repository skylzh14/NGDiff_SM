import os,sys
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import torchvision
from torchvision import transforms 
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import hdf5storage as hdf5
from sklearn.manifold import TSNE
from matplotlib.animation import FuncAnimation

from data import HSIDataLoader, TestDS, TrainDS
from unet3d import SimpleUnet
from diffusion import Diffusion
from utils import AvgrageMeter, recorder, show_img
from utils import device

batch_size = 8
patch_size = 64
select_spectral = []
spe = 104
channel = 1

epochs = 2000
lr = 1e-4
T=6#500  15
rgb = [1,5,8]#[50,60,100]
model_load_path = "./save_model/....."
model_name = "model_name.pkl"
save_feature_path_prefix = "D:/save_feature/....."

TList = [1]#[5, 10, 50, 100, 200, 400]



def plot_by_imgs(imgs, rgb=[0,1,2]):#rgb=[1,100,199]
    assert len(imgs) > 0
    batch, c, s, h, w = imgs[0].shape
    print(f'batch:{batch},imgs:{len(imgs)}')
    for i in range(batch):
        plt.figure(figsize=(12,8))
        for j in range(len(imgs)):
            plt.subplot(1,len(imgs),j+1)
            # img = imgs[j][i,0,rgb,:,:]
            img = torch.log(imgs[j][i,0,0,:,:] + imgs[j][i,0,1,:,:] + imgs[j][i,0,2,:,:] + 1)
            # show_img(img,)
            plt.imshow(img,cmap="gray")
        plt.show()            
    
def plot_by_images_v2(imgs, rgb=[0,4,8]):
    '''
    input image shape is (spectral, height, width)
    '''
    assert len(imgs) > 0
    s,h,w = imgs[0].shape
    plt.figure(figsize=(12,8))
    img = imgs[len(imgs)-1][rgb,:,:]
    show_img(img)

    # show_img(img)
    # for j in range(len(imgs)):
    #     plt.subplot(1,len(imgs),j+1)
    #     img = imgs[j][rgb,:,:]
    #     show_img(img)
    plt.show()            
    
def plot_spectral(x0, recon_x0, num=3):
    '''
    x0, recon_x0 shape is (batch, channel, spectral, h, w)
    '''
    batch, c, s, h ,w = x0.shape
    step = h // num
    plt.figure(figsize=(20,5))
    for ii in range(num):
        i = ii * step 
        x0_spectral = x0[0,0,:,i,i]
        recon_x0_spectral = recon_x0[0,0,:,i,i]
        plt.subplot(1,num,ii+1)
        plt.plot(x0_spectral, label="x0")
        plt.plot(recon_x0_spectral, label="recon")
        plt.legend()
    plt.show()
    

def recon_all_fig(diffusion, model, splitX, dataloader, big_img_size=[145, 145]):
    '''
    X shape is (spectral, h, w) => (batch, channel=1, 200, 145, 145)
    '''
    # 1. reconstruct
    t = torch.full((1,), diffusion.T-1, device=device, dtype=torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(torch.from_numpy(splitX.astype('float32')), t, device)
    _, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num = 2)
    
    # ---just for test---
    # recon_from_xt.append(torch.from_numpy(splitX.astype('float32')))
    # plot_by_imgs(recon_from_xt, rgb=rgb)

    # ---------

    res_xt_list = []
    for tempxt in recon_from_xt:
        big_xt = dataloader.split_to_big_image(tempxt.numpy()) 
        res_xt_list.append(big_xt)
    ori_data, _ = dataloader.get_ori_data()
    res_xt_list.append(ori_data)
    np.save('res_xt_list.npy', np.array(res_xt_list, dtype=object), allow_pickle=True)
    print("suc")
    exit()
    plot_by_images_v2(res_xt_list,rgb=rgb)
    
def sample_by_t(diffusion, model, X):
    num = 10
    choose_index = [3]
    x0 = torch.from_numpy(X[choose_index,:,:,:,:]).float()

    step = diffusion.T // num
    for ti in range(10, diffusion.T, step):
        t = torch.full((1,), ti, device=device, dtype=torch.long)
        xt, tmp_noise = diffusion.forward_diffusion_sample(x0, t, device)
        _, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num = 5)
        recon_x0 = recon_from_xt[-1]
        recon_from_xt.append(x0)
        print('---',ti,'---')
        plot_by_imgs(recon_from_xt, rgb=rgb)
        print("x0", x0.shape, "recon_x0", recon_x0.shape)
        plot_spectral(x0, recon_x0)

def inference_mini_batch(model, xt, t):
    mini_batch_size = 4 #4
    batch, channel, c, h, w = xt.shape
    step = batch // mini_batch_size + 1

    res_feature_t_list = []
    for i in range(step):
        start = i * mini_batch_size
        end = (i+1) * mini_batch_size
        temp_xt = xt[start:end, :, :, :, :]
        if temp_xt.shape[0] <= 0:
            break
        noise_pred = model(temp_xt, t, t)#feature=True
        temp_feature_t_list = model.return_features()
        if len(res_feature_t_list) == 0:
            res_feature_t_list = temp_feature_t_list[:]
        else:
            assert len(res_feature_t_list) == len(temp_feature_t_list)
            temp_res = []
            for j in range(len(temp_feature_t_list)):
                temp_res.append(np.concatenate([res_feature_t_list[j], temp_feature_t_list[j]]))
            res_feature_t_list = temp_res[:]
    for fea in res_feature_t_list:
        print(fea.shape)
    return res_feature_t_list


def classify(feature,num_class):
    print("开始分类")
    Softmax_linear = nn.Sequential(nn.Linear(512, num_class))
    feature_flatten = feature.float().reshape([512*512,-1])
    s_feature = Softmax_linear(feature_flatten)
    result = F.softmax(s_feature, -1)
    classification_map = torch.argmax(result, 1).reshape([512, 512]).cpu() + 1
    y_pre = classification_map.numpy()
    print(y_pre)
    # y_pre = [x for x in y_pre]
    hdf5.savemat(r'/save_path/',
            {'y_re': y_pre})
    print("分类结束")
def show_feature(feature,num_class):
    print(feature.shape)
    feature = feature[245:367, 191:315, :]
    gt = hdf5.loadmat("path of label.mat")["label"]
    gt = gt[245:367, 191:315] - 1
    label = gt.reshape(np.prod(gt.shape[:2]), )
    feature = feature.reshape(np.prod(feature.shape[:2]),np.prod(feature.shape[2:]))
    # 使用t-SNE算法进行降维
    # tsne = TSNE(n_components=3, random_state=42, perplexity=60)
    tsne = TSNE(n_components=2, random_state=42, perplexity=60)
    embedding = tsne.fit_transform(feature)
    # 创建散点图进行可视化
    fig, ax = plt.subplots(figsize=(8, 6))  # 2D
    for i, c in zip(range(3), ['red', 'green', 'blue']):  # 2D
        ax.scatter(embedding[label == i, 0], embedding[label == i, 1], c=c, label=str(i), s=10)
    # 创建动画
    ani = FuncAnimation(fig, update, frames=[], interval=50)  # 2d
    plt.savefig('2D.png', dpi=600)
    plt.show()
def update(frame):  #2d
    pass

def inference_by_t(dataloader, diffusion, model, X, ti):
    '''
    X shape is (batch, channel, spe, h, w)
    '''

    X = torch.from_numpy(X).float()
    t = torch.full((1,), ti, device=device, dtype=torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(X, t, device)

    # 1. 显示调用模型直接获取隐层特征
    # noise_pred = model(xt, t, feature=True)
    # feature_t_list = model.return_features()
    feature_t_list = inference_mini_batch(model, xt, t)
    for index, feature_matrix in enumerate(feature_t_list):
        path = "%s/t%s_%s.pkl" % (save_feature_path_prefix, ti, index)
        #np.save(path, feature_matrix)
        print("save matrix t=%s, index=%s done." % (ti, index))
        # feature_matrix shape is (batch, channel, spe, h, w)
        fb, fc, fs, fh, fw = feature_matrix.shape
        temp = feature_matrix.reshape((fb,fc*fs, fh, fw)).transpose((0,2,3,1))
        full_feature_img = dataloader.reconstruct_image_by_light_split(temp, pathch_size=patch_size)
        path = "%s/t%s_%s_full.pkl" % (save_feature_path_prefix, ti, index)
        np.save(path, full_feature_img)
        # if index == 0:
            # classify(torch.from_numpy(full_feature_img),3)
            # show_feature(torch.from_numpy(full_feature_img),3)
        print("save full matrix done. t=%s, index=%s, shape=%s" % (ti, index, str(full_feature_img.shape)))
        if ti != 1:
            break

    # 2. 对模型在该t下进行完全恢复尝试验证
    choose_index = [3]
    show_x0 = X[choose_index,:,:,:,:]
    show_xt = xt[choose_index, :,:,:,:]
    print(f'show_xt:{show_xt.shape}')
    if t >= 5:
        _, recon_from_xt = diffusion.reconstruct(model, xt=show_xt, tempT=t, num = 5) # recon_from_xt[0] shape (batch, channel, spe, h, w)
        recon_x0 = recon_from_xt[-1]
        print(recon_x0.shape)
        recon_from_xt.append(show_x0)
        print(f'recon_from_xt:{recon_from_xt[0].shape}')
        print('---',ti,'---')

        plot_by_imgs(recon_from_xt, rgb=rgb)
        plot_spectral(show_x0, recon_x0)
        show_x0 = show_x0.detach().cpu().numpy()
        recon_x0 = recon_x0.detach().cpu().numpy()
        hdf5.savemat('yuanlai1400_1200.mat', {'result': show_x0})
        hdf5.savemat('shengcheng1400_12001.mat', {'result': recon_x0})



def sample_eval(diffusion, model, X):
    all_size, channel, spe, h, w = X.shape
    num = 5
    step = all_size // num
    r,g,b = 1, 100, 199
    choose_index = list(range(0, all_size, step))
    x0 = torch.from_numpy(X[choose_index,:,:,:,:]).float()

    use_t = 499
    # from xt
    t = torch.full((1,), use_t, device=device, dtype=torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(x0, t, device)
    _, recon_from_xt = diffusion.reconstruct(model, xt=xt, tempT=t, num = 10)
    recon_from_xt.append(x0)
    plot_by_imgs(recon_from_xt, rgb=rgb)
    
    # from noise
    t = torch.full((1,), use_t, device=device, dtype=torch.long)
  
    _, recon_from_noise = diffusion.reconstruct(model, xt=x0, tempT=t, num = 10, from_noise=True, shape=x0.shape)
    plot_by_imgs(recon_from_noise, rgb=rgb)


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("save model done. path=%s" % path)


def extract():
    dataloader = HSIDataLoader({"data":{"data_sign":"512_512", "padding":False, "batch_size":batch_size, "patch_size":patch_size, "select_spectral":select_spectral}})
    train_loader,X,Y = dataloader.generate_torch_dataset(light_split=True)
    diffusion = Diffusion(T=T)
    model = SimpleUnet(_image_channels=channel)
    assert os.path.exists(model_load_path)
    if not os.path.exists(save_feature_path_prefix):
        os.makedirs(save_feature_path_prefix)

    model_path = "%s/%s" % (model_load_path, model_name)
    print('model path is ', model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
  
    model.to(device)
    print("load model done. model_path=%s" % (model_path))

    #注释，先使用一个t
    # for ti in TList:
    #     inference_by_t(dataloader, diffusion, model, X, ti)
    #     print("feature extract t=%s done." % ti)
    # recon_all_fig(diffusion,model,X,dataloader)
    inference_by_t(dataloader, diffusion, model, X, 1)  # 只提取t=5,t可以改
    print("feature extract t=%s done." % 0)
    
    print('done.')

if __name__ == "__main__":
    extract()
