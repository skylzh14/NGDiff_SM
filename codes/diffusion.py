import torch
import torchvision
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from data import HSIDataLoader
import numpy as np
from plot import show_tensor_image
from utils import device
from scipy.stats import gamma, invgamma
import math
import torch.distributions as dist
from scipy.stats import f
from torch.distributions import MixtureSameFamily, Normal, Categorical


class Hyper(nn.Module):#带有超参的方法
    def __init__(self, alpha=4.0, beta=1.0, sigma=0.05, epsilon=1e-3, channel = 16) -> None:
        super().__init__()
        # 固定分布参数
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.epsilon = epsilon
        self.T_max = 6
        self.channel = channel

        # 可学习强度控制参数（初始化值可根据任务微调）
        init_lambda_gauss= torch.tensor([0.0] * 3 + [1.0] * (self.channel-3))
        self.lambda_invGamma = nn.Parameter(torch.full((self.channel,), 1.0))  # [9]  # 可学习，乘性噪声控制
        # self.lambda_gauss = nn.Parameter(torch.full((self.channel,), 0.5))  # 可学习，加性噪声控制
        self.lambda_gauss = nn.Parameter(init_lambda_gauss.clone())  # 乘性噪声控制



    def cosine_lambda_schedule(self, T_max = 6, s=0.008, device='cpu'):
        steps = torch.arange(1, T_max + 1, dtype=torch.float32, device=device)
        f = torch.cos(((steps / T_max + s) / (1 + s)) * math.pi / 2) ** 2
        f0 = torch.cos(torch.tensor((s / (1 + s)) * math.pi / 2, device=device)) ** 2
        alpha_bars = f / f0
        alpha_bars = torch.cat([torch.tensor([1.0], device=device), alpha_bars])
        lambdas = alpha_bars[1:] / alpha_bars[:-1]
        return lambdas

    def forward(self, x0: torch.Tensor, t: torch.Tensor, device='cpu'):
        B, _, C, H, W = x0.shape
        device = x0.device

        t = t.to(device).view(B, 1, 1, 1, 1)  # [B,1,1,1,1]
        # 构造 sigma_t：线性划分 [0.001, 0.02] -> T_max 个值
        # sigma_list = torch.linspace(0.001, 0.02, self.T_max, device=device)  # [T_max]

        # 1) 乘性噪声 p ~ InverseGamma(alpha, beta)
        gamma_dist = torch.distributions.Gamma(self.alpha, self.beta)
        # 只采样一个样本（所有 B 批次共享），形状 [C, H, W]
        g = gamma_dist.sample((C, H, W)).to(device)
        # 归一化（可选）：确保均值稳定
        g = g / g.mean()
        # 扩展到 [B, C, H, W]
        g = g.unsqueeze(0).expand(B, -1, -1, -1)  # [B, C, H, W]
        p = 1.0 / g  # [B,C,H,W]
        # p = g
        p = p.unsqueeze(1)  # [B,1,C,H,W]

        # 构造 q = lambda_q * (p + 1)
        lambda_inv = self.lambda_invGamma.view(1, 1, -1, 1, 1).to(device)
        q = lambda_inv * (p + 1.0)  # [B,1,C,H,W]
        # q = p + 1.0

        # 余弦调度得到 lambdas，长度T
        # lambdas_all = self.cosine_lambda_schedule(self.T_max, device=device)  # [T_max]
        lambdas_all = torch.linspace(0.001, 0.02, self.T_max, device=device)  # [T_max]

        # 根据每个样本的t索引，计算加权和 sum_{k=1}^t q^{t-k} * lambda_k
        xt = torch.zeros_like(x0)
        q_pow_t = q.pow(t)  # [B,1,C,H,W]

        for b in range(B):
            tb = t[b].item()
            if tb == 0:
                xt[b] = x0[b]
                continue

            # 计算乘性累积 q^t
            part1 = q_pow_t[b]

            # 计算加权和
            lambdas_t = lambdas_all[:tb]  # [tb]
            powers = q_pow_t[b].pow(0)  # 先全1，待后面调整形状

            # 计算 q^{t-k} 的张量形式，需要广播
            # 这里 t是标量，k是时间步索引，q形状 [1,C,H,W]
            # 先构造 q^{t-k} 对应张量：
            q_val = q[b]  # [1,C,H,W]
            weighted_sum = torch.zeros_like(x0[b])
            for k in range(tb):
                power = tb - (k + 1)
                q_term = q_val.pow(power)
                weighted_sum += lambdas_t[k] * q_term

            # 生成标准高斯噪声a，形状[B,1,C,H,W]
            a = torch.randn((1, C, H, W), device=device)
            lambda_gauss = self.lambda_gauss.view(1, -1, 1, 1).to(device)  # [1,C,1,1]
            a = lambda_gauss * a  # 缩放噪声强度

            xt[b] = part1 * x0[b] + weighted_sum * a


        return xt, x0



class Diffusion(object):
    def __init__(self, T=1000) -> None:
        # 初始化方法，设置时间步数 T，默认为1000
        self.T = T

        # 通过调用 _linear_beta_schedule 方法生成线性的 beta 调度
        self.betas = self._linear_beta_schedule(timesteps=self.T)

        # Pre-calculate different terms for closed form
        # alphas 表示 1 - betas
        self.alphas = 1. - self.betas

        # alphas 的累积乘积，沿轴 0 计算
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        # 对 alphas_cumprod 进行前向填充（padding），用 1.0 填充，并去掉最后一个元素，补值为1，因为1*任何数还是任何数，从而取得alpha（t-1）的值
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 计算 1.0 / alphas 的平方根
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # 计算 alphas_cumprod 的平方根
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        # 计算 1 - alphas_cumprod 的平方根
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # 计算后验方差，根据给定的公式使用预先计算的项
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # alphas 的累加，沿轴 0 计算
        self.betas_cumsum = torch.cumsum(self.betas, axis=0)

    def generate_gamma(self, gamma_noise):#每一步的gamma噪声
        gamma_all = [0] * self.T
        for i in range(self.T):
            gamma_noise = torch.tensor(gamma.rvs(a=4, scale=1, size=(1, 64, 64)), dtype=torch.float).to(device) #* (2 + 0.01 * i)
            gamma_noise = gamma_noise / torch.mean(gamma_noise)
            gamma_all[i] = gamma_noise.clone()

        return gamma_all

    def generate_gamma_t_reverse(self, gamma_noise, t, device="cpu"):#逆向累乘
        gamma_reverse = torch.ones(gamma_noise[0].shape)
        gamma_reverse_all = [0] * (self.T + 1)
        t_end = t[0]
        i = self.T
        gamma_reverse_all[i] = gamma_reverse.clone()
        while i > t_end:
            gamma_reverse *= (gamma_noise[i-1].to(device) + 1)
            gamma_reverse = gamma_reverse / torch.mean(gamma_reverse)
            gamma_reverse_all[i-1] = gamma_reverse.clone()
            i -= 1
        return gamma_reverse_all



    def _linear_beta_schedule(self, timesteps, start=0.0001, end=0.01):
        """
            生成线性的 beta 调度，返回一个包含 timesteps 个元素的张量。

            参数:
                - timesteps: 调度的时间步数
                - start: 调度的起始值，默认为 0.0001
                - end: 调度的结束值，默认为 0.02

            返回:
                一个包含 timesteps 个元素的张量，表示线性变化的 beta 调度。
            """
        # 使用 torch.linspace 在指定范围内生成 timesteps 个均匀间隔的数值
        return torch.linspace(start, end, timesteps)

    def _get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        """
            在考虑批处理维度的情况下，从传入的值列表 vals 中返回特定索引 t 对应的值。

            参数：
                - vals: 一个包含待选择值的列表
                - t: 要选择的索引
                - x_shape: 输入张量的形状

            返回：
                一个包含具体索引 t 对应的值的张量，考虑了批处理维度。
            """
        # 获取批处理大小
        batch_size = t.shape[0]

        # 使用 gather 方法根据索引 t 从值列表 vals 中提取对应的值
        out = vals.gather(-1, t.cpu())
        # 重新整形张量，保留批处理维度
        #这里的目的是创建一个包含 (1, 1, ..., 1) 的元组，其中的元素数量是 len(x_shape) - 1。这个元组描述了在除了批处理维度外的每个维度上都只有一个元素。
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        """
            输入一个图像和一个时间步长，返回其带有噪声的版本。

            参数：
                - x_0: 输入图像的张量
                - t: 时间步长
                - device: 指定设备，用于处理计算，默认为 "cpu"

            返回：
                一个包含噪声版本的图像张量和对应噪声的元组。
            """
        # 生成与输入张量相同形状的随机噪声512_512_57：[8,1,64,64,64],t:[8]
        noise = torch.randn_like(x_0)
        # 获取时间步长 t 对应的 sqrt_alphas_cumprod 和 sqrt_one_minus_alphas_cumprod
        sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        # 计算噪声版本，包括均值和方差，返回结果和对应的噪声
        x_noise = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

        # # 尝试2D网络，即size变为(8, 16, 64, 64)，即(b, spe, h, w)
        # x_noise = torch.squeeze(x_noise, dim=1)
        # noise = torch.squeeze(noise, dim=1)

        return x_noise, noise.to(device)



    def forward_diffusion_sample_GMM(self, x_0, t, device="cpu"):
        # GMM parameters
        weights = torch.tensor([0.5, 0.3, 0.2], device=device)
        means = torch.tensor([-0.5, 0.0, 0.5], device=device)
        stds = torch.tensor([0.5, 1.0, 0.7], device=device)

        # Create GMM
        mix = Categorical(weights)
        comp = Normal(means, stds)
        gmm = MixtureSameFamily(mix, comp)

        # Sample noise
        noise = gmm.sample(x_0.shape)

        # Rest of the diffusion process remains the same
        sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        x_noise = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
                  + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

        return x_noise, noise.to(device)

    def forward_diffusion_log_gauss(self,x_0, t, device="cpu"):
        # 生成与输入张量相同形状的随机噪声512_512_57：[8,1,64,64,64],t:[8]
        noise = torch.randn_like(x_0)
        data_abs = torch.abs(x_0)
        data_log = torch.log(data_abs + 1e-8)
        channels_to_negate = [6, 7, 8]
        data_log[:, :, channels_to_negate] *= (-1)
        # 获取时间步长 t 对应的 sqrt_alphas_cumprod 和 sqrt_one_minus_alphas_cumprod
        # sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        # sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
        #     self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        # )
        alpha_t = self._get_index_from_list(self.betas,t,x_0.shape)
        # mean + variance
        # 计算噪声版本，包括均值和方差，返回结果和对应的噪声
        return torch.exp(data_log.to(device) + alpha_t.to(device) * noise.to(device)),x_0.to(device)
        # return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        #        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

    def forward_diffusion_gamma_cumprod(self,x_0, t, device="cpu"):
        gamma_noise = torch.tensor(gamma.rvs(a=4, scale=1, size=(1, 64, 64)), dtype=torch.float).to(device)
        gamma_noise = gamma_noise / torch.mean(gamma_noise)
        gamma_cumprod = gamma_noise
        gamma_noise_result = gamma_noise
        j = 0
        for i in range(t.shape[0]):
            t_s = t[i].to(device)
            while j < t_s:
                gamma_noise = torch.tensor(gamma.rvs(a=4, scale=1, size=(1, 64, 64)), dtype=torch.float).to(device)
                gamma_noise = gamma_noise / torch.mean(gamma_noise)
                gamma_cumprod *= gamma_noise.to(device) * (2+0.01*j)
                gamma_cumprod = gamma_cumprod / torch.mean(gamma_cumprod)
                j += 1
            if i == 0:
                gamma_noise_result = gamma_cumprod.unsqueeze(0)
            else:
                gamma_noise_result = torch.concat((gamma_noise_result,gamma_cumprod.unsqueeze(0)),dim=0)
        gamma_noise = gamma_noise_result.unsqueeze(4).repeat(1, 1, 1, 1, 16).permute(0, 1, 4, 2, 3)
        return gamma_noise*x_0.to(device), gamma_noise.to(device)

    def forward_diffusion_SVD_gamma(self, x_0, t, device="cpu"):
        gamma_noise = torch.tensor(gamma.rvs(a=4, scale=1, size=(8, 1, 64, 64)), dtype=torch.float)
        gamma_noise = gamma_noise / torch.mean(gamma_noise)
        gamma_noise = gamma_noise.unsqueeze(4).repeat(1, 1, 1, 1, 16).permute(0, 1, 4, 2, 3)
        gamma_log = torch.log(gamma_noise + 1e-8)
        U, s, V = torch.linalg.svd(x_0)
        log_s = torch.log(s)#(8, 1, 16, 64)
        log_Sigma = torch.zeros_like(x_0)
        for i in range(log_s.shape[0]):
            for j in range(log_s.shape[1]):
                for k in range(log_s.shape[2]):
                    sub_log_s = log_s[i, j, k, :]
                    S = torch.diag(sub_log_s)
                    log_Sigma[i, j, k, :, :] = S[:, :]
        data_svd = U @ log_Sigma @ V
        sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        result_add = sqrt_alphas_cumprod_t.to(device) * data_svd.to(device) \
                     + sqrt_one_minus_alphas_cumprod_t.to(device) * gamma_log.to(device)

        result_exp = torch.exp(result_add)
        channels_to_negate = [6, 7, 8]
        result_exp[:, :, channels_to_negate] *= (-1)
        return result_exp, gamma_log.to(device)

    def forward_diffusion_gamma(self, x_0, t, device="cpu"):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        gamma_noise = torch.tensor(gamma.rvs(a=4, scale=1, size=(8,1,64,64)), dtype=torch.float)
        gamma_noise = gamma_noise / torch.mean(gamma_noise)
        gamma_noise = gamma_noise.unsqueeze(4).repeat(1, 1, 1, 1, 16).permute(0, 1, 4, 2, 3)
        gamma_log = torch.log(gamma_noise + 1e-8)
        data_abs = torch.abs(x_0)
        data_log = torch.log(data_abs + 1e-8)
        channels_to_negate = [6, 7, 8]
        data_log[:, :, channels_to_negate] *= (-1)
        sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # mean + variance
        result_add = sqrt_alphas_cumprod_t.to(device) * data_log.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * gamma_log.to(device)

        result_exp = torch.exp(result_add)
        # channels_to_negate = [6, 7, 8]
        # result_exp[:, :, channels_to_negate] *= (-1)
        return result_exp, gamma_log.to(device)#gamma_noise.to(device)

    def forward_diffusion_addmultiple(self, x_0, t, device="cpu"):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        #[8, 1, 16, 64, 64]
        gamma_noise = torch.tensor(gamma.rvs(a=5.50, scale=1, size=(8,1,64,64)), dtype=torch.float)
        gamma_noise = gamma_noise / torch.mean(gamma_noise)
        gamma_noise = gamma_noise.unsqueeze(4).repeat(1, 1, 1, 1, 16).permute(0, 1, 4, 2, 3)
        gamma_log = torch.log(gamma_noise + 1e-8)
        gauss_noise = torch.randn_like(x_0[:,:,3:,:,:])
        data_abs = torch.abs(x_0)
        data_log = torch.log(data_abs + 1e-8)
        # channels_to_negate = [6, 7, 8]
        # data_log[:, :, channels_to_negate] *= (-1)
        sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0[:,:,3:,:,:].shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0[:,:,3:,:,:].shape
        )

        # mean + variance
        # result_add = sqrt_alphas_cumprod_t.to(device) * data_log.to(device) \
        # + sqrt_one_minus_alphas_cumprod_t.to(device) * gamma_log.to(device)
        result_multiple = gamma_noise.to(device) * x_0.to(device) + x_0.to(device)
        result_multiple[:,:,3:,:,:] = sqrt_alphas_cumprod_t.to(device) * result_multiple[:,:,3:,:,:] \
                              + sqrt_one_minus_alphas_cumprod_t.to(device) * gauss_noise.to(device)

        result_exp = torch.exp(result_multiple)
        # channels_to_negate = [6, 7, 8]
        # result_exp[:, :, channels_to_negate] *= (-1)
        return result_exp, x_0.to(device)#gamma_noise.to(device)

    def forward_diffusion_gongshi(self, x_0, t, device="cpu"):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        #[8, 1, 16, 64, 64]
        gamma_noise = torch.tensor(gamma.rvs(a=7, scale=1, size=(1, 64, 64)), dtype=torch.float).to(device)
        gamma_noise = gamma_noise / torch.mean(gamma_noise)

        gamma_noise_15 = self.generate_gamma(gamma_noise)
        gamma_noise_t = self.generate_gamma_t_reverse(gamma_noise_15, t.to(device))#(0,16)

        gamma_cumprod = torch.ones(gamma_noise.shape).to(device)
        post_accumulation = torch.zeros_like(x_0[0, :, 3:, :, :])
        result_front = []
        result_post = []

        t_all = list(range(self.T))
        t_all = torch.tensor(t_all, dtype=torch.long)
        alpha_t = self._get_index_from_list(self.betas, t_all, x_0.shape)

        gauss_noise = torch.randn_like(x_0[0, :, 3:, :, :])

        gauss_item = alpha_t.to(device) * gauss_noise.to(device)

        for i in range(t.shape[0]):
            j = 0
            t_current = t[i]
            if t_current == 0:
                result_front.append(torch.ones_like(gamma_cumprod))
                result_post.append(torch.zeros_like(post_accumulation))
                continue
            while j < t_current:
                gamma_cumprod *= (gamma_noise_15[j].to(device) + 1) #累乘
                j += 1
            result_front.append(gamma_cumprod)
            for k in range(t_current, self.T):
                gamma_item = gamma_noise_t[k+1].unsqueeze(3).repeat(1, 1, 1, 13).permute(0, 3, 1, 2)
                post_accumulation += gauss_item[k] * gamma_item.to(device)
            result_post.append(post_accumulation)

        rf = torch.cat(result_front, dim=0)
        rf = rf.unsqueeze(1)
        rf = rf.unsqueeze(2).repeat(1, 1, 16, 1, 1)

        rp = torch.cat(result_post, dim=0)
        rp = rp.unsqueeze(4).permute(0, 4, 1, 2, 3)
        # [8, 1, 16, 64, 64]
        # result = rf * x_0 + rp
        result = rf * x_0
        result[:, :, 3:, :, :] = result[:, :, 3:, :, :] + rp
        noise = torch.tensor(gamma.rvs(a=4, scale=1, size=result.shape), dtype=torch.float).to(device) + torch.randn_like(result).to(device)
        # noise = rf
        # noise[:, :, 3:, :, :] = noise[:, :, 3:, :, :] + rp
        return result, noise#x_0
    def forward_diffusion_gongshi_invgamma(self, x_0, t, device="cpu"):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        #[8, 1, 16, 64, 64]
        gamma_noise = torch.tensor(invgamma.rvs(a=2, scale=1, size=(1, 64, 64)), dtype=torch.float).to(device)
        gamma_noise = gamma_noise / torch.mean(gamma_noise)

        gamma_noise_15 = self.generate_gamma(gamma_noise)
        gamma_noise_t = self.generate_gamma_t_reverse(gamma_noise_15, t.to(device))#(0,16)

        gamma_cumprod = torch.ones(gamma_noise.shape).to(device)
        post_accumulation = torch.zeros_like(x_0[0, :, 3:, :, :])
        result_front = []
        result_post = []

        t_all = list(range(self.T))
        t_all = torch.tensor(t_all, dtype=torch.long)
        alpha_t = self._get_index_from_list(self.betas, t_all, x_0.shape)

        gauss_noise = torch.randn_like(x_0[0, :, 3:, :, :])

        gauss_item = alpha_t.to(device) * gauss_noise.to(device)

        for i in range(t.shape[0]):
            j = 0
            t_current = t[i]
            if t_current == 0:
                result_front.append(torch.ones_like(gamma_cumprod))
                result_post.append(torch.zeros_like(post_accumulation))
                continue
            while j < t_current:
                gamma_cumprod *= (gamma_noise_15[j].to(device) + 1) #累乘
                j += 1
            result_front.append(gamma_cumprod)
            for k in range(t_current, self.T):
                gamma_item = gamma_noise_t[k+1].unsqueeze(3).repeat(1, 1, 1, 13).permute(0, 3, 1, 2)
                post_accumulation += gauss_item[k] * gamma_item.to(device)
            result_post.append(post_accumulation)

        rf = torch.cat(result_front, dim=0)
        rf = rf.unsqueeze(1)
        rf = rf.unsqueeze(2).repeat(1, 1, 16, 1, 1)

        rp = torch.cat(result_post, dim=0)
        rp = rp.unsqueeze(4).permute(0, 4, 1, 2, 3)
        # [8, 1, 16, 64, 64]
        # result = rf * x_0 + rp
        result = rf * x_0
        result[:, :, 3:, :, :] = result[:, :, 3:, :, :] + rp
        noise = torch.tensor(gamma.rvs(a=4, scale=1, size=result.shape), dtype=torch.float).to(device) + torch.randn_like(result).to(device)
        # noise = rf
        # noise[:, :, 3:, :, :] = noise[:, :, 3:, :, :] + rp
        return result, noise#x_0
    def forward_diffusion_fisher(self, x_0, t, device="cpu"):
        #x_0: [8, 1, 16, 64, 64]
        # L = 5  #代表了纹理分布的平滑程度,L 越大，生成的纹理图像越平滑，噪声越小
        # M = 2  #控制了纹理的变化范围和离散程度,M 越大，图像中的噪声强度越大。
        # noise_level_initial = 0.01
        #
        # # 生成 Fisher 分布的纹理样本
        # u = f.rvs(2 * L, 2 * M, size=(8, 1, 16, 64, 64))
        # u = u / torch.mean(u)  # 归一化
        b, c, spe, h, w = x_0.shape
        gamma_noise = torch.tensor(gamma.rvs(a=5.5, scale=1, size=(b, c, spe, h, w)), dtype=torch.float).to(device)
        gamma_noise = gamma_noise / torch.mean(gamma_noise)
        alpha_t = self._get_index_from_list(self.betas_cumsum, t, x_0.shape)
        # 当前噪声 u 的调整
        gamma_noise = 1 + alpha_t * (gamma_noise - 1)  # 调整噪声强度

        #尝试2D网络，即size变为(8, 16, 64, 64)，即(b, spe, h, w)
        # gamma_noise = torch.squeeze(gamma_noise, dim=1)
        # x_0 = torch.squeeze(x_0, dim=1)
        # gamma_noise = torch.squeeze(gamma_noise, dim=1)

        return gamma_noise * x_0, gamma_noise

    def forward_diffusion_fisher_invG(self, x_0, t, device="cpu"):
        #x_0: [8, 1, 16, 64, 64]
        # L = 5  #代表了纹理分布的平滑程度,L 越大，生成的纹理图像越平滑，噪声越小
        # M = 2  #控制了纹理的变化范围和离散程度,M 越大，图像中的噪声强度越大。
        # noise_level_initial = 0.01
        #
        # # 生成 Fisher 分布的纹理样本
        # u = f.rvs(2 * L, 2 * M, size=(8, 1, 16, 64, 64))
        # u = u / torch.mean(u)  # 归一化
        b, c, spe, h, w = x_0.shape
        gamma_noise = torch.tensor(invgamma.rvs(a=2, scale=1, size=(b, c, spe, h, w)), dtype=torch.float).to(device)
        gamma_noise = gamma_noise / torch.mean(gamma_noise)
        alpha_t = self._get_index_from_list(self.betas_cumsum, t, x_0.shape)
        # 当前噪声 u 的调整
        gamma_noise = 1 + alpha_t * (gamma_noise - 1)  # 调整噪声强度

        #尝试2D网络，即size变为(8, 16, 64, 64)，即(b, spe, h, w)
        # gamma_noise = torch.squeeze(gamma_noise, dim=1)
        # x_0 = torch.squeeze(x_0, dim=1)
        # gamma_noise = torch.squeeze(gamma_noise, dim=1)

        return gamma_noise * x_0, gamma_noise

    def get_loss(self, model, x_0, t):
        hyper = Hyper()
        x_noisy, noise = hyper(x_0, t, device)
        # x_noisy, noise = self.forward_diffusion_sample(x_0, t, device)
        # x_noisy, noise = self.forward_diffusion_gamma(x_0, t, device)
        # x_noisy, noise = self.forward_diffusion_log_gauss(x_0, t, device)
        #x_noisy, noise = self.forward_diffusion_addmultiple(x_0, t, device)
        # x_noisy, noise = self.forward_diffusion_gongshi(x_0, t, device)
        # x_noisy, noise = self.forward_diffusion_fisher(x_0, t, device)
        # x_noisy, noise = self.forward_diffusion_gongshi_invgamma(x_0, t, device)
        # x_noisy, noise = self.forward_diffusion_fisher_invG(x_0, t, device)
        # x_noisy, noise = self.forward_diffusion_sample_GMM(x_0, t, device)
        # x_noisy, noise = self.forward_diffusion_SVD_gamma(x_0, t, device)
        # x_noisy, noise = self.forward_diffusion_gamma_cumprod(x_0, t, device)
        high_res = torch.randn(8, 1, 64, 64)  # 高分辨率输入
        low_res = torch.randn(8, 1, 64 // 2, 64 // 2)  # 低分辨率输入
        noise_pred = model(x_noisy, t)
        # noise_pred = model(x_noisy, t, low_res=low_res)
        return F.l1_loss(noise, noise_pred), x_noisy, noise, noise_pred
        # MSE = nn.MSELoss()
        # return MSE(noise, noise_pred), x_noisy, noise, noise_pred
    #l1 loss Mean Absolute Error,计算预测值和目标值之间的绝对差值的平均值


    @torch.no_grad()
    def sample_timestep(self, x, t, model):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        
        x is xt, t is timestamp
        return x_{t-1}
        如果 t == 0，则返回模型均值；否则，返回模型均值加上噪声。
        """
        # 获取时间步长 t 对应的 betas、sqrt_one_minus_alphas_cumprod 和 sqrt_recip_alphas
        betas_t = self._get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        # 调用模型，计算模型均值
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        # 获取时间步长 t 对应的 alpha（t-1）
        posterior_variance_t = self._get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    @torch.no_grad()
    def reconstruct(self, model, xt=None, tempT=None, num = 5, from_noise=False, shape=None):
        '''
        分别从纯noise和xt，逐步恢复信息
        如果不给定xt 则自动使用随机造成
        给定xt同时需要给定tempT，表明该xt是来自多少步造成生成
        '''
        stepsize = int(tempT.cpu().numpy()[0] / num)
        index = []
        res = []
        # Sample noise
        if from_noise:
            img = torch.randn(shape, device=device)
        else:
            img = xt

        if tempT is None:
            tempT = self.T

        for i in range(0, tempT)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = self.sample_timestep(img, t, model)
            if i % stepsize == 0:
                index.append(i)
                res.append(img.detach().cpu())
        index.append(i)
        res.append(img.detach().cpu())
        return index, res

    @torch.no_grad()
    def reconstruct_v2(self, model, xt=None, tempT=None, use_index=[], from_noise=False, shape=None):
        '''
        分别从纯noise和xt，逐步恢复信息
        如果不给定xt 则自动使用随机造成
        给定xt同时需要给定tempT，表明该xt是来自多少步造成生成
        '''
        index = []
        res = []
        # Sample noise
        if from_noise:
            img = torch.randn(shape, device=device)
        else:
            img = xt

        if tempT is None:
            tempT = self.T

        for i in range(0, tempT)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = self.sample_timestep(img, t, model)
            if i in use_index:
                index.append(i)
                res.append(img.detach().cpu())
        index.append(i)
        res.append(img.detach().cpu())
        return index, res
                
                
    




