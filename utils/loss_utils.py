#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""
Author: Redal
Date: 2025-11-18
Todo: Image quality assessment and loss function calculation,
      containing SSIM, L1, L2, and FusedSSIM metrics
Homepage: https://github.com/Rtwotwo/3DRepo
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim
    from diff_gaussian_rasterization._C import fusedssim_backward
except ImportError: pass


# 定义SSIM计算中的稳定常数
C1 = 0.01 ** 2
C2 = 0.03 ** 2


# 定义高效计算SSIM方法
class FusedSSIMMap(torch.autograd.Function):
    """基于PyTorch的自定义自动求导函数FusedSSIMMap,
    用于高效计算SSIM映射图并支持对输入图像img1的自动求导"""
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        # 接收输入C1,C2(SSIM计算中的稳定常数)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map
    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        # 调用fusedssim_backward计算输入图像的梯度
        # 仅对img1有梯度,img2梯度为None
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, grad
    

def fast_ssim(img1, img2):
    """使用FusedSSIMMap高效计算SSIM"""
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
    
        
def l1_loss(network_output, gt):
    """L1损失倾向于生成更稳健的结果"""
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    """L2损失则更关注减小大误差"""
    return ((network_output - gt) ** 2).mean()


# 创建基本SSIM计算方法
def gaussian(window_size, sigma):
    """生成一维高斯核Gaussian kernel的函数
    window_size窗口大小,即高斯核的维度
    sigma高斯分布的标准差,控制核的宽窄"""
    gauss = torch.Tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """生成中心权重高,边缘权重低,能实现对图像的平滑
    该窗口可适配不同通道数的平滑图像"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # _1D_window本身列向量,增加两个维度(对应批量和通道维度)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 用expand扩展为适配channel个通道的窗口
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size)).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """从局部区域的亮度(均值),对比度(方差)和结构(协方差)三个维度
    衡量图像相似度,最终输出越接近1表示图像越相似"""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    # 计算方差与图像协方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 **2
    C2 = 0.03 **2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average: return ssim_map.mean()
    # 否则返回每个样本的均值
    else: return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    """计算ssim"""
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    # 获取img1的设备,保证在同一设备
    if img1.is_cuda: window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)