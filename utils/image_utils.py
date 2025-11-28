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
Date: 2025-11-19
Todo: Compute mse and psnr between two images
Homepage: https://github.com/Rtwotwo/3DRepo
"""
import torch


def mse(img1, img2):
    # 计算平均平方误差MSE,其中view操作[N,C,H,W]->[N,C*H*W]
    return (((img1 - img2)**2).view(img1.shape[0], -1)).mean(dim=1, keepdim=True)


def psnr(img1, img2):
    """计算PSNR,其中输入的图象是归一化后的图像"""
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))