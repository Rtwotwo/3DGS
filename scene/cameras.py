"""
Author: Redal
Date: 2025-11-22
Todo: 
Homepage: https://github.com/Rtwotwo/3DRepo
"""
import cv2
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch


class Camera(nn.Module):
    """基于PyTorch的相机参数与图像数据封装类,核心用于3D视觉任务
    (如神经辐射场 NeRF、多视图重建等),负责整合相机内参、外参、图像数据、
    深度信息，并提供投影变换矩阵等关键计算结果"""
    def __init__(self, resolution, colmap_id, R, T, FoVx, Fovy,
                 depth_params, image, invdepthmap, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 data_device='cuda', train_test_exp=False, 
                 is_test_dataset=False, is_test_view=False):
        super(Camera, self).__init__()
        # 定义相机核心初始化参数
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = Fovy
        self.image_name = image_name
        # 判断数据所处设备位置
        try: self.data_device = torch.device(data_device)
        except Exception as e:
            print(f'[ Warning ] the Current data device initialized failed.',
                  f'The data device is not supported and occurred an error: {e}')
            self.data_device = torch.device('cpu')
        # 处理输入的图像数据-提取或生成alpha通道数据
        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...] # 排除例如alpha等的透明通道
        self.alpha_mask = None
        if resized_image_rgb.shape[0]==4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))
        # 判断实验train/test模式-用于训练/测试数据分割
        # 遮蔽图像左/右部分-测试模型是否能从剩余图像中恢复完整场景
        if train_test_exp and is_test_view:
            if is_test_view: self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else: self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0
        # clamp将原始图像像素值归一化在[0, 1.0]
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.originla_image.shape[1]
        # 处理逆深度图invdepthmap的加载,预处理与可靠性校验
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap<0] = 0 # 生成逆深度图
            self.depth_reliable = True
            if depth_params is not None:
                # 校验缩放系数scale是否在合理范围0.2~5倍中位数缩放系数med_scale
                if depth_params['scale']<0.2*depth_params['med_scale'] or \
                    depth_params['scale']>5*depth_params['med_scale']:
                    self.depth_reliable = False
                    self.depth_mask = None
                # 对逆深度图进行线性变换-缩放+偏移
                if depth_params['scale'] > 0:
                    self.invdepthmap = self.invdepthmap*depth_params['scale'] + depth_params['offset']
            # invdepthmap类型np.narray[H,W,C]->torch.Tensor[C,H,W]
            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)
        # 计算相机-世界转换矩阵
        self.znear = 0.01
        self.zfar = 100.0
        self.trans = trans
        self.scale = scale
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)) ).squeeze(0) # 批量乘法bmm
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    """PyTorch 环境下的迷你相机类MiniCam,
    核心用于存储相机参数并计算相机在3D世界中的位置"""
    def __init__(self, width, height, fovy, fovx, znear, zfar, 
                 world_view_transform, full_proj_transform):
        self.image_height = height
        self.image_width = width
        self.fovy = fovy
        self.fovx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]