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
Date: 2025-11-02
Todo: Build dataset reader for gaussian_model.py to create gaussian model
Homepage: https://github.com/Rtwotwo/3DRepo
"""
import os
import json
import torch 
import numpy as np
from torch import nn
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
try: from diff_gaussian_rasterization import SparseGaussianAdam
except: pass

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """过缩放和旋转参数构建3D高斯分布的协方差矩阵"""
            L = build_scaling_rotation(scaling_modifier, rotation)
            actual_covariance = L @ L.T
            symm = strip_symmetric(actual_covariance)
            return symm
        # 缩放-exp缩放激活函数,log反向激活函数
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation 
        # 不透明度-sigmoid将实数映射到(0,1)区间
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        # 旋转-归一化向量作为旋转四元数
        self.rotation_activation = torch.nn.functional.normalize
    def __init__(self, sh_degree, optimizer_type='default'):
        """初始化方法和相关设置函数,主要作用是初始化3D高斯模型的各种参数和功能函数"""
        self.active_sh_degree = 0 # 当前使用的球谐函数度数
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree # 最大球谐函数度数
        # 模型参数(初始为空张量)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0) # 直流颜色特征
        self._features_rest = torch.empty(0) # 高阶球谐颜色特征
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0) # 2D最大半径,用于渲染
        # 梯度统计相关
        self.xyz_grdient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        # 优化相关
        self.optimizer = None
        self.percent_dense = 0 # 密集百分比阈值
        self.spacial_lr_scale = 0 # 空间学习率缩放因子
        self.setup_functions()
    def capture(self):
        """法用于捕获并返回当前GaussianModel实例的所有重要状态信息
        通常用于保存模型检查点或在训练过程中传递模型状态"""
        return {
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D, 
            self.xyz_grdient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale}
    def restore(self, model_args, training_args):
        """restore方法是capture方法的逆操作,用于从保存的状态恢复GaussianModel的完整状态
        model_args从capture方法返回的状态元组;training_args训练参数配置"""
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling, 
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spacial_lr_scale) = model_args
        # 赋值处理
        self.xyz_grdient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    @property
    def get_scaling(self,):
        """获取高斯缩放参数,应用缩放激活函数"""
        return self.scaling_activation(self._scaling)
    @property
    def get_rotation(self,):
        """获取高斯旋转参数,应用旋转激活函数"""
        return self.rotation_activation(self._rotation)
    @property
    def get_xyz(self,):
        """获取高斯坐标参数"""
        return self._xyz
    @property
    def get_features(self,):
        """获取高斯特征参数,应用颜色激活函数"""
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    @property
    def get_features_dc(self,):
        """获取高斯DC特征参数,应用颜色激活函数"""
        return self._features_dc
    @property
    def get_features_rest(self,):
        """获取高斯REST特征参数,应用颜色激活函数"""
        return self._features_rest
    @property
    def get_opacity(self,):
        """获取高斯不透明度参数,应用不透明度激活函数"""
        return self.opacity_activation(self._opacity)
    @property
    def get_exposure(self,):
        """获取高斯曝光参数,应用曝光激活函数"""
        return self._exposure
    def get_exposure_from_name(self, image_name):
        """获取与特定图像名称关联的曝光(exposure)参数矩阵"""
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else: return self.pretrained_exposures[image_name]
    def get_covariance(self, scaling_modifier = 1):
        """计算高斯分布的协方差矩阵
        scaling_modifier缩放参数缩放因子,默认为1"""
        self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    def oneupSHdegree(self,):
        """升级球谐函数的度数"""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    def create_from_pcd(self, pcd:BasicPointCloud, 
                        cam_infos:int,
                        spatial_lr_scale:float):
        """从点云创建高斯模型,点云形状[N, ...]N代表点的数量
        pcd:BasicPointCloud点云对象,cam_infos:相机信息
        spatial_lr_scale:空间学习率缩放因子"""
        self.spacial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # features的形状为(N, 3, (SH_DEGREE+1)²)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree+1)**2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, :3, 1:] = 0.0
        print(f'初始化的点云数量为{fused_point_cloud.shape[0]}')
        # 计算初始尺度的参数,使用K近邻算法计算每个点到邻居的距离,以此估算初始尺度(N, 3)
        dist2 = torch.clmap_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        # 将所有点的初始旋转设置为单位四元数
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device='cuda')
        rots[:, 0] = 1
        # 将所有点的初始不透明度设置为0.1
        opacities = self.inverse_opacity_activation(0.1*torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device='cuda'))
        
        # 将所有参数设置为可学习参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1,2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1,2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')
        self.exposure_mapping = {cam_info.image_name:idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposures = torch.eye(3, 4, device='cuda')[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposures.requires_grad_(True))
    def training_setup(self, training_args):
        """训练设置,包括优化器,学习率调度,数据集等"""
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        # 参数分组+优化器配置
        l = [{'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, 'name':'xyz'}, 
            {'params': [self._features_dc], 'lr':training_args.feature_lr, 'name':'f_dc'},
            {'params': [self._features_rest], 'lr':training_args.feature_lr, 'name':'f_rest'},
            {'params': [self._opacity], 'lr':training_args.opacity_lr, 'name':'opacity'},
            {'params': [self._scaling], 'lr':training_args.scaling_lr, 'name':'scaling'},
            {'params': [self._rotation], 'lr':training_args.rotation_lr, 'name':'rotation'},]
        # 优化器选择
        if self.optimizer_type == 'default':
            self.optimizer = torch.optim.Adam(1, lr=0.0, eps=1e-15)
        elif self.optimizer_type == 'sparse_adam':
            try: self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except: # 启用稀疏adam需要光栅化器的一个特殊版本
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.exposure_optimizer = torch.optim.Adam([self._exposure])
        # 学习率调度器(指数衰减)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spacial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spacial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                         lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                         lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                         max_steps=training_args.iterations)
    def update_learning_rate(self, iteration):
        """学习率随着每步更新"""
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)
        # 更新主优化器中"xyz"参数的学习率
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            