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


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(self, scaling_modifier, rotation):
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
        # 
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
    def get_opacity(self,):
        """获取高斯不透明度参数,应用不透明度激活函数"""
        return self.opacity_activation(self._opacity)
