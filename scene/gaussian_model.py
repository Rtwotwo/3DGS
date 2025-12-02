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
Modifications by: Redal
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
    def construct_list_of_attributes(self):
        """属性名称列表,用于定义PLY文件中每个顶点包含的属性
        属性包括位置、颜色特征、不透明度、尺度和旋转等信息"""
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # 遍历所有DC/REST系数通道,为每个通道创建一个属性名
        for i in range(self._features_dc.shspae[1]*self._features_dc.shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        return l
    def save_ply(self, path):
        """保存ply文件"""
        mkdir_p(os.path.dirname(path))
        # 将基本属性存储为列表
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        # 构建结构化NumPy数组
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        # 使用plyfile库创建PLY元素,'vertex'表示点云
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    def replace_tensor_to_optimizer(self, tensor, name):
        """替换优化器optimizer中指定名称的参数组param_group的核心参数,
        并重置该参数对应的优化器状态,最终返回更新后的可优化参数字典"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == name:
                stored_state = self.optimizer.state.get(group['param'][0], None)
                stored_state['exp_avg'] = torch.zeros_like(tensor)
                stored_state['exp_avg_sq'] = torch.zeros_like(tensor)

                del self.optimizer.state[group['param'][0]]
                group['params'][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0] 
                # 收集替换后的可优化参数,返回字典形式
        return optimizable_tensors
    def reset_opacity(self):
        """神经网络参数重置代码核心功能是重置模型中opacity相关的可优化参数"""
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors['opacity']
    def load_ply(self, path, use_train_test_exp = False):
        """加载PLY点云文件并初始化模型可训练参数"""
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, 'r') as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name:torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"预训练的曝光度已加载...")
            else:
                print(f"在{exposure_file}中没有找到预训练的曝光度文件...")
                self.pretrained_exposures = None
                
        # PLY文件中存储了点云的每个点的属性,方法按固定字段名提取
        xyz = np.stack(np.asarray(plydata.elements[0]['x']),
                        np.asarray(plydata.elements[0]['y']),
                        np.asarray(plydata.elements[0]['z']))
        opacities = np.asarray(plydata.elements[0]['opacity'])[..., np.newaxis]
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]['f_dc_0'])
        features_dc[:, 1, 0] = np.asarray(plydata.elemnets[0]['f_dc_1'])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]['f_dc_2'])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('f_rest_')]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree+1) ** 2 -3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # 将(P，F*SH_coeffs)重塑为(P,F,除DC外的SH_coeffs)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree +1)**2 -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x:int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('rot')]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros(xyz.shape[0], len(rot_names))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # 注意features_dc和features_rest和形状需要进行转置
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device='cuda').requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device='cuda').requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device='cuda').requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device='cuda').requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
    def _prune_optimizer(self, mask):
        """根据掩码mask裁剪模型参数及优化器的状态信息,
        保留需要的参数并维持优化器训练状态的一致性,
        最终返回剪枝后的可优化参数字典"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 安全地从字典中获取指定键key对应的值value,若键不存在则返回预设的默认值None
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]

                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter((group['params'][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0]
            else:
                # 若存储的状态None未经过训练,直接用mask裁剪参数
                group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors
    def prune_points(self, mask):
        """剪枝筛选3D点云/神经辐射场NeRF类中无效点,根据掩码保留
        有效点,丢弃无效点,同步更新所有关联的张量和状态变量"""
        # 无效点掩码mask取反
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        # 模型核心状态的更新
        self._xyz = optimizable_tensors['xyz']
        self._features_dc = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacity = optimizable_tensors['opacity']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]
    def cat_tensors_to_optimizer(self, tensors_dict):
        """给优化器管理的参数张量(追加扩展张量),同步更新优化器的状态(如动量,二阶矩)
        确保扩展后的参数仍能正常参与训练优化"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group['params'])==1, f"group的尺寸{len(group['params'])}不为1!"
            extension_tensor = tensors_dict[group['name']]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=0)
                stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name'][0]] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group['name'][0]] = group['params'][0]
        return optimizable_tensors
    def densification_postfix(self, 
                              new_xyz,
                              new_features_dc,
                              new_features_rest,
                              new_opacities,
                              new_scaling,
                              new_rotation,
                              new_tmp_radii):
        """致密化后的状态更新,通过cat_tensors_to_optimizer
        整合到原模型中,并重置部分辅助变量"""
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        # 更新优化器的状态参数
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors['xyz']
        self._features_dc = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacity = optimizable_tensors['opacity']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']
        # 处理临时张量self.tmp_radii
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')
    # 在训练过程中动态地克隆clone,分裂split,剪枝prune高斯点,以自适应地匹配场景复杂度
    # 主要涉及针对张量更新的操作,状态对齐以及更新操作,实现对高斯点的自适应训练处理
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """对 “梯度满足阈值 + 尺度较大” 的点进行分裂(生成N个新点),
        实现区域细化(适合将大尺度点拆分为多个小尺度点,填充细节)"""
        n_init_points = self.get_xyz.shape[0]
        # 提取的点数需要满足梯度更新的条件
        padded_grad = torch.zeros((n_init_points), device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # 筛选掩码,满足两个条件的点会被选中:梯度值 ≥ 设定阈值;
        # 点的最大缩放尺度 > 场景范围的一定比例,避免在密集区域过度分裂小点
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                            torch.max(self.get_scaling, dim=1).values > \
                                self.percent_dense*scene_extent)
        
        # 据尺度生成正态分布的随机样本以及构造旋转矩阵
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        # 批量复制选中的元素并计算新的参量
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
                  self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        # 将新生成的点添加到模型中optimizer.state
        self.densification_postfix(new_xyz, new_features_dc,
                                   new_features_rest, new_opacity,
                                   new_scaling, new_rotation, 
                                   new_tmp_radii)
        # 创建剪枝过滤器,包含原始选中点和新增点的占位符
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N*selected_pts_mask.sum(), device='cuda', dtype=bool)))
        # 移除原始选中的点,原来的点被新的点替代
        self.prune_points(prune_filter)
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """对 “梯度满足阈值 + 尺度较小” 的点直接复制,
        实现简单增密(适合细节区域的精细采样)"""
        # 类似densify_and_split方法的更新
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, 
                            torch.max(self.get_scaling, dim=1).values <= \
                            self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc,
                                   new_features_rest, new_opacities,
                                   new_scaling, new_rotation, 
                                   new_tmp_radii)
    def densify_and_prune(self, max_grad, 
                          min_opacity, 
                          extent, 
                          max_screen_size, 
                          radii):
        """整合增密与剪枝,集中调用densify_and_split和densify_and_prune
        统一调用增密逻辑,再执行剪枝,完成 “增密有效点,删除无效点” 的完整流程"""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        # 1.执行增密算法
        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        # 2.执行剪枝算法
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            # 标记在世界空间中过大的点
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1*extent
        self.prune_points(prune_mask)
        # 处理剩余变量
        tmp_radii = self.tmp_radii
        self.tmp_radii = None
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """记录每个采样点的梯度强度"""
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    