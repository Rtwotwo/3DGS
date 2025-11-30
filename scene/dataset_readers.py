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
Todo: 3D Gaussian Splatting项目中用于加载和预处理3D场景数据的核心模块,
      统一读取不同格式(COLMAP/NeRF Synthetic Blender)的输入数据,构建标
      准化的场景信息结构SceneInfo,供后续训练和渲染使用
Homepage: https://github.com/Rtwotwo/3DRepo
"""
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import copy
import json
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


class CameraInfo(NamedTuple):
    """用于集中管理与相机和图像相关的各类参数"""
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    """用于集中管理与场景相关的各类参数"""
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    """作用是将多视角相机的位置统一归一化到合适的坐标系"""
    def get_center_and_diag(cam_centers):
        """全局新中心(3维向量),相机分布的最大对角线长度"""
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        # 计算所有相机中心的平均位置
        center = avg_cam_center
        # 计算每个相机中心到新中心的距离并且取最大值作为对角线长度
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    # 提取相机在世界坐标系中的3D中心坐标
    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    # 计算归一化参数包括归一化半径radius和归一化平移translate
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = - center
    return {'translate': translate, 'radius': radius}


def readColmapCameras():
    """读取colmap的相机参数"""