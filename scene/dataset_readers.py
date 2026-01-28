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


def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params,
                      images_folder, depths_folder, test_cam_names_list):
    """读取colmap的相机参数: 返回相机参数的各类设置参数类的集合"""                  
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # 打印详细的输出信息
        sys.stdout.write(f'[INFO] Reading camera {idx+1}/{len(cam_extrinsics)}')
        sys.stdout.flush()
        # 获取相机的参数信息设置
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        # 获取相机内参的参数类型进行判断
        if intr.model=='SIMPLE_PINHOLE':
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=='PINHOLE':
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else: assert False, f'[ERROR] 目前仅能支持PINHOLE和SIMPLE_PINHOLE内参类型'
        # 获取相机参数信息information
        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try: depth_params = depths_params[extr.name[:-n_remove]]
            except: print(f'[INFO] Keys:{key}, Not Found in Depth Params')
        # 获取图像的信息
        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f'{extr.name[:-n_remove]}.png') if depths_folder != "" else ""
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image_path=image_path,
                              image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    """从PLY文件中读取点云数据并返回基础点云对象
    BasicPointCloud:包含点位置、颜色和法线的基础点云对象"""
    # 获取PLY的文件的数据信息
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    # 读取顶点坐标/颜色/法向量并转换numpy数据格式
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']])
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']])
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    """"""
