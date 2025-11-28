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
Todo: Learn how to build scene folder from GRAPHDECO 
      research group https://team.inria.fr/graphdeco
Homepage: https://github.com/Rtwotwo/3DRepo
"""
import torch
import math
import numpy as np
from typing import NamedTuple


class BasicPointCloud(NamedTuple):
      """定义继承NamedTuple的不可变的类BasicPointCloud属性"""
      points: np.array
      colors: np.array
      normals: np.array


def geom_transform_points(points:np.array, 
                          transf_matrix:np.array
                          ) -> np.array:
      """齐次坐标的矩阵乘法,实现了三维点集的统一几何变换
      transf_matrix的形状通常是4x4的形状"""
      P, _ = points.shape
      ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
      points_hom = torch.cat([points, ones], dim=1) # points_hom形状是[P, 4]=[num_points, [x, y, z, 1]]
      points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))
      # 除以第四个分量w才能还原为三维坐标,归一化矩阵
      denom = points_out[..., 3:] + 1e-7
      return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
      """构建从世界坐标系到视图(相机)坐标系的变换矩阵
      返回的Rt是4x4的齐次变换矩阵(用于3D坐标的变换)"""
      Rt = np.zeros((4,4))
      Rt[:3, :3] = R.transpose()
      Rt[:3, 3] = t
      Rt[3, 3] = 1.0
      return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
      """在getWorld2View的基础上,增加了对相机位置的平移translate和缩放scale调整
      再构建世界到视图的变换矩阵,返回的Rt依旧是4x4的变换矩阵"""
      Rt = np.zeros((4,4))
      Rt[:3, :3] = R.transpose()
      Rt[:3, 3] = t
      Rt[3, 3] = 1.0
      # 相机到世界的变换矩阵,即是逆矩阵
      # 先平移相机, 再缩放其位置
      C2W = np.linalg.inv(Rt)
      cam_center = C2W[:3, 3]
      cam_center = (cam_center + translate) * scale
      C2W[:3, 3] = cam_center
      # 将平移和缩放之后的矩阵变换回世界到相机变换矩阵
      Rt = np.linalg.inv(C2W)
      return np.float32(Rt)
      

def getProjectionMatrix(znear, zfar, fovX, fovY):
      """用于将3D空间中的点投影到2D图像平面,常见于计算机图形学和3D渲染中
      znear/zfar: 视锥体的near和far平面的距离,限制可见深度范围
      fovX/fovY: 视野角度,表示相机的视场大小,定相机的视野范围"""
      tanHalfFovY = math.tan((fovY / 2))
      tanHalffovX = math.tan((fovX / 2))
      # 计算视锥体的上下左右的距离
      top = tanHalfFovY * znear
      bottom = -top
      right = tanHalffovX * znear
      left = -right
      # 构造投影矩阵P,z_sign=1.0表示采用右手坐标系
      P = torch.zeros(4, 4)
      z_sign = 1.0
      # 填充投影矩阵P的元素,填充规律按照基于相机空间到裁剪空间的映射,
      # 再通过透视除法转换为标准化设备坐标(NDC)
      P[0, 0] = 2.0 * znear / (right - left)
      P[1, 1] = 2.0 * znear / (top - bottom)
      P[0, 2] = (right + left) / (right - left)
      P[1, 2] = (top + bottom) / (top - bottom)
      P[3, 2] = z_sign
      P[2, 2] = z_sign * zfar / (zfar - znear)
      P[2, 3] = -(zfar * znear) / (zfar - znear)
      return P


def fov2focal(fov, pixels):
      """输入视场角fov,单位通常为弧度和像素数量
      如传感器宽度对应的像素数,计算等效焦距"""
      return pixels / (2.0 * math.tan(fov / 2.0))


def focal2fov(focal, pixels):
      """输入焦距和像素数量,反推视场角"""
      return 2.0 * math.atan(pixels / (2 * focal))