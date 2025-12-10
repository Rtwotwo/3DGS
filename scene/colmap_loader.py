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
Date: 2025-11-22
Todo: 供一组工具函数和数据结构,用于读取、解析和处理COLMAP生成的模型文件和相关数据,  
      包括相机参数、图像姿态、三维点云等信息
Homepage: https://github.com/Rtwotwo/3DRepo
"""
import numpy as np
import collections
import struct


CameraModel = collections.namedtuple("CameraModel", ["mdoel_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
# 包含多个CameraModel实例,每个实例表示一种相机模型及其对应的参数数量
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    """这段代码的作用是将一个四元数转换为对应的旋转矩阵
    return R: 3x3的旋转矩阵"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    """旋转矩阵R转换为对应的四元数
    return qvec: [qw, qx, qy, qz]"""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    # 对称矩阵K便于提取四元数
    k = np.array([Rxx - Ryy - Rzz, 0, 0, 0],
                 [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                 [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                 [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]) / 3.0
    # 计算矩阵K的特征值和特征向量
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax[eigvals]]
    # 确保四元数的正号约定
    if qvec[0] < 0: qvec *= -1
    return qvec


class Image(BaseImage):
    """定义基础类实现四元数到旋转矩阵的转换"""
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

    
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """从二进制文件中读取并解包接下来的字节
    :param fid: 文件对象，应以二进制模式打开
    :param num_bytes: 要读取的字节数，必须是 {2, 4, 8} 的组合之和，例如 2, 6, 16, 30 等
    :param format_char_sequence: 格式字符序列，取值范围为 {c, e, f, d, h, H, i, I, l, L, q, Q}
    :param endian_character: 表示字节序的字符，可选值为 {@, =, <, >, !}
    :return: 读取并解包后的值，返回一个元组"""
    data = fid.read(num_bytes)
    # 使用struct.unpack函数将二进制数据转换为可读的数值
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_text(path_to_model_file):
    """see: src/base/reconstruction.cc
    void Reconstruction::ReadPoints3DBinary(const std::string& path)
    void Reconstruction::WritePoints3DBinary(const std::string& path)"""
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        # 定义点云位置、颜色、误差等信息的列表
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))
        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 
                    num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])

            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii"*track_length)
            
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
        return xyzs, rgbs, errors
                        

def read_intrinsics_text(path):
    """"""



