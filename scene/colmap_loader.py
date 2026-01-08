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


def read_points3D_text(path):
    """see: src/base/reconstruction.cc
    void Reconstruction::ReadPoints3DText(const std::string& path)
    void Reconstruction::WritePoints3DText(const std::string& path)"""
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    # 初步读取points3D.bin文件的num_points信息,确定点的数量
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line: break
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                num_points += 1
    # 创建点云的xyz，rgb，误差参数矩阵
    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, 'r') as fid:
        while True:
            line = fid.readline()
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1
    return xyzs, rgbs, errors


def read_points3D_binary(path_to_model_file):
    """see: src/base/reconstruction.cc标准读取points3D文件的方法
    void Reconstruction::ReadPoints3DBinary(const std::string& path)
    void Reconstruction::WritePoints3DBinary(const std::string& path)"""
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        # 定义点云位置、颜色、误差等信息的列表
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))
        for p_id in range(num_points):
            # Q点的ID,ddd点的三维坐标,BBB点的颜色, d点误差
            binary_point_line_properties = read_next_bytes(fid, 
                    num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            # 读取跟踪信息:长度(无符号长整型),跟踪的元素-格式为ii的重复序列,表示图像ID和2D点索引
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
        return xyzs, rgbs, errors
                        

def read_intrinsics_text(path):
    """来自https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    读取相机的内参信息,并将其解析为一个字典结构,每个相机的内参信息包括相机ID、模型类型、分辨率以及相机参数"""
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line: break
            line = line.strip()
            if len(line)>0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model=='PINHOLE', f'[INFO] 当加载器适配其他格式的输入,剩下的支持PHONE格式'
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_intrisics_binary(path_to_model_file):
    """see: src/base/reconstruction.cc
    void Reconstruction::ReadImagesBinary(const std::string& path)
    void Reconstruction::WriteImagesBinary(const std::string& path)"""
    images = {}
    with open(path_to_model_file, 'rb') as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5]) 
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode('utf-8')
                current_char = read_next_bytes(fid, 1, 'c')[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D, format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_extrinsics_binary(path_to_model_file):
    """see: src/base/reconstruction.cc
    void Reconstruction::ReadImagesBinary(const std::string& path)
    void Reconstruction::WriteImagesBinary(const std::string& path)"""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_extrinsics_text(path):
    """来自https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    读取相机的外参信息,并将其解析为一个字典结构,每个相机的外参信息包括相机ID、旋转矩阵、平移向量"""
    images = {}
    with open(path, 'r') as fid:
        while True:
            line = fid.readline()
            if not line: break
            line = line.strip()
            # 读取相关的相机外参的配置
            if len(line) > 0 and line[0] != '#':
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple[float, ...](map[float](float, elems[1:5])))
                tvec = np.array(tuple[float, ...](map[float](float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                # 处理位置、点云数据信息
                elems = fid.readline().split()
                xys = np.column_stack([tuple[float,...](map[float](float, elems[0::3])),
                                       tuple[float,...](map[float](float, elems[1::3]))])
                points3D_ids = np.array(tuple[int, ...](map[int](int, elems[2::3])))
                images[image_id] = Image(
                    id = image_id, qvec = qvec, tvec = tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=points3D_ids)
    return images


def read_colmap_bin_array(path):
    """从COLMAP二进制文件中读取数组数据https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py
    path: 给定的二进制的文件路径来进行读取数据; 返回np.ndarray格式的数据关于点云的位置、颜色、深度等信息"""
    with open(path, 'rb') as fid:
        width, height, channels = np.genfromtxt(fid, delimiter='&', max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3: break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order='F')
    return np.transpose(array, (1, 0, 2)).squeeze()


