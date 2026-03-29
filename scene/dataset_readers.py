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
from typing import Any, NamedTuple
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
    depth_params: Any
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool


class SceneInfo(NamedTuple):
    """用于集中管理与场景相关的各类参数"""
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


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
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
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


def storePly(path, xyz, rgb):
    """"""
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, rgb, normals), axis=1)
    elements[:] = list[tuple](map[tuple](tuple, attributes))

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    """读取COLMAP场景信息,包括相机内外参、深度参数、点云数据等,并构建场景信息对象
    path (str): 数据集路径
    images (str): 图像文件夹名称,如果为None则使用默认的"images"
    depths (str): 深度图文件夹名称，如果为空字符串则不使用深度信息
    eval (bool): 是否进行评估模式，决定是否划分测试集
    train_test_exp (bool): 训练测试实验标志
    llffhold (int, optional): LLFF数据集的采样间隔,默认为8
    SceneInfo: 包含点云、训练相机、测试相机、NeRF归一化参数等信息的场景对象"""
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## 如果没有深度参数文件但有深度文件 ->抛出异常
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF的'transform_matrix'是从相机到世界坐标系的变换
            c2w = np.array(frame["transform_matrix"])
            # 将OpenGL/Blender的相机坐标系轴 (Y向上，Z向后) 转换为 COLMAP (Y向下，Z向前)
            c2w[:3, 1:3] *= -1

            # 获得世界坐标系到相机的变换并设置 R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # 由于CUDA代码中的'glm'原因，这里的R被转置存储了
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # 因为数据集没有colmap数据，我们从随机点生成点云
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # 我们在合成Blender场景内生成随机点
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
