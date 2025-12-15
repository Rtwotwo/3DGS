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
Todo: 用于加载,管理,保存3D场景数据(包括相机,点云,训练/测试集等),
      并和高斯模型(GaussianModel)协同工作
Homepage: https://github.com/Rtwotwo/3DRepo
"""
import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos


class Scene:
      gaussians: GaussianModel
      def __init__(self, args:ModelParams, 
                   gaussians:GaussianModel, 
                   load_iteration=None,
                   shuffle=True,
                   resolution_scales=[1.0]):
            """初始化场景数据,加载/初始化高斯模型,处理相机参数
            为后续的 3D 重建或渲染任务做准备"""
            self.model_path = args.model_path
            self.loaded_iter = None
            self.gaussians = gaussians
            # 指定加载预训练模型的迭代步数,-1表示加载最新模型
            if load_iteration:
                  if load_iteration == -1: 
                        self.loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
                  else: self.loaded_iter = load_iteration
                  print(f'[INFO] 正在加载来自{self.loaded_iter}的模型...')
            # 识别并加载场景数据(Colmap/Blender数据集)
            self.train_cameras = {}
            self.test_cameras = {}
            if os.path.exists(os.path.join(args.source_path, "sparse")):
                  scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, 
                              args.images, args.depths, args.eval, args.train_test_exp)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                  scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, 
                              args.white_background, args.depths, args.eval)
            else: assert False, "[WARNING] 源数据集格式错误!"
            # 首次训练时的初始化
            if not self.loaded_iter: 
                  with open(scene_info.ply_path, 'rb') as src_file, \
                        open(os.path.join(args.model_path, "input.ply"), 'wb') as dest_file:
                        dest_file.write(src_file.read())
                  json_cams = []
                  camlist = []
                  if scene_info.test_cameras: 
                        camlist.extend(scene_info.test_cameras)
                  if scene_info.train_cameras: 
                        camlist.extend(scene_info.train_cameras)
                  for id, cam in enumerate(camlist):
                        json_cams.append(camera_to_JSON(id, cam))
                  with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                        json.dump(json_cams, file)
            # 多分辨率一致的随机打乱
            if shuffle:
                  random.shuffle(scene_info.train_cameras)
                  random.shuffle(scene_info.test_cameras)
            # 处理多分辨率相机参数
            self.cameras_extent = scene_info.nerf_normalization["radius"]
            for resolution_scale in resolution_scales:
                  print(f"[INFO] 正在加载Training相机参数...")
                  self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, 
                                          resolution_scale, args, scene_info.is_nerf_synthetic, False)
                  print(f"[INFO] 正在加载Testing相机参数...")
                  self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, 
                                        resolution_scale, args, scene_info.is_nerf_synthetic, True)
            # 初始化/加载高斯模型
            if self.loaded_iter:
                  self.gaussians.load_ply(os.path.join(self.model_path, 
                                                       "point_cloud",
                                                       "iteration_"+str(self.loaded_iter),
                                                       "point_cloud.ply"), args.train_test_exp)
            else: self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)
      def save(self, iteration):
            """在指定迭代步数时,保存高斯模型的点云数据和曝光参数"""
            point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            exposure_path = {image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
                             for image_name in self.gaussians.exposure_mapping}
            # 保存曝光参数为 JSON 文件
            with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
                  json.dump(exposure_path, f, indent=2)
      def getTrainCameras(self, scale=1.0):
            return self.train_cameras[scale]
      def getTestCameras(self, scale=1.0):
            return self.test_cameras[scale]    
