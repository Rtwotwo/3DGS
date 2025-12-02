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
            """"""
