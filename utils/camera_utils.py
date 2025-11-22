"""
Author: Redal
Date: 2025-11-18
Todo: Load and manage camera data
Homepage: https://github.com/Rtwotwo/3DRepo
"""
from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
WARNED = False


def loadCam(args, id, cam_info, resolution_scale, 
            is_nerf_synthetic, is_test_dataset):
    """用于加载相机数据的函数,核心作用是读取图像/深度图,
    根据配置缩放图像分辨率并封装成Camera类实例返回
    id当前相机的唯一标识序号; args命令行参数
    cam_info相机元信息对象包含图像路径,位姿,内参等
    resolution_scale分辨率额外缩放系数"""
    image = Image.open(cam_info.image_path)
    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic: invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else: invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)
        except FileNotFoundError: print(f'Error: The depth file at path \'{cam_info.depth_path}\' was not found.'); raise
        except IOError: print(f'Error: Unable to open the image file \'{cam_info.depth_path}\'. It may be corrupted or an unsupported format.'); raise
        except Exception as e: print(f'An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}'); raise
    else: invdepthmap = None
    # 计算目标分辨率,包括2种图像缩放逻辑:固定比例缩放,自适应/指定比例缩放
    # 注意缩放比例计算出的resolution是Tuple元组类型
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        # 固定比例缩放
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(orig_h / (resolution_scale * args.resolution))
    else:
        # 自适应/指定比例缩放
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] 遇到相当大的输入图像(宽度>1.6K像素),正在将其缩放到1.6K\n",
                          "如果不希望出现这种情况,请明确将'--resolution/-r'指定为 1")
                    WARNED = True
                global_down = orig_w / 1600
            else: global_down = 1
        else: global_down = orig_w / args.resolution
        # 计算缩放比例
        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap, 
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)


def cameraList_from_camInfos(cam_infos, resolution_scale, args, 
                             is_nerf_synthetic, is_test_dataset):
    """用于从相机信息列表构建相机实例列表,调用loadCam函数
    批量处理相机配置并生成可直接使用的相机对象"""
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, 
                                   is_nerf_synthetic, is_test_dataset))
    return camera_list


def camera_to_JSON(id, camera:Camera):
    """将相机Camera对象的关键参数转换为可序列化的JSON格式字典
    输入相机ID和Camera类实例,输出包含相机ID,图像信息,内外参数的字典"""
    # 构建相机外参矩阵Rt相机→世界的变换
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0
    # 计算世界→相机的变换矩阵W2C及相机位姿
    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_camera_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'image_name':camera.image_name,
        'width':camera.width,
        'height':camera.height,
        'position':pos.tolist(),
        'rotation':serializable_camera_2d,
        'fy':fov2focal(camera.FovY, camera.height),
        'fx':fov2focal(camera.FovX, camera.width)}
    return camera_entry