import numpy as np
import cv2
import configparser
import os

rgb_img = cv2.imread('./1.png')

def image_crop_ellipse(rgb_img, center=None, axes=None):
    """
    在原始 RGB 图像上裁切椭圆区域，其他区域置为黑色
    center: (x, y) 椭圆中心，默认图像中心
    axes: (a, b) 椭圆半轴长度，默认覆盖整个图像
    """
    height, width = rgb_img.shape[:2]
    # 默认中心为图像中心
    if center is None:
        center = (width // 2, height // 2)
    # 默认椭圆半轴覆盖整个图像
    if axes is None:
        axes = (width // 2, height // 2)
    # 创建 mask
    mask = np.zeros((height, width), dtype=np.uint8)
    # 画椭圆
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    # 应用 mask
    masked_img = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)

    return masked_img

# ==========================
# 示例用法
# ==========================

# 假设你已经有 YUV420 数据
# data_bytes = ...
# width, height = 640, 480

# rgb_img = yuv420_to_rgb(data_bytes, width, height)
# cropped_img = crop_ellipse(rgb_img)

config_path = r"/home/chenkejing/PycharmProjects/EMdoorTotalDetect/blazepalmProject/RKNNDevelopmentBoardProgram/RKNNlite_Frame/fastdds_camera_frame_detect/config.ini"

if os.path.exists(config_path):
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    crop_config = {
        "center_x": config.getint("ImageCropConfig", "center_x"),
        "center_y": config.getint("ImageCropConfig", "center_y"),
        "axes_w": config.getint("ImageCropConfig", "axes_w"),
        "axes_h": config.getint("ImageCropConfig", "axes_h")
    }
else:
    crop_config = {
        "center_x": 130,
        "center_y": 117,
        "axes_w": 120,
        "axes_h": 125
    }

    print(crop_config)
cropped_img = image_crop_ellipse(rgb_img, center=(252, 234), axes=(250, 210))

# # 可视化
cv2.imshow("Cropped Ellipse", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
