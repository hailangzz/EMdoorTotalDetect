import numpy as np
import cv2
import configparser
import os

rgb_img = cv2.imread('./1.png')


def image_crop_ellipse(rgb_img, center=None, axes=None):
    """
    在原始 RGB 图像上裁切椭圆区域，并输出紧贴椭圆边缘的最小外接矩形区域。

    参数：
        rgb_img: 输入 RGB 图像
        center: (x, y)，椭圆中心，默认图像中心
        axes: (a, b)，椭圆半轴长度，默认覆盖整个图像

    返回：
        cropped_img: 紧贴椭圆外框的 RGB 图像
    """
    height, width = rgb_img.shape[:2]

    # 默认中心为图像中心
    if center is None:
        center = (width // 2, height // 2)
    # 默认半轴覆盖整张图
    if axes is None:
        axes = (width // 2, height // 2)

    # 创建二值 mask（椭圆区域为255）
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # 椭圆外设为黑色
    masked_img = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)

    # 获取非零区域的最小外接矩形
    coords = cv2.findNonZero(mask)
    if coords is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)

    # 裁剪出紧贴椭圆边缘的图像区域
    cropped_img = masked_img[y:y + h, x:x + w]

    return cropped_img

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
