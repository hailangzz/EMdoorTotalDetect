import cv2
import numpy as np
import sys
import os
#from rknn.api import RKNN
from rknnlite.api import RKNNLite

import blazepalm_utils as but

# ======================
# 基本配置
# ======================
# IMAGE_PATH = 'thumbs_up.jpg'
IMAGE_PATH = "fastdds_test.png"
SAVE_IMAGE_PATH = 'output.png'

RKNN_MODEL = 'palm_detection_full.rknn'  # 使用 .rknn 模型

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 192
ANCHOR_PATH = 'anchors_192.npy'
CHANNEL_FIRST = False


def imread(filename, flags=cv2.IMREAD_COLOR):
    if not os.path.isfile(filename):
        sys.exit(f"❌ 文件不存在: {filename}")
    data = np.fromfile(filename, np.int8)
    img = cv2.imdecode(data, flags)
    return img

def get_savepath(arg_path, src_path, prefix='', post_fix='_res', ext=None):
    if '.' in arg_path:
        arg_base, arg_ext = os.path.splitext(arg_path)
        new_ext = arg_ext if ext is None else ext
        new_path = arg_base + new_ext
    else:
        src_base, src_ext = os.path.splitext(os.path.basename(src_path))
        new_ext = src_ext if ext is None else ext
        new_path = os.path.join(arg_path, prefix + src_base + post_fix + new_ext)
    dirname = os.path.dirname(new_path)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
    return new_path

def display_result(img, detections, with_keypoints=True):
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    n_keypoints = detections.shape[1] // 2 - 2
    for i in range(detections.shape[0]):
        ymin, xmin, ymax, xmax = detections[i, :4].astype(int)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        if with_keypoints:
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k*2])
                kp_y = int(detections[i, 4 + k*2 + 1])
                cv2.circle(img, (kp_x, kp_y), 2, (0, 0, 255), thickness=2)
    return img

# ======================
# 主推理函数
# ======================
def recognize_from_image():
    # 加载图片
    src_img = imread(IMAGE_PATH)
    img256, _, scale, pad = but.resize_pad(src_img[:, :, ::-1], IMAGE_WIDTH)
    input_data = img256.astype('float32') / 255.
    input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)
    if not CHANNEL_FIRST:
        input_data = input_data.transpose((0, 2, 3, 1))

    # ======================
    # 直接加载 RKNN 模型
    # ======================
    # rknn = RKNN()
    rknn_lite = RKNNLite()

    # 设置目标平台为模拟器
    # rknn.config(target_platform='rk3588', mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]])

    print(f'加载 RKNN 模型: {RKNN_MODEL}')
    # ret = rknn.load_rknn(RKNN_MODEL)
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('❌ 加载 RKNN 模型失败！')
        exit(ret)

    # 初始化 RKNN runtime
    print('--> 初始化 RKNN runtime...')
    # ret = rknn.init_runtime(target='rk3588')
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print('❌ 初始化 RKNN runtime 失败！')
        exit(ret)

    # ======================
    # 模型推理
    # ======================
    print("🚀 开始推理...")
    # outputs = rknn.inference(inputs=[input_data])
    outputs = rknn_lite.inference(inputs=[input_data])

    print(f"📤 获得 {len(outputs)} 个输出张量：")
    for i, out in enumerate(outputs):
        print(f"Output[{i}] shape: {out.shape}")
    preds = outputs

    normalized_detections = but.postprocess(preds, anchor_path=ANCHOR_PATH, resolution=IMAGE_WIDTH)[0]
    detections = but.denormalize_detections(normalized_detections, scale, pad, resolution=IMAGE_WIDTH)

    # ======================
    # 显示与保存结果
    # ======================
    result_img = display_result(src_img, detections)
    savepath = get_savepath(SAVE_IMAGE_PATH, "/")
    cv2.imwrite(savepath, result_img)
    print(f'💾 结果已保存至: {savepath}')

    # rknn.release()
    rknn_lite.release()

def main():
    recognize_from_image()

if __name__ == '__main__':
    main()
