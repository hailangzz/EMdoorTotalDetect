import cv2, numpy as np
from rknn.api import RKNN

# 创建RKNN对象
rknn = RKNN(verbose=True)
# rknn.config(mean_values=[0, 0, 0], std_values=[255, 255, 255], target_platform='rk3588')
rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]], target_platform='rk3588')

# 加载模型
ret = rknn.load_tflite(model=r'/home/chenkejing/EMdoor_TotalProgram/MediapipeDemo/models/palm_detection_full.tflite')
if ret != 0:
    print('Load RKNN model failed!')
    exit(ret)

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=False)
if ret != 0:
        print('Build model failed!')
        exit(ret)
print('done')

# Export rknn model
print('--> Export rknn model')
ret = rknn.export_rknn('./palm_detection_full.rknn')
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)
print('done')
