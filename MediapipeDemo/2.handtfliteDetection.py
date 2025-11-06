import cv2, numpy as np
from rknn.api import RKNN

# ========== 工具函数 ==========
def to_rgb_norm(img_bgr, size):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return img

def crop_by_box(img_bgr, box_xyxy, pad=0.1):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    # 扩边一点，防止裁剪太紧
    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
    bw = (x2 - x1); bh = (y2 - y1)
    bw *= (1 + pad); bh *= (1 + pad)
    x1 = max(0, int(cx - bw/2)); x2 = min(w, int(cx + bw/2))
    y1 = max(0, int(cy - bh/2)); y2 = min(h, int(cy + bh/2))
    return img_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def norm_landmarks_to_roi_xy(lm_21x2, roi_xyxy):
    x1, y1, x2, y2 = roi_xyxy
    rw = x2 - x1
    rh = y2 - y1
    # 将 ROI 内归一化坐标(0~1)还原到整图像素坐标
    pts = []
    for i in range(21):
        x = x1 + lm_21x2[i,0] * rw
        y = y1 + lm_21x2[i,1] * rh
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)

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

# # Export rknn model
# print('--> Export rknn model')
# ret = rknn.export_rknn('./palm_detection_full.rknn')
# if ret != 0:
#     print('Export rknn model failed!')
#     exit(ret)
# print('done')

ret = rknn.init_runtime()



# # ========== 推理一张图片 ==========
img_path = r'/home/chenkejing/EMdoor_TotalProgram/thumbs_up.jpg'
ori = cv2.imread(img_path)
H, W = ori.shape[:2]

# 1) 手部检测（按你的输入尺寸修改）
det_in_size = (192, 192)
det_in = to_rgb_norm(ori, det_in_size)
det_in = np.expand_dims(det_in, 0)  # NHWC
det_out = rknn.inference([det_in])

print(len(det_out))
print(det_out[1].shape,det_out[0].shape)
# ★★ 请打印 det_out 看真实结构，然后在此处解析 box ★★
# 假设输出是 [N, 6]: x1,y1,x2,y2,score,class；坐标为相对比例

boxes = det_out[0]
boxes = np.array(boxes).reshape(-1,6)  # 若维度不同请调整
boxes = boxes[boxes[:,4] > 0.5]       # 置信度阈值
if len(boxes) == 0:
    print('No hand detected'); exit(0)    

# 取置信度最高的一个
best = boxes[np.argmax(boxes[:,4])]
x1, y1, x2, y2 = best[:4]
x1, y1, x2, y2 = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
print(x1, y1, x2, y2)

roi, roi_xyxy = crop_by_box(ori, (x1,y1,x2,y2), pad=0.15)


# # 2) 关键点
# lm_in_size = (224, 224)  # 按你的模型实际输入
# lm_in = to_rgb_norm(roi, lm_in_size)
# lm_in = np.expand_dims(lm_in, 0)
# lm_out = lm.inference([lm_in])

# # ★★ 请打印 lm_out 看真实结构 ★★
# # 常见： [1, 21, 3] 或 [1,63]；x,y 为 0~1 的 ROI 归一化坐标
# lm_arr = lm_out[0]
# lm_arr = np.array(lm_arr).reshape(21, -1)     # 取前两列为 x,y
# lm_xy_roi = lm_arr[:,:2]
# lm_xy = norm_landmarks_to_roi_xy(lm_xy_roi, roi_xyxy)


# # 结果可视化
# for (x,y) in lm_xy.astype(int):
#     cv2.circle(ori, (x,y), 2, (0,255,0), -1)
# cv2.rectangle(ori, (x1,y1), (x2,y2), (0,128,255), 2)

# cv2.imwrite('result.jpg', ori)

# # 释放
# det.release(); lm.release(); 
# print('Saved result.jpg')