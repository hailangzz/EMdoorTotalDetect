import cv2
import time
import os

# ---------- 配置 ----------
camera_index = 0                 # 摄像头索引，一般主摄像头为 0
save_dir = "./camera_images"     # 保存路径
interval = 5                    # 间隔秒数
total_images = 10                # 总共拍摄张数

# 创建保存目录
os.makedirs(save_dir, exist_ok=True)

# 打开摄像头
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("can not open camera")
    exit()

print(f"start get {total_images} images，each {interval} minits one...")

for i in range(total_images):
    ret, frame = cap.read()
    if not ret:
        print(f" {i+1} picters get fail!")
        continue

    # 保存图片
    filename = os.path.join(save_dir, f"image_{i+1}.jpg")
    cv2.imwrite(filename, frame)
    print(f"save picters: {filename}")

    if i < total_images - 1:
        time.sleep(interval)  # 间隔

# 释放摄像头
cap.release()
print("done！")
