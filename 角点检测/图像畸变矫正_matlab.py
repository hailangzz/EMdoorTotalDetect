import cv2
import numpy as np

import cv2
import numpy as np

# 相机内参
A = np.array([[411.381960, 0, 637.411269],
              [0, 411.535817, 351.293005],
              [0, 0, 1]], dtype=np.float32)

# 畸变参数
D = np.array([1.171783, 0.226351, 0.000015, 0.000199, 0.003289], dtype=np.float32)

fx, fy = A[0,0], A[1,1]
cx, cy = A[0,2], A[1,2]
k1, k2, p1, p2, k3 = D[0], D[1], D[2], D[3], D[4]

# 读取图像
# I_d = cv2.imread("/home/chenkejing/Desktop/image_49.jpg")
I_d = cv2.imread("./image_49_resized.jpg")  # 替换为你的图像路径
# I_d = cv2.imread("/home/chenkejing/Desktop/image_49.jpg")
I_d = I_d.astype(np.float32) / 255.0  # 转为 [0,1] 浮点数
m, n = I_d.shape[:2]

# 生成去畸变映射
u = np.arange(n)
v = np.arange(m)
U, V = np.meshgrid(u, v)

# 归一化坐标
X = (U - cx) / fx
Y = (V - cy) / fy
r2 = X**2 + Y**2

# 畸变校正公式（这里是逆畸变，需要迭代或近似）
X_dist = X*(1 + k1*r2 + k2*r2**2 + k3*r2**3) + 2*p1*X*Y + p2*(r2 + 2*X**2)
Y_dist = Y*(1 + k1*r2 + k2*r2**2 + k3*r2**3) + 2*p2*X*Y + p1*(r2 + 2*Y**2)

# 像素坐标
U_dist = fx*X_dist + cx
V_dist = fy*Y_dist + cy

# 使用 cv2.remap 做插值
U_dist = U_dist.astype(np.float32)
V_dist = V_dist.astype(np.float32)
img_undistorted = cv2.remap(I_d, U_dist, V_dist, interpolation=cv2.INTER_LINEAR)

# 显示结果
cv2.imshow("/home/chenkejing/Desktop/image_49.jpg", I_d)
cv2.imshow("/home/chenkejing/Desktop/un_image_49.jpg", img_undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()


