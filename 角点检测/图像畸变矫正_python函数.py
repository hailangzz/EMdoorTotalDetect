import cv2
import numpy as np


def undistort_image(img_path, K, D):
    """
    对畸变图像进行去畸变操作（保持原始图像大小，不裁剪）
    :param img_path: 图像路径
    :param K: 相机内参矩阵 3x3 (fx, fy, cx, cy)
    :param D: 畸变参数 [k1,k2,p1,p2,k3] 或更多高阶参数
    :return: 去畸变后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("图像读取失败，请检查路径是否正确")

    h, w = img.shape[:2]

    # 使用原始相机矩阵，不裁剪
    new_K = K.copy()

    # 生成去畸变映射
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_32FC1)
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    return undistorted


# ----------------- 示例使用 -----------------
if __name__ == "__main__":
    # 相机内参矩阵
    K = np.array([[411.381960, 0, 637.411269],
                  [0, 411.535817, 351.293005],
                  [0, 0, 1]], dtype=np.float32)

    # 畸变参数: k1, k2, p1, p2, k3
    D = np.array([1.171783, 0.226351, 0.000015, 0.000199, 0.003289], dtype=np.float32)

    # img_path = "/home/chenkejing/Desktop/image_49.jpg"  # 替换为你的图像路径
    img_path = "./image_49_resized.jpg"  # 替换为你的图像路径
    undistorted_img = undistort_image(img_path, K, D)

    # 显示结果
    cv2.imshow("原图", cv2.imread(img_path))
    cv2.imshow("去畸变后", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存去畸变图像
    cv2.imwrite("undistorted_image_full.jpg", undistorted_img)
