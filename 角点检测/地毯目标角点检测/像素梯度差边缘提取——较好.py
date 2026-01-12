
## 像素梯度差边缘算法，来提取目标框内边界框吗？
# import cv2
# import numpy as np
#
# def gradient_difference_edges(img, bbox, ksize=3, thresh=30):
#     """
#     提取目标框内的边界（像素梯度差方法）
#     img: BGR图像
#     bbox: (left, top, right, bottom)
#     ksize: Sobel/Scharr卷积核大小
#     thresh: 梯度差阈值
#     return: edges 二值图, contours 点列表
#     """
#     left, top, right, bottom = bbox
#     h, w = img.shape[:2]
#
#     # 裁剪 ROI
#     left = max(0, left)
#     top = max(0, top)
#     right = min(w, right)
#     bottom = min(h, bottom)
#     if right <= left or bottom <= top:
#         return None, []
#
#     roi = img[top:bottom, left:right]
#
#     # 1. 灰度
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#
#     # 2. 计算梯度
#     grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
#     grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
#
#     # 3. 计算梯度幅值
#     grad_mag = cv2.magnitude(grad_x, grad_y)
#
#     # 4. 梯度差阈值化
#     edges = np.zeros_like(grad_mag, dtype=np.uint8)
#     edges[grad_mag > thresh] = 255
#
#     # 5. 形态学操作增强边缘连续性
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#     edges = cv2.dilate(edges, None, iterations=1)
#
#     # 6. 提取轮廓
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#     # 映射回原图
#     mapped_contours = []
#     for cnt in contours:
#         cnt = cnt.reshape(-1, 2)
#         cnt[:, 0] += left
#         cnt[:, 1] += top
#         mapped_contours.append(cnt)
#
#     return edges, mapped_contours
#
# # --------------------------
# # 测试示例
# # --------------------------
# if __name__ == "__main__":
#     img = cv2.imread("carpet.jpg")
#     # bbox = (110, 620, 1070, 920)
#     bbox = (90, 600, 900, 740)
#
#     edges, contours = gradient_difference_edges(img, bbox, ksize=3, thresh=30)
#
#     vis = img.copy()
#     for cnt in contours:
#         cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
#
#     cv2.imshow("Edges", edges)
#     cv2.imshow("Contours", vis)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#


## 以上代码形成的边界框，提出掉离散的边界框点
# import cv2
# import numpy as np
#
# def gradient_edges_cleaned(img, bbox, ksize=3, thresh=30, min_points=50):
#     """
#     提取目标框内的边界（像素梯度差方法），并去掉离散点
#     img: BGR图像
#     bbox: (left, top, right, bottom)
#     ksize: Sobel卷积核大小
#     thresh: 梯度幅值阈值
#     min_points: 轮廓点最少数量，小于这个丢弃
#     return: edges 二值图, cleaned_contours 点列表
#     """
#     left, top, right, bottom = bbox
#     h, w = img.shape[:2]
#
#     # ROI裁剪
#     left = max(0, left)
#     top = max(0, top)
#     right = min(w, right)
#     bottom = min(h, bottom)
#     if right <= left or bottom <= top:
#         return None, []
#
#     roi = img[top:bottom, left:right]
#
#     # 1. 灰度
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#
#     # 2. Sobel梯度
#     grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
#     grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
#     grad_mag = cv2.magnitude(grad_x, grad_y)
#
#     # 3. 阈值化
#     edges = np.zeros_like(grad_mag, dtype=np.uint8)
#     edges[grad_mag > thresh] = 255
#
#     # 4. 形态学增强
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#     edges = cv2.dilate(edges, None, iterations=1)
#
#     # 5. 提取轮廓
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#     # 6. 去掉离散小轮廓
#     cleaned_contours = []
#     for cnt in contours:
#         if cnt.shape[0] >= min_points:
#             cnt = cnt.reshape(-1, 2)
#             cnt[:, 0] += left
#             cnt[:, 1] += top
#             cleaned_contours.append(cnt)
#
#     return edges, cleaned_contours
#
#
#
# # --------------------------
# # 测试示例
# # --------------------------
# if __name__ == "__main__":
#     img = cv2.imread("carpet.jpg")
#     bbox = (90, 600, 1100, 940)
#
#     edges, contours = gradient_edges_cleaned(img, bbox, ksize=3, thresh=30, min_points=80)
#
#     vis = img.copy()
#     for cnt in contours:
#         cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
#
#     cv2.imshow("Edges", edges)
#     cv2.imshow("Cleaned Contours", vis)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



# ## 以上代码提取中，只保留区域面积最大的边界框
# import cv2
# import numpy as np
#
# def gradient_edges_largest_contour(img, bbox, ksize=3, thresh=30):
#     """
#     提取目标框内的边界（像素梯度差方法），只保留面积最大的轮廓
#     img: BGR图像
#     bbox: (left, top, right, bottom)
#     ksize: Sobel卷积核大小
#     thresh: 梯度幅值阈值
#     return: edges 二值图, largest_contour 点列表
#     """
#     left, top, right, bottom = bbox
#     h, w = img.shape[:2]
#
#     # ROI裁剪
#     left = max(0, left)
#     top = max(0, top)
#     right = min(w, right)
#     bottom = min(h, bottom)
#     if right <= left or bottom <= top:
#         return None, []
#
#     roi = img[top:bottom, left:right]
#
#     # 1. 灰度
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#
#     # 2. Sobel梯度
#     grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
#     grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
#     grad_mag = cv2.magnitude(grad_x, grad_y)
#
#     # 3. 阈值化
#     edges = np.zeros_like(grad_mag, dtype=np.uint8)
#     edges[grad_mag > thresh] = 255
#
#     # 4. 形态学增强
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#     edges = cv2.dilate(edges, None, iterations=1)
#
#     # 5. 提取轮廓
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#     if not contours:
#         return edges, []
#
#     # 6. 选取最大面积轮廓
#     largest_contour = max(contours, key=cv2.contourArea)
#     largest_contour = largest_contour.reshape(-1, 2)
#     largest_contour[:, 0] += left
#     largest_contour[:, 1] += top
#
#     return edges, largest_contour
#
#
# # --------------------------
# # 测试示例
# # --------------------------
# if __name__ == "__main__":
#     img = cv2.imread("carpet.jpg")
#     bbox = (90, 600, 1100, 940)
#
#     edges, largest_contour = gradient_edges_largest_contour(img, bbox, ksize=3, thresh=30)
#
#     vis = img.copy()
#     if len(largest_contour) > 0:
#         cv2.drawContours(vis, [largest_contour], -1, (0, 255, 0), 2)
#
#     cv2.imshow("Edges", edges)
#     cv2.imshow("Largest Contour", vis)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#


## 边缘线部分区域曲折波动、重复、考虑是否使用什么像素最临近膨蚀的方法，做最临近直接连接
import cv2
import numpy as np

def gradient_edges_connected_contour(img, bbox, ksize=3, thresh=30, dilate_iter=2, smooth_window=15):
    """
    提取目标框内的边界（像素梯度差方法），最大轮廓 + 邻近膨胀连接 + 均值平滑
    """
    left, top, right, bottom = bbox
    h, w = img.shape[:2]

    # ROI裁剪
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)
    if right <= left or bottom <= top:
        return None, []

    roi = img[top:bottom, left:right]

    # 1. 灰度
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 2. Sobel梯度
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # 3. 阈值化
    edges = np.zeros_like(grad_mag, dtype=np.uint8)
    edges[grad_mag > thresh] = 255

    # 4. 邻近膨胀连接
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 5. 提取轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return edges, []

    # 6. 最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour = largest_contour.reshape(-1, 2)

    # 映射回原图
    largest_contour[:, 0] += left
    largest_contour[:, 1] += top

    # 7. 均值平滑
    if len(largest_contour) >= smooth_window:
        half_win = smooth_window // 2
        smoothed = np.zeros_like(largest_contour, dtype=np.float32)
        for i in range(len(largest_contour)):
            start = max(0, i - half_win)
            end = min(len(largest_contour), i + half_win + 1)
            smoothed[i] = np.mean(largest_contour[start:end], axis=0)
        largest_contour = smoothed.astype(np.int32)

    return edges, largest_contour

# --------------------------
# 测试
# --------------------------
if __name__ == "__main__":
    #img = cv2.imread("carpet1.jpg")
    # bbox = (90, 600, 1100, 940)
    # bbox = (90, 600, 1100, 940)

    img = cv2.imread("carpet.jpg")
    # bbox = (90, 600, 1100, 940)
    bbox = (0, 66, 590, 635)



    # edges, largest_contour = gradient_edges_connected_contour(
    #     img, bbox, ksize=3, thresh=38, dilate_iter=2, smooth_window=7
    # )

    edges, largest_contour = gradient_edges_connected_contour(
        img, bbox, ksize=3, thresh=100, dilate_iter=2, smooth_window=7
    )

    vis = img.copy()
    if len(largest_contour) > 0:
        cv2.drawContours(vis, [largest_contour], -1, (0, 255, 0), 2)

    cv2.imshow("Connected Edges", edges)
    cv2.imshow("Largest Smoothed Contour", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()











