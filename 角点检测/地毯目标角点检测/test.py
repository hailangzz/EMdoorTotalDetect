# import cv2
# import numpy as np
#
# def gradient_edges_connected_contour(img, bbox, ksize=3,
#                                      percentile=90,
#                                      dilate_iter=2,
#                                      smooth_window=15):
#     left, top, right, bottom = bbox
#     h, w = img.shape[:2]
#
#     left = max(0, left)
#     top = max(0, top)
#     right = min(w, right)
#     bottom = min(h, bottom)
#     if right <= left or bottom <= top:
#         return None, []
#
#     roi = img[top:bottom, left:right]
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#
#     grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
#     grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
#     grad_mag = cv2.magnitude(grad_x, grad_y)
#
#     # ⭐ 自适应阈值
#     thresh = np.percentile(grad_mag, percentile)
#     edges = np.zeros_like(grad_mag, dtype=np.uint8)
#     edges[grad_mag > thresh] = 255
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
#     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not contours:
#         return edges, []
#
#     largest_contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
#
#     # 映射回原图坐标
#     largest_contour[:, 0] += left
#     largest_contour[:, 1] += top
#
#     # 均值平滑
#     if len(largest_contour) >= smooth_window:
#         half = smooth_window // 2
#         smoothed = np.zeros_like(largest_contour, dtype=np.float32)
#         for i in range(len(largest_contour)):
#             s = max(0, i-half)
#             e = min(len(largest_contour), i+half+1)
#             smoothed[i] = np.mean(largest_contour[s:e], axis=0)
#         largest_contour = smoothed.astype(np.int32)
#
#     return edges, largest_contour
#
#
# # --------------------------
# # 测试
# # --------------------------
# if __name__ == "__main__":
#
#     img = cv2.imread("carpet.jpg")
#     bbox = (0, 66, 590, 635)
#
#     # img = cv2.imread("carpet1.jpg")
#     # bbox = (90, 600, 1100, 940)
#
#     # img = cv2.imread("carpet3.jpg")
#     # bbox = (0, 170, 640, 620)
#
#     # img = cv2.imread("carpet4.jpg")
#     # bbox = (833, 615, 1920, 1080)
#
#     edges, contour = gradient_edges_connected_contour(img, bbox, percentile=78, smooth_window=7)
#
#     vis = img.copy()
#
#     # ✅ 画 bbox
#     cv2.rectangle(
#         vis,
#         (bbox[0], bbox[1]),
#         (bbox[2], bbox[3]),
#         (255, 0, 0),
#         2
#     )
#
#     # ✅ 画平滑轮廓
#     if len(contour) > 0:
#         cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
#
#     # cv2.imshow("ROI Edges", edges)
#     cv2.imshow("Contour + BBox", vis)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



import cv2
import numpy as np


def extract_carpet_boundary_robust(
    img,
    bbox,
    ksize=3,
    percentile=90,
    dilate_iter=2,
    smooth_window=11,
    edge_margin=20
):
    """
    抗干扰地毯边界提取（工程版）
    - bbox 边缘带约束
    - 自适应梯度阈值
    - 轮廓长度筛选
    - 均值平滑
    """

    left, top, right, bottom = bbox
    h, w = img.shape[:2]

    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)
    if right <= left or bottom <= top:
        return None, []

    roi = img[top:bottom, left:right]

    # -----------------------------
    # 1. 灰度 + 梯度
    # -----------------------------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    grad = cv2.magnitude(gx, gy)

    # -----------------------------
    # 2. 自适应阈值
    # -----------------------------
    thresh = np.percentile(grad, percentile)
    edges = np.zeros_like(grad, dtype=np.uint8)
    edges[grad > thresh] = 255

    # -----------------------------
    # 3. 邻近连接
    # -----------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # -----------------------------
    # 4. 提取轮廓
    # -----------------------------
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return edges, []

    # -----------------------------
    # 5. 只保留「靠近 bbox 边缘」的轮廓
    # -----------------------------
    H, W = roi.shape[:2]
    candidates = []

    for cnt in contours:
        cnt = cnt.reshape(-1, 2)

        near_edge = (
            (cnt[:, 0] < edge_margin) |
            (cnt[:, 0] > W - edge_margin) |
            (cnt[:, 1] < edge_margin) |
            (cnt[:, 1] > H - edge_margin)
        )

        ratio = np.sum(near_edge) / len(cnt)

        # ⭐ 至少 60% 点靠近 bbox 边缘
        if ratio > 0.6:
            candidates.append(cnt)

    if not candidates:
        return edges, []

    # -----------------------------
    # 6. 选“最长轮廓”（不是最大面积）
    # -----------------------------
    best = max(candidates, key=lambda c: cv2.arcLength(c, False))

    # 映射回原图坐标
    best[:, 0] += left
    best[:, 1] += top

    # -----------------------------
    # 7. 均值平滑去抖
    # -----------------------------
    if len(best) >= smooth_window:
        half = smooth_window // 2
        smooth = np.zeros_like(best, dtype=np.float32)
        for i in range(len(best)):
            s = max(0, i - half)
            e = min(len(best), i + half + 1)
            smooth[i] = np.mean(best[s:e], axis=0)
        best = smooth.astype(np.int32)

    return edges, best


# --------------------------------
# 测试示例
# --------------------------------
if __name__ == "__main__":
    img = cv2.imread("carpet4.jpg")
    bbox = (833, 615, 1920, 1080)

    edges, contour = extract_carpet_boundary_robust(
        img,
        bbox,
        percentile=92,      # ⭐ 基本只调这个
        edge_margin=20,     # ⭐ bbox 边缘带宽度
        smooth_window=9
    )

    vis = img.copy()

    # 画 bbox
    cv2.rectangle(
        vis,
        (bbox[0], bbox[1]),
        (bbox[2], bbox[3]),
        (255, 0, 0),
        2
    )

    # 画地毯边界
    if len(contour) > 0:
        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)

    cv2.imshow("edges", edges)
    cv2.imshow("robust carpet boundary", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




