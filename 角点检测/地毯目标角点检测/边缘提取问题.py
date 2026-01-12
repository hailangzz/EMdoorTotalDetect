import cv2
import numpy as np

def extract_dense_edges_in_bbox(img, bbox):
    """
    在 bbox 内提取尽可能“密集”的所有边缘细节
    返回：edge_map, contours
    """
    left, top, right, bottom = bbox
    h, w = img.shape[:2]

    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)

    roi = img[top:bottom, left:right]

    # 1. 灰度
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 2. 双边滤波（保边去噪）
    blur = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 3. Scharr 梯度（比 Sobel 更敏感）
    grad_x = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(blur, cv2.CV_32F, 0, 1)

    # 4. 梯度幅值
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
    grad_mag = grad_mag.astype(np.uint8)

    # 5. 自适应阈值（保留弱边）
    edges = cv2.adaptiveThreshold(
        grad_mag,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # 6. 形态学操作：闭运算 + 细化
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 7. 轻微膨胀，增强连通性
    edges = cv2.dilate(edges, None, iterations=1)

    # 8. 提取轮廓（所有）
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    # 映射回原图坐标
    mapped_contours = []
    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        cnt[:, 0] += left
        cnt[:, 1] += top
        mapped_contours.append(cnt)

    return edges, mapped_contours



if __name__ == "__main__":
    img = cv2.imread("carpet.jpg")
    # bbox = (110, 620, 1070, 920)
    bbox = (90, 600, 1100, 940)

    edges, contours = extract_dense_edges_in_bbox(img, bbox)

    vis = img.copy()
    for cnt in contours:
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 1)

    cv2.imshow("dense edges", edges)
    cv2.imshow("all edge details", vis)
    cv2.waitKey(0)