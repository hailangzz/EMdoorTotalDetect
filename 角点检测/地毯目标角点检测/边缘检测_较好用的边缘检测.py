import cv2
import numpy as np
from scipy import interpolate

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


def fit_contours_to_lines_and_curves(contours, img_shape, distance_thresh=5):
    lines = []

    for cnt in contours:
        if len(cnt) < 5:
            continue

        # 拟合直线
        [vx, vy, x0, y0] = cv2.fitLine(cnt.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        line_dir = np.array([vx[0], vy[0]])
        line_point = np.array([x0[0], y0[0]])

        # 投影形成端点
        projections = np.dot(cnt - line_point, line_dir)
        min_proj, max_proj = projections.min(), projections.max()
        start_pt = (line_point + min_proj * line_dir).astype(int)
        end_pt   = (line_point + max_proj * line_dir).astype(int)
        lines.append((tuple(start_pt), tuple(end_pt)))

        # 可选：拟合平滑曲线
        if len(cnt) >= 10:
            x, y = cnt[:,0], cnt[:,1]
            tck, u = interpolate.splprep([x, y], s=2)
            u_new = np.linspace(0, 1, max(50, len(x)))
            x_new, y_new = interpolate.splev(u_new, tck)
            curve_pts = np.vstack([x_new, y_new]).T.astype(int)
            lines.append(curve_pts)

    return lines


def fit_edges_to_lines_and_curves(edges, min_points=5):
    """
    将边缘图 edges 拟合成连续的直线或曲线
    edges: 二值边缘图
    return: list of lines/curves, 每条为 Nx2
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    fitted = []

    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        if len(cnt) < min_points:
            continue

        # 拟合直线
        [vx, vy, x0, y0] = cv2.fitLine(cnt.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        line_dir = np.array([vx[0], vy[0]])
        line_point = np.array([x0[0], y0[0]])

        # 投影形成端点
        projections = np.dot(cnt - line_point, line_dir)
        min_proj, max_proj = projections.min(), projections.max()
        start_pt = (line_point + min_proj * line_dir).astype(int)
        end_pt   = (line_point + max_proj * line_dir).astype(int)
        fitted.append(np.array([start_pt, end_pt]))

        # 拟合平滑曲线
        if len(cnt) >= 10:
            x, y = cnt[:,0], cnt[:,1]
            tck, u = interpolate.splprep([x, y], s=2)
            u_new = np.linspace(0, 1, max(50, len(x)))
            x_new, y_new = interpolate.splev(u_new, tck)
            curve_pts = np.vstack([x_new, y_new]).T.astype(int)
            fitted.append(curve_pts)

    return fitted

# --------------------------
# 测试示例
# --------------------------

if __name__ == "__main__":
    img = cv2.imread("carpet.jpg")
    # bbox = (110, 620, 1070, 920)
    bbox = (90, 600, 1100, 940)

    roi = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(blur, 30, 120)

    lines_curves = fit_edges_to_lines_and_curves(edges)

    vis = img.copy()
    for lc in lines_curves:
        for i in range(len(lc) - 1):
            cv2.line(vis, tuple(lc[i]), tuple(lc[i + 1]), (0, 0, 255), 2)

    cv2.imshow("edges", edges)
    cv2.imshow("fitted lines & curves", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()