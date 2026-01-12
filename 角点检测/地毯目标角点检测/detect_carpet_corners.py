import cv2
import numpy as np


def expand_bbox(bbox, img_shape, ratio=0.3):
    """对 YOLO bbox 做扩展，避免边缘被裁掉"""
    x1, y1, x2, y2 = bbox
    h, w = img_shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    dx = int(bw * ratio)
    dy = int(bh * ratio)

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)

    return x1, y1, x2, y2


def line_intersection(l1, l2):
    """两条直线求交点"""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    A = np.array([
        [x2 - x1, x3 - x4],
        [y2 - y1, y3 - y4]
    ], dtype=np.float32)

    B = np.array([
        x3 - x1,
        y3 - y1
    ], dtype=np.float32)

    if abs(np.linalg.det(A)) < 1e-6:
        return None

    t = np.linalg.solve(A, B)
    px = x1 + t[0] * (x2 - x1)
    py = y1 + t[0] * (y2 - y1)
    return np.array([px, py])


def order_corners(pts):
    """统一角点顺序：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def detect_carpet_corners_by_edges(image, bbox, debug=False):
    """
    基于 边缘 + 直线 + 交点 的角点检测
    返回：4x2 ndarray（原图坐标系），失败返回 None
    """

    # 1. bbox 扩展
    bbox = expand_bbox(bbox, image.shape, ratio=0.4)
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2].copy()

    if roi.size == 0:
        return None

    # 2. 灰度 + 去噪
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny 边缘
    edges = cv2.Canny(blur, 50, 150)

    # 4. 霍夫直线检测
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=0.5 * min(roi.shape[0], roi.shape[1]),
        maxLineGap=30
    )

    if lines is None:
        return None

    # 5. 按方向分组
    h_lines = []
    v_lines = []

    for line in lines:
        x1_, y1_, x2_, y2_ = line[0]
        angle = abs(np.arctan2(y2_ - y1_, x2_ - x1_))

        if angle < np.pi / 12:  # 近似水平
            h_lines.append((x1_, y1_, x2_, y2_))
        elif abs(angle - np.pi / 2) < np.pi / 12:  # 近似垂直
            v_lines.append((x1_, y1_, x2_, y2_))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    # 6. 选最外侧的四条边
    top = min(h_lines, key=lambda l: min(l[1], l[3]))
    bottom = max(h_lines, key=lambda l: max(l[1], l[3]))
    left = min(v_lines, key=lambda l: min(l[0], l[2]))
    right = max(v_lines, key=lambda l: max(l[0], l[2]))

    # 7. 求交点
    corners = []
    for l1, l2 in [(top, left), (top, right), (bottom, right), (bottom, left)]:
        p = line_intersection(l1, l2)
        if p is None:
            return None
        corners.append(p)

    corners = np.array(corners)

    # 8. 坐标映射回原图
    corners[:, 0] += x1
    corners[:, 1] += y1

    corners = order_corners(corners)

    if debug:
        dbg = roi.copy()
        for l in [top, bottom, left, right]:
            cv2.line(dbg, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2)
        for p in corners:
            cv2.circle(image, tuple(p.astype(int)), 8, (0, 0, 255), -1)

    return corners


# ==========================
# 示例主程序
# ==========================
if __name__ == "__main__":
    img = cv2.imread("/home/chenkejing/Desktop/image_49.jpg")

    # YOLO 检测框
    bbox = (47, 250, 630, 403)

    corners = detect_carpet_corners_by_edges(img, bbox, debug=True)

    if corners is not None:
        for p in corners:
            cv2.circle(img, tuple(p.astype(int)), 8, (255, 0, 0), -1)

    cv2.imshow("result", img)
    cv2.waitKey(0)
