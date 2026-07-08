import cv2
import numpy as np

def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # 左上
    ordered[2] = pts[np.argmax(s)]  # 右下
    ordered[1] = pts[np.argmin(d)]  # 右上
    ordered[3] = pts[np.argmax(d)]  # 左下
    return ordered

def detect_dark_rectangle(frame):
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 暗い中央動画領域を抽出
    _, mask = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

    # 内部の白い細胞部分などで穴が空くので埋める
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    frame_area = h * w

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.08 * frame_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        cx = x + bw / 2
        cy = y + bh / 2

        # 中央付近の縦長長方形だけを採用
        if not (0.35 <= aspect <= 0.75):
            continue
        if not (0.25 * w <= cx <= 0.75 * w):
            continue
        if not (0.25 * h <= cy <= 0.85 * h):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)

        candidates.append((area, order_points(quad)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]