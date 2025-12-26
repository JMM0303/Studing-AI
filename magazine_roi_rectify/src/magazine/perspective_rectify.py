import cv2
import numpy as np


def detect_vertical_edges(img_gray):
    edges = cv2.Canny(img_gray, 80, 180)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=200,
        maxLineGap=20
    )

    left_lines = []
    right_lines = []

    h, w = img_gray.shape[:2]

    if lines is None:
        raise RuntimeError("No lines detected.")

    for (x1, y1, x2, y2) in lines[:, 0]:
        # 거의 수직선만 수집
        if abs(x1 - x2) < 10:
            # 좌측/우측 분류
            if x1 < w // 2:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    if len(left_lines) == 0 or len(right_lines) == 0:
        raise RuntimeError("Left/right vertical edges not detected.")

    # 가장 길어보이는 대표 선 선택
    left_line = max(left_lines, key=lambda L: abs(L[1] - L[3]))
    right_line = max(right_lines, key=lambda L: abs(L[1] - L[3]))

    return left_line, right_line


def compute_shear_matrix(left_line, right_line):
    x1, y1, x2, y2 = left_line
    slope_L = (y2 - y1) / (x2 - x1 + 1e-6)

    x3, y3, x4, y4 = right_line
    slope_R = (y4 - y3) / (x4 - x3 + 1e-6)

    # 두 기울기 각도
    ang_L = np.arctan(slope_L)
    ang_R = np.arctan(slope_R)

    # 시점 보정
    theta = (ang_L - ang_R) / 2.0

    # x축 스큐(원근 펴기)
    H_shear = np.array([
        [1, np.tan(-theta), 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return H_shear


def homography_perspective_rectify(img_bgr, H_from_pts, dst_size):
    # 1) pts 기반 정면화 1차
    w, h = dst_size
    img_rect1 = cv2.warpPerspective(img_bgr, H_from_pts, (w, h))

    # 그레이 변환
    gray = cv2.cvtColor(img_rect1, cv2.COLOR_BGR2GRAY)

    # 2) 좌/우 벽 직선 검출
    left_line, right_line = detect_vertical_edges(gray)

    # 3) 원근 보정 행렬 H₂ 구하기
    H_shear = compute_shear_matrix(left_line, right_line)

    # 4) 최종 합성 Homography
    H_total = H_shear @ H_from_pts

    # 5) 최종 정면화
    img_rect2 = cv2.warpPerspective(img_bgr, H_total, (w, h))

    return img_rect2, H_total
