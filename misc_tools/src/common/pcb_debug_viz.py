# src/pipeline/pcb_debug_viz.py
import cv2
import numpy as np


def draw_ychain_and_widths(img, y_chain, widths_px, angles_deg, x0_band, x1_band):
    """
    img        : (H,W,3) BGR (매거진 ROI)
    y_chain    : 각 PCB 중심 y(px) 배열
    widths_px  : 각 PCB 폭(px, 수평 기준)
    angles_deg : 각 PCB 기울기 각도(deg, estimate_angles_and_draw 결과)
    x0_band    : 폭 측정 시작 x
    x1_band    : 폭 측정 끝 x

    → 각도까지 반영해서 '기울어진 폭 선'을 그림.
      선 길이는 width_px / cos(theta) 로 보정해서 실제 edge 길이에 맞춤.
    """
    vis = img.copy()
    H, W = vis.shape[:2]

    y_chain = np.asarray(y_chain, dtype=np.float32)
    widths_px = np.asarray(widths_px, dtype=np.float32)
    angles_deg = np.asarray(angles_deg, dtype=np.float32)

    cx = 0.5 * (float(x0_band) + float(x1_band))  # 선의 중심 x (밴드 중앙)
    text_x = min(W - 120, max(0, int(x1_band) + 5))

    for idx, (y, w, ang) in enumerate(zip(y_chain, widths_px, angles_deg)):
        y0 = float(y)
        y_int = int(round(y0))
        y_int = max(0, min(H - 1, y_int))

        theta = np.deg2rad(float(ang))
        cos_t = np.cos(theta)
        cos_safe = cos_t if abs(cos_t) > 1e-3 else (1e-3 * np.sign(cos_t) if cos_t != 0 else 1e-3)

        # 수평 폭을 각도 보정 → 실제 edge 길이(대략)
        length = float(w) / cos_safe

        dx = 0.5 * length * np.cos(theta)
        dy = 0.5 * length * np.sin(theta)

        cx_f = float(cx)
        cy_f = y0

        x1 = int(round(cx_f - dx))
        y1 = int(round(cy_f - dy))
        x2 = int(round(cx_f + dx))
        y2 = int(round(cy_f + dy))

        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))

        # 기울어진 폭 선
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

        # 양 끝점 표시
        cv2.circle(vis, (x1, y1), 3, (255, 0, 0), -1)
        cv2.circle(vis, (x2, y2), 3, (0, 0, 255), -1)

        # 텍스트
        txt = f"{idx}: y={y_int}, w={w:.1f}, a={ang:.1f}"
        cv2.putText(
            vis,
            txt,
            (text_x, max(12, y_int - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return vis
