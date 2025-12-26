# src/utils/viz_utils.py

from pathlib import Path
import cv2


def save_slot_debug_image(img_bgr, slot_rois, out_path):
    """
    슬롯 ROI가 이미지 상에서 어떻게 나뉘는지 확인하기 위한
    디버그용 이미지 저장 함수.

    - img_bgr: 원본 BGR 이미지 (cv2.imread 결과)
    - slot_rois: [(y_start, y_end), ...] 리스트
    - out_path: 저장할 파일 경로 (Path 또는 str)
    """
    debug_img = img_bgr.copy()
    height, width, _ = debug_img.shape

    # 슬롯마다 다른 색으로 가로선 그리기
    for idx, (y_start, y_end) in enumerate(slot_rois):
        color = (0, 255, 0)  # 슬롯 경계선 색 (초록)
        thickness = 1

        # 상단 라인
        cv2.line(debug_img, (0, y_start), (width - 1, y_start), color, thickness)
        # 하단 라인
        cv2.line(debug_img, (0, y_end), (width - 1, y_end), color, thickness)

        # 슬롯 번호 표시
        text = f"{idx}"
        cv2.putText(
            debug_img,
            text,
            (5, max(y_start + 15, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug_img)

def save_pcb_rows_debug_image(img_bgr, pcb_rows, out_path):
    """
    PCB가 있다고 판단한 y 위치들을 원본 이미지 위에 그려서 저장한다.
    - pcb_rows: [y0, y1, ...] (float, 이미지 전체 기준)
    """
    debug_img = img_bgr.copy()
    h, w, _ = debug_img.shape

    for idx, y in enumerate(pcb_rows):
        yy = int(round(y))
        # 보라색 가로선
        cv2.line(debug_img, (0, yy), (w - 1, yy), (255, 0, 255), 1)
        # 텍스트로 index, y값 표시
        cv2.putText(
            debug_img,
            f"{idx}:{y:.1f}",
            (5, max(yy - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug_img)

