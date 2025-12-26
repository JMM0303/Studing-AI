# src/pipeline/rectify_mag_roi.py

from typing import Tuple, Dict, Any, List

import cv2
import numpy as np


def undistort_if_needed(
    img_bgr: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """
    카메라 내·외부 파라미터가 cfg["camera"]에 있을 경우
    렌즈 왜곡 보정을 수행하고, 없거나 use_undistort=False면 원본 그대로 반환.
    """
    cam_cfg = cfg.get("camera", {})
    use_undistort = cam_cfg.get("use_undistort", False)

    if not use_undistort:
        return img_bgr

    camera_matrix = np.array(cam_cfg["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(cam_cfg["dist_coeffs"], dtype=np.float32)

    h, w = img_bgr.shape[:2]
    new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1.0, newImgSize=(w, h)
    )
    img_undist = cv2.undistort(img_bgr, camera_matrix, dist_coeffs, None, new_cam_mtx)
    return img_undist


def crop_magazine_roi(
    img_bgr: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    cfg["magazine_roi"]에 정의된 (x, y, w, h) 영역을 crop.
    반환:
        - img_roi: 잘라낸 BGR 이미지
        - roi_box: (x, y, w, h) 원본 기준 ROI 박스
    """
    roi_cfg = cfg["magazine_roi"]
    x = int(roi_cfg["x"])
    y = int(roi_cfg["y"])
    w = int(roi_cfg["w"])
    h = int(roi_cfg["h"])

    h_img, w_img = img_bgr.shape[:2]
    if x < 0 or y < 0 or x + w > w_img or y + h > h_img:
        raise ValueError(
            f"ROI ({x},{y},{w},{h}) is out of image bounds ({w_img}x{h_img})."
        )

    img_roi = img_bgr[y:y + h, x:x + w].copy()
    return img_roi, (x, y, w, h)


def compute_homography(
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    cfg['rectify']의 src_pts, dst_width, dst_height로 Homography 계산.
    src_pts는 ROI 좌표계(= crop된 이미지 기준)에서의 매거진 네 귀퉁이.
    """
    rect_cfg = cfg["rectify"]

    src_pts: List[List[float]] = rect_cfg["src_pts"]
    if len(src_pts) != 4:
        raise ValueError("rectify.src_pts must contain exactly 4 points.")

    dst_width = int(rect_cfg["dst_width"])
    dst_height = int(rect_cfg["dst_height"])

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(
        [
            [0.0, 0.0],
            [float(dst_width - 1), 0.0],
            [float(dst_width - 1), float(dst_height - 1)],
            [0.0, float(dst_height - 1)],
        ],
        dtype=np.float32,
    )

    H, status = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("Failed to compute homography matrix.")
    return H, (dst_width, dst_height)


def rectify_magazine(
    img_roi_bgr: np.ndarray,
    H: np.ndarray,
    dst_size: Tuple[int, int],
) -> np.ndarray:
    """
    ROI 이미지에 Homography를 적용하여 정면화.
    """
    dst_w, dst_h = dst_size
    img_rect = cv2.warpPerspective(img_roi_bgr, H, (dst_w, dst_h))
    return img_rect
