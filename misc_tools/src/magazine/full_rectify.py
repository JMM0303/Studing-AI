# src/pipeline/rectify_mag_pipeline.py

from pathlib import Path
import cv2
import numpy as np
import yaml

# 원래: from geometry.z_from_geometry import estimate_z_for_rows
try:
    from geometry.z_from_geometry import estimate_z_for_rows
except ModuleNotFoundError:
    # geometry 모듈이 아직 없을 때를 위한 임시 스텁
    def estimate_z_for_rows(*args, **kwargs):
        """
        TODO: 나중에 폭 기반 Z 계산으로 교체 예정.
        지금은 길이만 맞는 0 배열을 리턴해서 파이프라인만 통과시키는 용도.
        """
        import numpy as _np
        if not args:
            return None
        y_rows = args[0]
        try:
            n = len(y_rows)
        except TypeError:
            return None
        return _np.zeros(n, dtype=_np.float32)


def _load_rectify_yaml(yaml_path: str):
    """
    rectify_mag.yaml 형식:
      H: [[...],[...],[...]]
      dst_w: 220
      dst_h: 430
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    H = np.array(data["H"], dtype=np.float32)
    dst_w = int(data.get("dst_w", 0))
    dst_h = int(data.get("dst_h", 0))
    return H, dst_w, dst_h


def undistort_with_intrinsics(img_bgr, cam_cfg: dict):
    """
    ① 카메라 캘리브레이션 기반 왜곡 보정

    config 예시:
      camera:
        fx: 1200.0
        fy: 1200.0
        cx: 960.0
        cy: 540.0
        dist_coeffs: [k1, k2, p1, p2, k3]  # 없으면 왜곡 보정 생략
    """
    h, w = img_bgr.shape[:2]

    fx = float(cam_cfg.get("fx", 0.0))
    fy = float(cam_cfg.get("fy", 0.0))
    cx = float(cam_cfg.get("cx", w / 2.0))
    cy = float(cam_cfg.get("cy", h / 2.0))

    # 카메라 행렬 K
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    dist_list = cam_cfg.get("distortion", None)

    if not dist_list:
        # distortion 정보 없음 → undistort 건너뜀
        return img_bgr

    # distortion 정보 있을 때만 실행
    dist = np.array(dist_list, dtype=np.float32).reshape(-1, 1)
    img_undist = cv2.undistort(img_bgr, K, dist)
    return img_bgr.copy()



def rectify_mag_plane(img_bgr, rect_cfg: dict):
    """
    매거진 평면 Homography 정면화.

    config 예시:
      rectify:
        use: true
        left_x_px: 0
        right_x_px: 120
        top_y_px: 0
        bottom_y_px: 430   # 생략하면 자동으로 이미지 높이 사용
        dst_w: 220
        dst_h: 430
    """
    if not rect_cfg.get("use", False):
        return img_bgr.copy()

    h, w = img_bgr.shape[:2]

    left_x  = int(rect_cfg.get("left_x_px", 0))
    right_x = int(rect_cfg.get("right_x_px", w - 1))
    top_y   = int(rect_cfg.get("top_y_px", 0))
    bottom_y = int(rect_cfg.get("bottom_y_px", h - 1))

    # 범위 클램프
    left_x  = max(0, min(left_x,  w - 2))
    right_x = max(left_x + 1, min(right_x, w - 1))
    top_y   = max(0, min(top_y,  h - 2))
    bottom_y = max(top_y + 1, min(bottom_y, h - 1))

    dst_w = int(rect_cfg.get("dst_w", right_x - left_x))
    dst_h = int(rect_cfg.get("dst_h", bottom_y - top_y))

    # src: 원본에서 매거진 직사각형
    src = np.float32([
        [left_x,  top_y],
        [right_x, top_y],
        [left_x,  bottom_y],
        [right_x, bottom_y],
    ])

    # dst: 정면화된 직사각형
    dst = np.float32([
        [0,       0],
        [dst_w-1, 0],
        [0,       dst_h-1],
        [dst_w-1, dst_h-1],
    ])

    H = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(img_bgr, H, (dst_w, dst_h))

    return rectified



def rectify_magazine_full(img_bgr, config: dict):
    """
    1) (현재는 noop) undistort
    2) Homography로 매거진 정면화
    """
    cam_cfg = config.get("camera", {})
    rect_cfg = config.get("rectify", {})

    img_undist = undistort_with_intrinsics(img_bgr, cam_cfg)
    img_rect = rectify_mag_plane(img_undist, rect_cfg)

    debug_info = {
        "undist_used": False,                # 지금은 항상 False
        "rectify_used": bool(rect_cfg.get("use", False)),
    }
    return img_rect, debug_info

