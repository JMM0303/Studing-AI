# src/geometry/z_from_geometry.py

from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class RowInfo:
    index: int
    y_center_px: float
    tilt_deg: float
    score: float

@dataclass
class CameraConfig:
    fx: float
    fy: float
    cx: float
    cy: float
    R_cam_to_world: np.ndarray  # shape (3, 3)
    t_cam_to_world: np.ndarray  # shape (3, )

@dataclass
class MagazineGeomConfig:
    plane_normal_world: np.ndarray  # shape (3, )
    plane_point_world: np.ndarray   # shape (3, )  # 평면 위 한 점
    z_axis_world: np.ndarray        # z축 방향 단위벡터

def estimate_z_for_rows(
    rows: List[RowInfo],
    cam_cfg: CameraConfig,
    geom_cfg: MagazineGeomConfig,
) -> List[float]:
    """
    각 PCB 행(Row)의 중심 픽셀 위치와 카메라/매거진 기하 파라미터를 이용해
    기하학적으로 Z(mm)를 계산한다.
    """
    fx, fy, cx, cy = cam_cfg.fx, cam_cfg.fy, cam_cfg.cx, cam_cfg.cy
    R = cam_cfg.R_cam_to_world
    t = cam_cfg.t_cam_to_world
    n = geom_cfg.plane_normal_world
    p0 = geom_cfg.plane_point_world
    z_axis = geom_cfg.z_axis_world / np.linalg.norm(geom_cfg.z_axis_world)

    z_list_mm: List[float] = []

    for row in rows:
        # 1) row의 rectified 이미지에서 (x, y) 중심 픽셀 좌표 가져오기
        #    여기서는 이미지 중앙을 x = cx 로 가정 (수평 중앙에 PCB가 있다고 가정)
        u = cx
        v = row.y_center_px

        # 2) 카메라 좌표계에서의 광선 방향 (정규화 전)
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        ray_dir_cam = np.array([x_cam, y_cam, 1.0])
        ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)

        # 3) 월드 좌표계로 변환
        ray_dir_world = R @ ray_dir_cam
        cam_origin_world = t  # 카메라 중심

        # 4) 광선과 매거진 평면의 교점 계산
        denom = np.dot(n, ray_dir_world)
        if abs(denom) < 1e-6:
            # 광선이 평면과 거의 평행한 경우 -> 예외 처리
            z_list_mm.append(float("nan"))
            continue

        t_param = np.dot(n, (p0 - cam_origin_world)) / denom
        hit_point_world = cam_origin_world + t_param * ray_dir_world

        # 5) 월드 좌표계에서 hit_point_world를 z축 방향으로 투영해 Z(mm) 계산
        #    p0를 기준 평면(z=0)로 두고, 그 위에서 z_axis 방향 거리로 정의
        vec = hit_point_world - p0
        z_mm = float(np.dot(vec, z_axis))
        z_list_mm.append(z_mm)

    return z_list_mm
