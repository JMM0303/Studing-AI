# src/depth/z_from_y_only.py
"""Y 데이터만을 사용한 Z(mm) 추정.

프로젝트 최종 코드에서 사용한 형태:
- 깊이 보정 결과를 YAML로 저장
- z_mm = a*y_px + b (선형)
"""

from pathlib import Path
import yaml

DEFAULT_DEPTH_CALIB_PATH = str(Path("depth_calib.yaml"))


def load_depth_from_y(path: str = DEFAULT_DEPTH_CALIB_PATH):
    """YAML에서 y->z 선형 모델 계수(a,b)를 로드."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    d = data.get("depth_from_y", {}) or {}
    a = float(d.get("a", 0.0))
    b = float(d.get("b", 0.0))
    return a, b


def z_from_y(y_px: float, a: float, b: float) -> float:
    """z_mm = a*y_px + b"""
    return a * float(y_px) + b
