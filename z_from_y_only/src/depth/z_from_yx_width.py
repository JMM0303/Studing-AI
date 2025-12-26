# src/depth/z_from_yx_width.py
"""Y(행 중심) + X-폭(width_px) 기반 Z(mm) 추정(확장용).

현재 프로젝트의 최종 파이프라인에서는 width_px가
1) PCB 유효 슬롯 필터링(협폭 제거),
2) 'Y만으로 Z를 추정'할 때의 품질 지표
로 활용됩니다.

그러나 발표자료의 핵심 메시지처럼, Y와 폭을 함께 쓰면
Z에 대한 선형 보정항을 추가해 오차를 줄일 수 있습니다.

아래는 그 확장을 위한 최소 골격(모델/추정식)입니다.
"""

from dataclasses import dataclass


@dataclass
class ZFromYXModel:
    """폭 보정이 포함된 선형 모델.

    z_mm = a*y_px + b + c*(w_px - w_ref_px)

    - (a,b): Y 기반 1차 보정(슬롯별 또는 전역)
    - c: 폭 기반 보정 계수
    - w_ref_px: 기준 폭(예: 가장 앞 슬롯 혹은 캘리브레이션 평균)
    """

    a: float
    b: float
    c: float = 0.0
    w_ref_px: float = 0.0

    def predict(self, y_px: float, w_px: float) -> float:
        return (self.a * float(y_px)) + self.b + (self.c * (float(w_px) - self.w_ref_px))
