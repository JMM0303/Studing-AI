# src/pipeline/depth_midas.py

import cv2
import torch
import numpy as np
from typing import Optional


class MidasDepthEstimator:
    """
    MiDaS_small 기반 단안 깊이 추정기.

    - 입력: img_bgr (H, W, 3), BGR, uint8
    - 출력: depth_map (H, W), float32 (0~1로 정규화된 상대 깊이)
    """

    def __init__(self, model_type: str = "MiDaS_small", device: Optional[str] = None):
        """
        model_type:
            - 기본값 "MiDaS_small" (torch.hub에서 제공하는 엔트리 이름)

        device:
            - "cuda", "cpu", None (None이면 자동 선택)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 1) MiDaS 모델 로드
        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                model_type,
                trust_repo=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MiDaS model '{model_type}' via torch.hub: {e}"
            )

        self.model.to(self.device)
        self.model.eval()

        # 2) MiDaS 전처리 transform 로드
        try:
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MiDaS transforms via torch.hub: {e}"
            )

        # MiDaS_small → small_transform 사용
        self.transform = midas_transforms.small_transform

    @torch.no_grad()
    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        img_bgr: (H, W, 3), BGR, uint8
        return: depth_map (H, W), float32, 0~1 범위
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty image passed to MidasDepthEstimator")

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # MiDaS transform 적용 (resize, normalize, tensor 변환 등)
        input_batch = self.transform(img_rgb).to(self.device)

        # 네트워크 forward
        prediction = self.model(input_batch)

        # 출력 (1, H', W') -> 원본 이미지 크기 (H, W)로 보간
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),              # (1,1,H',W')
            size=img_rgb.shape[:2],               # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)                   # (H, W)

        depth = prediction.cpu().numpy().astype(np.float32)

        # 0~1 범위로 정규화
        d_min, d_max = float(np.nanmin(depth)), float(np.nanmax(depth))
        if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max - d_min < 1e-6:
            depth_norm = np.zeros_like(depth, dtype=np.float32)
        else:
            depth_norm = (depth - d_min) / (d_max - d_min)

        return depth_norm.astype(np.float32)
