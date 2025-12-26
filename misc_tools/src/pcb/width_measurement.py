# src/pipeline/pcb_width_new.py
import cv2
import numpy as np
import os


def _green_score(img_bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    return g / (b + g + r + 1e-6)


def _longest_true_run(mask_1d: np.ndarray):
    best_s, best_l = None, 0
    s = None
    for i, v in enumerate(mask_1d):
        if v:
            if s is None:
                s = i
        else:
            if s is not None:
                l = i - s
                if l > best_l:
                    best_s, best_l = s, l
                s = None
    if s is not None:
        l = len(mask_1d) - s
        if l > best_l:
            best_s, best_l = s, l
    return best_s, best_l


def pcb_width_px_new(
    img_bgr: np.ndarray,
    y_chain,
    x0: int,
    x1: int,
    band_half_px: int = 3,
    smooth_win_px: int = 9,
    min_width_px: int = 20,
    dbg_dir: str | None = None,
):
    """
    각 y_center에서 PCB 폭(px)을 robust하게 구하는 함수.
    """
    H, W, _ = img_bgr.shape
    x0 = max(0, int(x0))
    x1 = min(W - 1, int(x1))
    if x1 <= x0:
        raise ValueError(f"Invalid x-range: x0={x0}, x1={x1}")

    g = _green_score(img_bgr)
    y_chain = np.asarray(y_chain, dtype=np.float32)

    widths = np.full_like(y_chain, np.nan)
    x_lefts = np.full_like(y_chain, np.nan)
    x_rights = np.full_like(y_chain, np.nan)

    if smooth_win_px % 2 == 0:
        smooth_win_px += 1
    kernel = np.ones(smooth_win_px, dtype=np.float32) / smooth_win_px

    if dbg_dir:
        os.makedirs(dbg_dir, exist_ok=True)

    for idx, y in enumerate(y_chain):
        yc = int(round(y))
        y0 = max(0, yc - band_half_px)
        y1 = min(H - 1, yc + band_half_px)

        band = g[y0:y1 + 1, x0:x1 + 1]
        if band.size == 0:
            continue

        prof = band.mean(axis=0).astype(np.float32)
        prof_s = np.convolve(prof, kernel, mode="same")

        pmin, pmax = float(prof_s.min()), float(prof_s.max())
        if pmax - pmin < 1e-6:
            continue

        prof_norm = (prof_s - pmin) / (pmax - pmin)
        prof_8u = np.clip(prof_norm * 255.0, 0, 255).astype(np.uint8)

        # Otsu로 스칼라 임계값만 얻어서 1D 마스크 생성
        thr_val, _ = cv2.threshold(
            prof_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mask = (prof_8u >= thr_val)

        s, l = _longest_true_run(mask)
        if s is None or l < min_width_px:
            continue

        xl = x0 + s
        xr = x0 + s + l - 1

        widths[idx] = float(l)
        x_lefts[idx] = float(xl)
        x_rights[idx] = float(xr)

        if dbg_dir:
            vis = img_bgr[y0:y1 + 1, x0:x1 + 1].copy()
            cv2.line(vis, (xl - x0, 0), (xl - x0, vis.shape[0] - 1), (0, 0, 255), 1)
            cv2.line(vis, (xr - x0, 0), (xr - x0, vis.shape[0] - 1), (0, 0, 255), 1)
            out_path = os.path.join(dbg_dir, f"width_row_{idx:02d}.png")
            cv2.imwrite(out_path, vis)

    return widths, x_lefts, x_rights
