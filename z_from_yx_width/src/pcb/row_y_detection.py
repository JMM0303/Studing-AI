# src/pipeline/pcb_y_tilt_scan.py

from pathlib import Path
import cv2
import numpy as np


def normalize(v):
    v = v.astype(np.float32)
    m = float(v.max()) if v.size else 1.0
    return (v / m) if m > 1e-9 else np.zeros_like(v, np.float32)


def moving_average(x, k):
    k = max(1, int(k))
    if k == 1:
        return x.astype(np.float32)
    ker = np.ones(k, np.float32) / k
    return np.convolve(x.astype(np.float32), ker, mode="same")


def shear_y(img, tan_theta):
    h, w = img.shape[:2]
    M = np.float32([[1, tan_theta, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def green_profile(roi, hue_lo, hue_hi, sat_lo, win_px):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    Hh, Ss = hsv[:, :, 0], hsv[:, :, 1]
    gmask = ((Hh >= hue_lo) & (Hh <= hue_hi) & (Ss >= sat_lo)).astype(np.float32)
    prof = gmask.mean(axis=1)
    prof = moving_average(prof, win_px)
    return normalize(prof)


def auto_band(gprof, pad_px, min_run_px, exclude_top, exclude_bottom, thr_percentile):
    H = gprof.shape[0]
    a, b = max(0, exclude_top), max(0, H - exclude_bottom)
    if b - a < 5:
        return 1, H - 2
    thr = np.percentile(gprof[a:b], thr_percentile)
    mask = (gprof >= thr).astype(np.uint8)
    runs = []
    s = None
    for i in range(H):
        if mask[i] and s is None:
            s = i
        elif (not mask[i]) and s is not None:
            runs.append((s, i - 1))
            s = None
    if s is not None:
        runs.append((s, H - 1))
    runs = [(u, v) for (u, v) in runs if (v - u + 1) >= min_run_px]
    if not runs:
        return max(1, exclude_top), min(H - 2, H - exclude_bottom)
    u, v = max(runs, key=lambda t: t[1] - t[0])
    return max(1, u - pad_px), min(H - 2, v + pad_px)


def build_projection(roi, hue_lo, hue_hi, sat_lo, alpha_grad, stripes, stripe_band_frac):
    h, w = roi.shape[:2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    Hh, Ss = hsv[:, :, 0], hsv[:, :, 1]
    col_mask = ((Hh >= hue_lo) & (Hh <= hue_hi) & (Ss >= sat_lo)).astype(np.float32)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    g = clahe.apply(gray)
    g = cv2.bilateralFilter(g, 7, 30, 30)

    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    gy = np.abs(gy)

    band = max(10, int(w * stripe_band_frac))
    centers = np.linspace(band // 2, w - band // 2 - 1, max(1, int(stripes))).astype(int)

    acc_g = np.zeros(h, np.float32)
    acc_s = np.zeros(h, np.float32)

    for cx in centers:
        xs0, xs1 = max(0, cx - band // 2), min(w, cx + band // 2)
        wmask = 0.5 + 0.5 * col_mask[:, xs0:xs1]
        acc_g += (gy[:, xs0:xs1] * wmask).sum(axis=1)
        acc_s += (Ss[:, xs0:xs1] * wmask).mean(axis=1)

    proj = alpha_grad * normalize(acc_g) + (1.0 - alpha_grad) * normalize(acc_s)
    return proj


def nms(signal, y0, y1, min_sep, thr):
    peaks = []
    last = -10 ** 9
    for y in range(y0 + 1, y1 - 1):
        if signal[y] >= thr and signal[y] > signal[y - 1] and signal[y] >= signal[y + 1]:
            if y - last >= min_sep:
                peaks.append(y)
                last = y
            else:
                if signal[y] > signal[peaks[-1]]:
                    peaks[-1] = y
                    last = peaks[-1]
    return peaks


def select_chain_by_pitch(cands_y, cands_w, pitch_hint_px, tol_frac=0.25, min_sep_px=12, min_chain=5):
    if not cands_y:
        return []
    if len(cands_y) <= 2:
        idx = np.argsort(-np.asarray(cands_w))
        chain = [int(np.asarray(cands_y)[i]) for i in idx]
        return chain if len(chain) >= min_chain else []
    p = max(min_sep_px, float(pitch_hint_px))
    tol = max(min_sep_px, int(round(p * tol_frac)))
    y = np.array(cands_y, np.int32)
    w = np.array(cands_w, np.float32)
    order = np.argsort(y)
    y, w = y[order], w[order]
    N = len(y)
    dp = w.copy()
    prev = [-1] * N
    length = [1] * N
    for i in range(N):
        for j in range(i):
            d = y[i] - y[j]
            k = int(round(d / p))
            if k < 1:
                continue
            if abs(d - k * p) <= tol and d >= min_sep_px:
                gain = w[i] + dp[j] + 0.2 * w[j]
                if gain > dp[i]:
                    dp[i] = gain
                    prev[i] = j
                    length[i] = length[j] + 1
    best_i, best_score = -1, -1e9
    for i in range(N):
        if length[i] >= min_chain and dp[i] > best_score:
            best_score = dp[i]
            best_i = i
    if best_i == -1:
        return []
    chain = []
    i = best_i
    while i != -1:
        chain.append(int(y[i]))
        i = prev[i]
    return sorted(chain)


def fit_angle_for_line(roi, y_center, half_px, min_col_strength_pct):
    """
    roi       : (H, W, 3) BGR, 매거진 ROI 안쪽
    y_center  : 대략적인 PCB 중심 y
    half_px   : y 주변 ±half_px 범위에서만 검사
    min_col_strength_pct : (더 이상 사용 X, 인터페이스 유지용)
    """

    h, w = roi.shape[:2]

    # 1) HSV에서 green 마스크 생성 (PCB 색)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    Hh, Ss, _ = cv2.split(hsv)
    hue_lo, hue_hi, sat_lo = 25, 105, 60
    green_mask = ((Hh >= hue_lo) & (Hh <= hue_hi) & (Ss >= sat_lo)).astype(np.uint8)

    # 2) y 범위 제한 (라인 주변 좁은 밴드만 사용)
    y0 = max(0, int(round(y_center)) - half_px)
    y1 = min(h - 1, int(round(y_center)) + half_px)

    xs, ys = [], []

    # 3) 각 x 컬럼에서 "가장 위쪽 green 픽셀" 선택
    for x in range(w):
        col = green_mask[y0:y1 + 1, x]
        if col.max() == 0:
            continue  # 이 컬럼에는 PCB가 없음

        # 위에서 아래로 스캔해서 첫 green 위치 사용 → PCB 윗 경계
        rel_y = int(np.argmax(col))    # 0 ~ (y1-y0)
        yy = y0 + rel_y

        xs.append(float(x))
        ys.append(float(yy))

    # 4) 포인트가 너무 적으면 실패
    if len(xs) < 4:
        return float("nan"), 0.0, 0

    xs = np.array(xs, np.float32)
    ys = np.array(ys, np.float32)

    # 5) 1차 직선 피팅
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    y_pred = slope * xs + intercept

    # 6) 잔차 기반 간단 outlier 제거 후 재피팅 (더 안정적으로)
    residuals = np.abs(ys - y_pred)
    inlier_mask = residuals < 3.0  # 3px 이상 벗어나는 점은 버림

    if inlier_mask.sum() >= 4:
        xs_in = xs[inlier_mask]
        ys_in = ys[inlier_mask]
        A_in = np.vstack([xs_in, np.ones_like(xs_in)]).T
        slope, intercept = np.linalg.lstsq(A_in, ys_in, rcond=None)[0]
        xs, ys = xs_in, ys_in
        y_pred = slope * xs + intercept

    mse = float(np.mean((ys - y_pred) ** 2))
    var = float(np.var(ys)) + 1e-6
    r2 = max(0.0, 1.0 - mse / var)
    angle_deg = float(np.degrees(np.arctan(slope)))

    return angle_deg, r2, int(len(xs))


def compute_depth_map(img_bgr: np.ndarray, depth_estimator=None) -> np.ndarray:
    """
    공용 깊이맵 계산 함수.
    - depth_estimator 가 주어지면: 외부 깊이 모델을 호출
    - None 이면: 더미(위→아래 선형) 깊이맵 사용
    """
    if depth_estimator is not None:
        # depth_estimator 는 img_bgr -> (H,W) float32 depth_map 을 반환한다고 가정
        depth = depth_estimator(img_bgr)
        return depth.astype(np.float32)

    # 기본: 더미 깊이맵 (0~1, 위→아래 증가)
    H, W = img_bgr.shape[:2]
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    depth = np.repeat(yy, W, axis=1)
    return depth


def estimate_angles_and_draw(
    img_bgr,
    x0,
    x1,
    y_chain,
    mm_per_px=1.0,
    fit_band_half_px=10,
    fit_col_strength_pct=60.0,
    depth_estimator=None,
    dummy_depth_max_mm=50.0,
    depth_offset_mm=0.0,
):
    """
    y_chain: 후보 행들
    1차: 각 y 주변의 green_ratio + gray std 를 계산해
         'PCB스럽지 않은' 후보(후면/바닥/프레임)를 제거
    2차: 남은 후보에 대해서만 각도 피팅 + 시각화
    """
    if not y_chain:
        return [], img_bgr.copy()

    H, W = img_bgr.shape[:2]
    x0 = max(0, min(x0, W - 2))
    x1 = max(x0 + 1, min(x1, W - 1))

    roi = img_bgr[:, x0:x1]
    vis = img_bgr.copy()

    # --- 공통 준비: HSV / Gray ---
    hsv_full = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    Hh_full, Ss_full, _ = cv2.split(hsv_full)
    gray_full = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # green 마스크 (PCB 색 계열)
    hue_lo, hue_hi, sat_lo = 25, 105, 60
    green_mask_full = ((Hh_full >= hue_lo) & (Hh_full <= hue_hi) & (Ss_full >= sat_lo)).astype(np.float32)

    # y 주변 밴드 폭
    band_half = 8

    # 1차 패스: 각 후보에 대해 green_ratio / texture_std 측정
    stats = []  # (idx, yc, green_ratio, tex_std)
    for idx, yc in enumerate(y_chain):
        yc_i = int(round(yc))
        y0b = max(0, yc_i - band_half)
        y1b = min(roi.shape[0] - 1, yc_i + band_half)

        band_green = green_mask_full[y0b:y1b + 1, :]
        band_gray = gray_full[y0b:y1b + 1, :]

        green_ratio = float(band_green.mean())
        tex_std = float(band_gray.std())

        stats.append((idx, float(yc), green_ratio, tex_std))

    if not stats:
        return [], vis

    # 전체 후보 중 최고값 기준으로 상대 threshold 설정
    max_green = max(s[2] for s in stats)
    max_tex = max(s[3] for s in stats)

    # 너무 낮은 라인은 제거 (후면/바닥 제거용)
    green_rel_thr = 0.21   # max_green 의 21% 미만이면 버림
    tex_rel_thr = 0.40     # max_tex 의 40% 미만이면 버림
    abs_green_min = 0.02   # 절대 최소 green 비율

    # --- 1차: angle 계산만 하고 저장 ---
    raw_infos = []
    for idx, yc, g_ratio, t_std in stats:
        if g_ratio < abs_green_min:
            print(f"DEBUG skip idx={idx}, y={yc:.1f}, green_ratio={g_ratio:.4f} (too low abs)")
            continue
        if max_green > 1e-6 and g_ratio < max_green * green_rel_thr:
            print(f"DEBUG skip idx={idx}, y={yc:.1f}, green_ratio={g_ratio:.4f} (rel)")
            continue
        if max_tex > 1e-6 and t_std < max_tex * tex_rel_thr:
            print(f"DEBUG skip idx={idx}, y={yc:.1f}, tex_std={t_std:.2f} (texture)")
            continue

        angle_deg, r2, npts = fit_angle_for_line(
            roi, yc,
            half_px=fit_band_half_px,
            min_col_strength_pct=fit_col_strength_pct,
        )

        raw_infos.append(
            {
                "index": int(idx),
                "y_px": float(yc),
                "angle_deg": float(angle_deg),
                "fit_r2": float(r2),
                "fit_points": int(npts),
                "green_ratio": float(g_ratio),
                "texture_std": float(t_std),
            }
        )

    if not raw_infos:
        return [], vis

    # ---- 깊이맵 생성 (현재는 더미) ----
    depth_map = compute_depth_map(img_bgr, depth_estimator=depth_estimator)

    H_img, W_img = img_bgr.shape[:2]

    # 슬롯별 y_mm, z_mm 채우기
    for info in raw_infos:
        y_px = info["y_px"]
        info["y_mm"] = float(y_px * mm_per_px)

        if depth_map is None:
            info["z_mm"] = float("nan")
            continue

        yc = int(round(y_px))
        y0d = max(0, yc - fit_band_half_px)
        y1d = min(H_img - 1, yc + fit_band_half_px)

        x0d = max(0, min(x0, W_img - 2))
        x1d = max(x0d + 1, min(x1, W_img - 1))

        depth_patch = depth_map[y0d:y1d + 1, x0d:x1d]

        if depth_patch.size == 0:
            info["z_mm"] = float("nan")
            info["depth_raw"] = float("nan")
        else:
            d_norm = float(depth_patch.mean())  # 0~1
            info["depth_raw"] = d_norm          # ← 0~1 사이의 원시 depth 값 저장

            depth_raw_mm = d_norm * float(dummy_depth_max_mm)
            info["z_mm"] = depth_raw_mm + float(depth_offset_mm)

    # --- 2차: 보라색 얇은 선으로 그리기 + 최종 infos ---
    infos = []
    H_roi, W_roi = roi.shape[:2]

    for info in raw_infos:
        yc = info["y_px"]
        angle_deg = info["angle_deg"]

        if not np.isnan(angle_deg):
            # ROI 내부에서 x 범위는 그대로 사용 (x0~x1)
            xL_line = 0
            xR_line = W_roi - 1

            # 직선 방정식: y = slope * x + intercept
            slope = np.tan(np.radians(angle_deg))
            intercept = yc - slope * (W_roi * 0.5)

            yL = int(round(slope * xL_line + intercept))
            yR = int(round(slope * xR_line + intercept))

            # 전체 이미지 좌표로 보정
            yL = max(0, min(H - 1, yL))
            yR = max(0, min(H - 1, yR))
            pt1 = (x0 + xL_line, yL)
            pt2 = (x0 + xR_line, yR)

            cv2.line(vis, pt1, pt2, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(
                vis,
                f"{info['index']}:{angle_deg:.1f}deg",
                (pt1[0] + 6, max(0, pt1[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        infos.append(info)

    print("PCB tilt + depth results (index, y_px, y_mm, angle_deg, z_mm):")
    for info in infos:
        print(
            f"  idx={info['index']}, "
            f"y_px={info['y_px']:.1f}, "
            f"y_mm={info['y_mm']:.2f}, "
            f"angle={info['angle_deg']:.2f} deg, "
            f"z_mm={info['z_mm']:.2f}"
        )

    return infos, vis, depth_map


def detect_pcb_rows_tilt_scan(
    img_bgr,
    mm_per_px,
    x0,
    x1,
    pitch_mm=10.0,
    hue_lo=25,
    hue_hi=105,
    sat_lo=60,
    alpha_grad=0.70,
    stripes=7,
    stripe_band_frac=0.05,
    smooth_win_px=9,
    thr_percentile=88.0,
    min_sep_px=20,
    min_sep_frac=0.45,
    green_win_px=21,
    green_thr_percentile=70.0,
    band_pad_px=10,
    band_min_px=120,
    exclude_top_px=6,
    exclude_bottom_px=18,
    tilt_scan_deg=20.0,
    tilt_step_deg=1.0,
    edge_guard_px=10,
    min_chain=5,
    # ---- 디버그 추가 인자 (기존 호출에는 영향 없음) ----
    debug_dir=None,
    image_stem: str = "img",
    expected_pcb_count=None,
):
    """
    pcb_y_detect_with_angle.py 의 핵심 로직에서
    - 기울기 스캔 + 피크 검출 + pitch 기반 체인 선택까지 수행.
    반환값: dict {x0, x1, y0_band, y1_band, y_chain(list)}
    """
    H, W = img_bgr.shape[:2]
    x0 = max(0, min(x0, W - 2))
    x1 = max(x0 + 1, min(x1, W - 1))
    roi0 = img_bgr[:, x0:x1]  # 매거진 ROI (x 밴드)

    # ==================================================================
    # [디버그] 1~4단계: 색/에지/row_score/peaks 시각화 (deg=0 기준)
    # ==================================================================
    if debug_dir is not None:
        dbg_dir = Path(debug_dir)
        dbg_dir.mkdir(parents=True, exist_ok=True)

        H_roi, W_roi = roi0.shape[:2]

        # 1) 색 기반: green mask (hue+sat)
        hsv = cv2.cvtColor(roi0, cv2.COLOR_BGR2HSV)
        Hh, Ss, _ = cv2.split(hsv)
        green_mask = ((Hh >= hue_lo) & (Hh <= hue_hi) & (Ss >= sat_lo)).astype(np.float32)

        # 2) 에지 기반: 수평 Scharr dy
        gray_dbg = cv2.cvtColor(roi0, cv2.COLOR_BGR2GRAY)
        gy_dbg = cv2.Scharr(gray_dbg, cv2.CV_32F, 0, 1)
        gy_dbg = np.abs(gy_dbg)

        # (1) 녹색 점수 맵 저장
        green_norm = cv2.normalize(green_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        green_color = cv2.applyColorMap(green_norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(dbg_dir / f"{image_stem}_y1_green_score.png"), green_color)

        # (2) 에지 맵 저장
        edge_norm = cv2.normalize(gy_dbg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edge_color = cv2.applyColorMap(edge_norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(dbg_dir / f"{image_stem}_y2_edge_score.png"), edge_color)

        # (3) y별 row_score(색+에지 결합) 계산
        row_green = green_mask.mean(axis=1)
        row_edge = gy_dbg.mean(axis=1)

        row_green_n = normalize(row_green)
        row_edge_n = normalize(row_edge)

        alpha_row = 1.5
        row_score_dbg = row_edge_n * (1.0 + alpha_row * row_green_n)
        row_score_smooth_dbg = moving_average(row_score_dbg, smooth_win_px)

        # row_score → 1D 그래프 이미지로
        rs = row_score_smooth_dbg.astype(np.float32)
        rs_min, rs_max = float(rs.min()), float(rs.max())
        eps = 1e-6
        W_plot = 200
        img_row = np.zeros((H_roi, W_plot, 3), dtype=np.uint8)

        rs_norm = (rs - rs_min) / (rs_max - rs_min + eps)
        rs_norm = np.clip(rs_norm, 0.0, 1.0)
        rs_len = (rs_norm * (W_plot - 1)).astype(np.int32)

        # threshold (row_score용은 percentile 사용)
        thr_dbg = float(np.percentile(rs, thr_percentile))
        thr_x = int((thr_dbg - rs_min) / (rs_max - rs_min + eps) * (W_plot - 1))
        thr_x = max(0, min(W_plot - 1, thr_x))

        for y in range(H_roi):
            x_end = int(rs_len[y])
            cv2.line(img_row, (0, y), (x_end, y), (255, 255, 255), 1)
        cv2.line(img_row, (thr_x, 0), (thr_x, H_roi - 1), (0, 0, 255), 1)

        cv2.imwrite(str(dbg_dir / f"{image_stem}_y3_row_score.png"), img_row)

        # (4) row_score NMS peak 표시 (디버그용)
        if min_sep_px and min_sep_px > 0:
            min_sep_dbg = int(min_sep_px)
        else:
            min_sep_dbg = max(4, 12)

        peaks_dbg = nms(rs, 0, H_roi, min_sep_dbg, thr_dbg)

        img_peak = img_row.copy()
        for y in peaks_dbg:
            cv2.line(img_peak, (0, y), (W_plot - 1, y), (0, 255, 0), 1)

        cv2.imwrite(str(dbg_dir / f"{image_stem}_y4_row_peaks.png"), img_peak)
    # ==================================================================

    # 1) green 기반 Y 밴드 추정 (기존 로직 그대로)
    gprof = green_profile(roi0, hue_lo, hue_hi, sat_lo, green_win_px)
    y0_auto, y1_auto = auto_band(
        gprof,
        band_pad_px,
        band_min_px,
        exclude_top_px,
        exclude_bottom_px,
        green_thr_percentile,
    )
    guard = int(edge_guard_px)
    y0_auto = min(y1_auto - 2, max(y0_auto + guard, 1))
    y1_auto = max(y0_auto + 2, min(y1_auto - guard, roi0.shape[0] - 2))

    all_y, all_w = [], []
    pitch_px = max(1.0, pitch_mm / max(mm_per_px, 1e-6))

    # 2) tilt 스캔 (기존 튜닝 유지)
    for deg in np.arange(-abs(tilt_scan_deg), abs(tilt_scan_deg) + 1e-6, abs(tilt_step_deg)):
        tan_t = np.tan(np.deg2rad(deg))
        sheared = shear_y(img_bgr, tan_t)
        roi = sheared[:, x0:x1]

        proj = build_projection(
            roi, hue_lo, hue_hi, sat_lo,
            alpha_grad, stripes, stripe_band_frac
        )
        proj = moving_average(proj, smooth_win_px)

        thr = float(np.percentile(proj[y0_auto:y1_auto], thr_percentile))
        if min_sep_px and min_sep_px > 0:
            min_sep = int(min_sep_px)
        else:
            min_sep = max(4, int(min_sep_frac * pitch_px))

        # --- 2-1) 1차 피크 검출 ---
        peaks = nms(proj, y0_auto, y1_auto, min_sep, thr)

        # --- 2-2) 큰 간격 구간에서 약한 피크 보충 ---
        if len(peaks) >= 2:
            peaks_sorted = sorted(peaks)
            diffs = np.diff(peaks_sorted)

            # 로컬 pitch 힌트 추정
            diffs_valid = [d for d in diffs if 8 <= d <= 60]
            if len(diffs_valid) >= 1:
                pitch_hint_local = float(np.median(diffs_valid))
            elif len(diffs) >= 1:
                pitch_hint_local = float(np.median(diffs))
            else:
                pitch_hint_local = float(pitch_px)

            big_gap_factor = 1.6      # pitch 의 1.6배 이상이면 PCB 하나 빠진 것으로 의심
            thr2 = thr * 0.85         # 보충 탐색용 임계값 (조금 낮춤)

            extra_peaks = set()
            for i in range(len(peaks_sorted) - 1):
                y_a, y_b = peaks_sorted[i], peaks_sorted[i + 1]
                gap = y_b - y_a
                if gap < big_gap_factor * pitch_hint_local:
                    continue

                search_y0 = int(y_a + min_sep)
                search_y1 = int(y_b - min_sep)
                if search_y1 <= search_y0:
                    continue

                cand = nms(proj, search_y0, search_y1, min_sep, thr2)
                if cand:
                    # 이 구간에서 가장 강한 후보 하나만 추가
                    best_idx = int(max(cand, key=lambda yy: proj[yy]))
                    extra_peaks.add(best_idx)

            if extra_peaks:
                peaks = sorted(set(peaks) | extra_peaks)

        # --- 2-3) 최종 peaks -> all_y/all_w 누적 ---
        for y in peaks:
            ys0 = max(y0_auto, y - min_sep // 2)
            ys1 = min(y1_auto - 1, y + min_sep // 2)

            loc = float(proj[y])

            gy = cv2.Scharr(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1)
            gy = np.abs(gy)
            gmean = float(gy[ys0:ys1 + 1, :].mean())

            sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
            smean = float(sat[ys0:ys1 + 1, :].mean())

            w = 0.6 * loc + 0.3 * (gmean / (gmean + 1e-6)) + 0.1 * (smean / (smean + 1e-6))
            all_y.append(int(y))
            all_w.append(float(w))

    # 3) pitch 기반 체인 선택 (기존 로직)
    chain = []
    if all_y:
        order = np.argsort(all_y)
        y_sorted = np.array(all_y, np.int32)[order]
        w_sorted = np.array(all_w, np.float32)[order]

        merged_y, merged_w = [], []
        min_sep_basic = max(6, int(min_sep_px or 12))
        for yy, ww in zip(y_sorted, w_sorted):
            if (not merged_y) or abs(yy - merged_y[-1]) >= min_sep_basic:
                merged_y.append(int(yy))
                merged_w.append(float(ww))
            else:
                if ww > merged_w[-1]:
                    merged_y[-1] = int(yy)
                    merged_w[-1] = float(ww)

        diffs = np.diff(merged_y)
        diffs_valid = [d for d in diffs if 8 <= d <= 60]
        if len(diffs_valid) >= 1:
            pitch_hint = np.median(diffs_valid)
        elif len(diffs) >= 1:
            pitch_hint = np.median(diffs)
        else:
            pitch_hint = pitch_px

        chain = select_chain_by_pitch(
            merged_y,
            merged_w,
            pitch_hint,
            tol_frac=min_sep_frac,
            min_sep_px=min_sep_basic,
            min_chain=min_chain,
        )

    return {
        "x0": x0,
        "x1": x1,
        "y0_band": int(y0_auto),
        "y1_band": int(y1_auto),
        "y_chain": chain,
    }
