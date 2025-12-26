import argparse, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb.row_y_detection import detect_pcb_rows_tilt_scan, estimate_angles_and_draw

def main():
    ap = argparse.ArgumentParser(description="04: Tilt estimation from y_chain")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mm_per_px", type=float, default=1.0)
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--x1", type=int, default=0)
    ap.add_argument("--pitch_mm", type=float, default=10.0)
    ap.add_argument("--out_image", default="outputs/tilt_overlay.png")
    args = ap.parse_args()

    Path(args.out_image).parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")
    H,W = img.shape[:2]
    x1 = args.x1 if args.x1>0 else W-1

    res = detect_pcb_rows_tilt_scan(img, args.mm_per_px, args.x0, x1, pitch_mm=args.pitch_mm, debug_dir=None)
    y_chain = res.get("y_chain", [])
    infos, vis, _depth_map = estimate_angles_and_draw(img, args.x0, x1, y_chain, mm_per_px=args.mm_per_px)

    cv2.imwrite(args.out_image, vis)

    angles=[it.get("angle_deg") for it in infos if isinstance(it, dict) and "angle_deg" in it]
    ang_mean = float(np.mean(angles)) if angles else None

    print("[OK] saved:", args.out_image)
    print("[ANGLES_DEG]", angles)
    print("[ANGLE_MEAN_DEG]", ang_mean)

if __name__ == "__main__":
    main()
