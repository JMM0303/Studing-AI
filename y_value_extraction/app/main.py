import argparse, sys, csv
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb.row_y_detection import detect_pcb_rows_tilt_scan

def main():
    ap = argparse.ArgumentParser(description="03: Extract Y values (px/mm) from image")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mm_per_px", type=float, default=1.0)
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--x1", type=int, default=0)
    ap.add_argument("--pitch_mm", type=float, default=10.0)
    ap.add_argument("--out_csv", default="outputs/y_values.csv")
    args = ap.parse_args()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")
    H,W = img.shape[:2]
    x1 = args.x1 if args.x1>0 else W-1

    res = detect_pcb_rows_tilt_scan(img, args.mm_per_px, args.x0, x1, pitch_mm=args.pitch_mm, debug_dir=None)
    y_chain = res.get("y_chain", [])

    rows=[]
    for idx,y_px in enumerate(y_chain):
        y_mm = float(y_px) * args.mm_per_px
        rows.append((idx, float(y_px), y_mm))

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["index","y_px","y_mm"])
        w.writerows(rows)

    print("[OK] saved:", args.out_csv)
    print("[Y_CHAIN]", y_chain)

if __name__ == "__main__":
    main()
