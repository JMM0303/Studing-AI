import argparse, sys
from pathlib import Path
import cv2, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb.row_y_detection import detect_pcb_rows_tilt_scan

def main():
    ap = argparse.ArgumentParser(description="02: PCB row/slot detection (y_chain)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mm_per_px", type=float, default=1.0)
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--x1", type=int, default=0, help="0이면 이미지 끝까지")
    ap.add_argument("--pitch_mm", type=float, default=10.0)
    ap.add_argument("--debug_dir", default="outputs/debug", help="디버그 이미지 저장 폴더(비우면 저장 안 함)")
    ap.add_argument("--expected_pcb_count", type=int, default=None)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    H,W = img.shape[:2]
    x1 = args.x1 if args.x1>0 else W-1

    dbg = args.debug_dir if args.debug_dir else None
    if dbg:
        Path(dbg).mkdir(parents=True, exist_ok=True)

    res = detect_pcb_rows_tilt_scan(
        img_bgr=img,
        mm_per_px=args.mm_per_px,
        x0=args.x0,
        x1=x1,
        pitch_mm=args.pitch_mm,
        debug_dir=dbg,
        image_stem=Path(args.image).stem,
        expected_pcb_count=args.expected_pcb_count,
    )
    print("[RESULT] tilt-scan:", res)

if __name__ == "__main__":
    main()
