import argparse, sys, os
from pathlib import Path
import cv2, yaml

# allow "src" package imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magazine.roi_rectify import crop_magazine_roi, compute_homography, rectify_magazine, undistort_if_needed

def load_yaml(path: str|None):
    if not path: return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="01: Magazine ROI + Rectification")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--config", default=None, help="YAML config containing roi + rectify (+camera optional)")
    ap.add_argument("--out_dir", default="outputs", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    cfg = load_yaml(args.config) or {}
    img_u = undistort_if_needed(img, cfg)

    # ROI crop (optional)
    if "roi" in cfg and all(k in cfg["roi"] for k in ("x","y","w","h")):
        img_roi, roi_box = crop_magazine_roi(img_u, cfg)
    else:
        img_roi, roi_box = img_u, (0,0,img_u.shape[1], img_u.shape[0])

    # homography rectify (optional)
    if "rectify" in cfg and "src_pts" in cfg["rectify"]:
        H, dst_size = compute_homography(cfg)
        img_rect = rectify_magazine(img_roi, H, dst_size)
    else:
        img_rect = img_roi

    cv2.imwrite(str(out_dir / "mag_roi.png"), img_roi)
    cv2.imwrite(str(out_dir / "mag_rectified.png"), img_rect)

    print("[OK] saved:")
    print(" -", out_dir / "mag_roi.png")
    print(" -", out_dir / "mag_rectified.png")
    print("[INFO] roi_box=", roi_box)

if __name__ == "__main__":
    main()
