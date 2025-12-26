import argparse, sys, csv
from pathlib import Path
import cv2, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb.row_y_detection import detect_pcb_rows_tilt_scan
from pcb.width_measurement import pcb_width_px_new
from depth.z_from_yx_width import ZFromYXModel

def load_model(path: str|None, a: float, b: float, c: float, w_ref_px: float):
    if path:
        with open(path,"r",encoding="utf-8") as f:
            d=yaml.safe_load(f)
        return ZFromYXModel(
            a=float(d["a"]),
            b=float(d["b"]),
            c=float(d.get("c", 0.0)),
            w_ref_px=float(d.get("w_ref_px", 0.0)),
        )
    return ZFromYXModel(a=a,b=b,c=c,w_ref_px=w_ref_px)

def main():
    ap = argparse.ArgumentParser(description="06: Z(mm) from Y + width_px (skeleton model)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mm_per_px", type=float, default=1.0)
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--x1", type=int, default=0)
    ap.add_argument("--pitch_mm", type=float, default=10.0)

    ap.add_argument("--model_yaml", default=None, help="YAML with a,b,(c,w_ref_px)")
    ap.add_argument("--a", type=float, default=0.0)
    ap.add_argument("--b", type=float, default=0.0)
    ap.add_argument("--c", type=float, default=0.0)
    ap.add_argument("--w_ref_px", type=float, default=0.0)

    ap.add_argument("--out_csv", default="outputs/yz_from_yx_width.csv")
    ap.add_argument("--dbg_dir", default=None, help="폭 측정 디버그 이미지 저장 폴더")
    args = ap.parse_args()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    if args.dbg_dir:
        Path(args.dbg_dir).mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")
    H,W = img.shape[:2]
    x1 = args.x1 if args.x1>0 else W-1

    res = detect_pcb_rows_tilt_scan(img, args.mm_per_px, args.x0, x1, pitch_mm=args.pitch_mm, debug_dir=None)
    y_chain = res.get("y_chain", [])

    widths = pcb_width_px_new(img, y_chain=y_chain, x0=args.x0, x1=x1, dbg_dir=args.dbg_dir)

    model = load_model(args.model_yaml, args.a, args.b, args.c, args.w_ref_px)

    rows=[]
    for idx,(y_px,w_px) in enumerate(zip(y_chain, widths)):
        z_mm = model.predict(float(y_px), float(w_px))
        rows.append((idx, float(y_px), float(y_px)*args.mm_per_px, float(w_px), z_mm))

    with open(args.out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["index","y_px","y_mm","width_px","z_mm_pred"])
        w.writerows(rows)

    print("[OK] saved:", args.out_csv)
    print("[MODEL]", model)

if __name__ == "__main__":
    main()
