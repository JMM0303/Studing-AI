import argparse, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main():
    ap = argparse.ArgumentParser(description="07: Misc tools")
    sp = ap.add_subparsers(dest="cmd", required=True)

    ap_geom = sp.add_parser("geometry_z", help="기하학 기반 Z(실험용) 계산")
    ap_geom.add_argument("--image", required=True)
    ap_geom.add_argument("--help_only", action="store_true", help="모듈 사용 위치 안내만 출력")

    ap_midas = sp.add_parser("midas_depth", help="MiDaS 기반 상대 깊이맵(실험용)")
    ap_midas.add_argument("--image", required=True)
    ap_midas.add_argument("--out", default="outputs/midas_depth.png")

    args = ap.parse_args()

    if args.cmd == "geometry_z":
        if args.help_only:
            print("geometry_z는 프로젝트에서 대체 Z 계산 실험용 모듈입니다.")
            print("필요 시 src/common/geometry_z.py 또는 src/geometry/z_from_geometry.py를 참고하세요.")
            return
        # 실제 계산은 카메라 파라미터/기하 모델이 필요해서 여기서는 안내만 제공
        print("[WARN] geometry_z는 카메라 파라미터/세팅이 필요합니다. --help_only 사용을 권장합니다.")
        return

    if args.cmd == "midas_depth":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Failed to read image: {args.image}")

        try:
            from common.depth_midas import MidasDepthEstimator
        except Exception as e:
            raise SystemExit(f"MiDaS 모듈 import 실패(추가 의존성 필요): {e}\n"
                             f"requirements에 torch/torchvision/timm 등을 추가하거나, 이 서브커맨드를 사용하지 마세요.")
        est = MidasDepthEstimator(model_type="DPT_SMALL")
        depth = est.predict_depth(img)  # float32
        # normalize to 0..255 for visualization
        d = depth.astype(np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        vis = (d * 255.0).astype(np.uint8)
        cv2.imwrite(args.out, vis)
        print("[OK] saved:", args.out)

if __name__ == "__main__":
    main()
