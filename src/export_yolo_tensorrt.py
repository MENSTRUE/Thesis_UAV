"""Ekspor model Ultralytics .pt menjadi TensorRT FP16 .engine.

Jalankan di perangkat target karena engine TensorRT tidak selalu portabel antar
GPU/versi TensorRT. Jangan menimpa model .pt asli.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--workspace", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    output = YOLO(str(args.model)).export(
        format="engine",
        imgsz=args.imgsz,
        half=True,
        dynamic=False,
        batch=args.batch,
        workspace=args.workspace,
        simplify=True,
    )
    print("TensorRT engine:", output)


if __name__ == "__main__":
    main()
