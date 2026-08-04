"""Cek singkat GPU, ONNX Runtime, OpenCV, Ultralytics, PyAV, dan aset wajah."""

import json
import platform
from pathlib import Path

import cv2
import onnxruntime as ort
import ultralytics


def main():
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "ultralytics": ultralytics.__version__,
        "onnxruntime": ort.__version__,
        "onnx_providers": ort.get_available_providers(),
    }
    try:
        import av

        result["pyav"] = av.__version__
    except Exception as exc:
        result["pyav_error"] = str(exc)
    try:
        import torch

        result.update({
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as exc:
        result["torch_error"] = str(exc)
    try:
        import insightface

        result["insightface"] = insightface.__version__
    except Exception as exc:
        result["insightface_error"] = str(exc)

    assets = Path("face_assets")
    result["face_assets"] = {
        "model": (assets / "models" / "MiniFASNetV2.onnx").is_file(),
        "embeddings": sum(1 for p in (assets / "database" / "embeddings").glob("*/anchor_centroid.npy")),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
