"""Cek singkat apakah GPU, ONNX Runtime, OpenCV, dan Ultralytics terbaca."""

import json
import platform

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
        import torch
        result.update({
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as exc:
        result["torch_error"] = str(exc)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
