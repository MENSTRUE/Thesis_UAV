"""Helper runtime: PATH CUDA/DLL, device YOLO, dan urutan provider ONNX.

Disarikan dari drone_e99_face_recognition/src/utils.py agar onnxruntime-gpu
menemukan DLL nvidia-* (Windows) tanpa menginstal CUDA Toolkit terpisah.
"""

from __future__ import annotations

import os
import sys


def setup_cuda_paths() -> None:
    """Tambah direktori DLL nvidia-* (site-packages) ke PATH (Windows)."""
    if sys.platform != "win32":
        return
    site_pkg = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages"))
    import glob
    paths = set()
    for pattern in ("nvidia/*/bin", "nvidia/*/bin/x86_64", "nvidia/cu13/bin/x86_64", "nvidia/cudnn/bin"):
        paths.update(glob.glob(os.path.join(site_pkg, pattern)))
    for p in sorted(paths):
        if os.path.isdir(p) and p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")


def resolve_ort_providers(prefer_dml=True):
    """Daftar provider ONNX Runtime urutan prioritas: CUDA > DML > CPU.

    TensorRT sengaja tidak dimasukkan default (membutuhkan library terpisah
    dan menyebabkan LoadLibrary error + fallback yang lambat). CUDA EP sudah
    memaksimalkan GPU untuk model ONNX."""
    import onnxruntime as ort
    order = ["CUDAExecutionProvider"]
    if prefer_dml:
        order.append("DmlExecutionProvider")
    order.append("CPUExecutionProvider")
    available = ort.get_available_providers()
    return [p for p in order if p in available]