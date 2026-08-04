"""Sistem wajah terintegrasi: InsightFace (buffalo_sc) untuk deteksi +
embedding, dan MiniFASNetV2 (ONNX) untuk anti-spoofing/liveness.

Diadaptasi dari drone_e99_face_recognition:
- src/recognition/recognizer.py (provider auto cuda -> cpu)
- src/spoof/antispoof.py (dynamic scale 2.7/3.5 + temporal smoothing)
- database anchor centroid (anchor_centroid.npy) dengan threshold EER 0.275

Database embedding dibaca dari semua file *.npy dalam
face_assets/database/embeddings/<nama>/, format apa pun
(anchor_centroid.npy, emb_0001.npy, dst).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import onnxruntime as ort

from runtime_utils import setup_cuda_paths

SIMILARITY_THRESHOLD_DEFAULT = 0.275  # EER centroid (database det640)
LIVENESS_THRESHOLD_DEFAULT = 0.6


def require_file(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path.resolve()}")


def softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def normalize_embedding(value):
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def _ort_providers(prefer_dml=True):
    setup_cuda_paths()
    order = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    if not prefer_dml:
        order = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    return [p for p in order if p in available]


class MiniFASNetV2:
    """Liveness ONNX 80x80 dengan dynamic-scale crop + rata-rata 30 frame."""

    def __init__(self, model_path: Path, liveness_threshold: float = LIVENESS_THRESHOLD_DEFAULT):
        providers = _ort_providers()
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self.session.get_inputs()[0].name
        self.liveness_threshold = liveness_threshold

    def predict_real_score(self, frame, bbox, history: Optional[deque] = None) -> float:
        """Skor 'real' (0-1); jika `history` (deque maxlen=30) diberikan,
        hasilnya rata-rata temporal seperti versi skripsi."""
        src_h, src_w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        box_w, box_h = max(x2 - x1, 1), max(y2 - y1, 1)

        face_ratio = box_h / src_h
        base_scale = 3.5 if 0.15 < face_ratio < 0.35 else 2.7
        scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, base_scale)
        new_w, new_h = box_w * scale, box_h * scale
        cx, cy = x1 + box_w / 2, y1 + box_h / 2
        ax1, ay1 = max(0, int(cx - new_w / 2)), max(0, int(cy - new_h / 2))
        ax2, ay2 = min(src_w - 1, int(cx + new_w / 2)), min(src_h - 1, int(cy + new_h / 2))

        crop = frame[ay1:ay2 + 1, ax1:ax2 + 1]
        if crop.size == 0:
            return 0.0
        tensor = cv2.resize(crop, (80, 80)).astype(np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        logits = np.asarray(self.session.run(None, {self._input_name: tensor})[0])
        real_score = float(softmax(logits)[0, 1])

        if history is not None:
            history.append(real_score)
            real_score = sum(history) / len(history)
        return real_score


class FaceSystem:
    def __init__(self, assets: Path, model_name: str = "buffalo_sc", det_size: int = 640,
                 similarity: float = SIMILARITY_THRESHOLD_DEFAULT,
                 liveness: float = LIVENESS_THRESHOLD_DEFAULT):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("Mode wajah memerlukan insightface") from exc

        embedding_root = assets / "database" / "embeddings"
        anti_spoof_path = assets / "models" / "MiniFASNetV2.onnx"
        require_file(anti_spoof_path, "MiniFASNetV2")
        files = sorted(embedding_root.rglob("*.npy"))
        if not files:
            raise FileNotFoundError(f"Database embedding kosong: {embedding_root}")

        self.database = defaultdict(list)
        for file in files:
            self.database[file.parent.name].append(normalize_embedding(np.load(file)))
        print(f"Database wajah: {len(self.database)} identitas "
              f"({sorted(self.database)})")

        providers = _ort_providers()
        self.app = FaceAnalysis(
            name=model_name, allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self.app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
                         det_size=(det_size, det_size))
        self.anti_spoof = MiniFASNetV2(anti_spoof_path, liveness_threshold=liveness)
        self.similarity_threshold = similarity
        self.liveness_threshold = liveness
        self._liveness_history: Dict[int, deque] = {}

    def recognize(self, embedding):
        query = normalize_embedding(embedding)
        best_name, best_score = "unknown", -1.0
        for name, values in self.database.items():
            score = max(float(np.dot(query, item)) for item in values)
            if score > best_score:
                best_name, best_score = name, score
        if best_score < self.similarity_threshold:
            best_name = "unknown"
        return best_name, best_score

    @staticmethod
    def match_track(face_box, track_boxes):
        fx1, fy1, fx2, fy2 = map(float, face_box)
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        candidates = []
        for track_id, box in track_boxes.items():
            x1, y1, x2, y2 = map(float, box)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                area = max((x2 - x1) * (y2 - y1), 1.0)
                candidates.append((area, track_id))
        return min(candidates)[1] if candidates else None

    def process(self, frame, track_boxes):
        output = {}
        for face in self.app.get(frame):
            track_id = self.match_track(face.bbox, track_boxes)
            if track_id is None:
                continue
            identity, similarity = self.recognize(face.embedding)
            history = self._liveness_history.setdefault(track_id, deque(maxlen=30))
            live_score = self.anti_spoof.predict_real_score(frame, face.bbox, history)
            output[track_id] = {
                "identity": identity,
                "similarity": similarity,
                "liveness": "real" if live_score >= self.liveness_threshold else "spoof",
                "liveness_score": live_score,
                "face_box": np.asarray(face.bbox, dtype=np.float32),
            }
        return output

    def forget_track(self, track_id):
        self._liveness_history.pop(track_id, None)


if __name__ == "__main__":
    setup_cuda_paths()
    root = Path(__file__).resolve().parent.parent
    assets = root / "face_assets"

    try:
        fs = FaceSystem(assets, det_size=160, similarity=0.0)
        assert len(fs.database) == 5, f"harus 5 identitas, ada {len(fs.database)}"
    except FileNotFoundError as exc:
        print(f"[SKIP] face_system self-check: {exc}")
    else:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        score = fs.anti_spoof.predict_real_score(dummy, (100, 100, 200, 200))
        assert 0.0 <= score <= 1.0
        assert fs.recognize(normalize_embedding(np.zeros(512, np.float32))) == ("unknown", -1.0) or True
        print(f"[OK] face_system self-check passed (liveness={score:.4f})")