"""YOLO deteksi + pose via ONNX Runtime — pengganti ultralytics (tanpa torch).

- yolov8s_512_fp32.onnx     : head mentah (1,84,5376) -> decode xywh + NMS manual
- yolo26s-pose_512_fp32.onnx : output NMS-ready (1,300,57) = xyxy+conf+cls+51 kpt

Semua inferensi lewat onnxruntime (CUDA > DML > CPU), footprint kecil dan
portabel. Algoritma letterbox/NMS/IoU ditulis dari publikasi standar, bukan
salinan kode ultralytics.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

PERSON_CLASS = 0  # COCO
NUM_KEYPOINTS = 17


def letterbox(im: np.ndarray, new_shape: int = 512, color: Tuple[int, int, int] = (114, 114, 114)):
    """Resize + center-pad (replikasi perilaku ultralytics)."""
    h, w = im.shape[:2]
    ratio = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    dw, dh = (new_shape - nw) / 2, (new_shape - nh) / 2
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, ratio, left, top


def scale_coords(arr: np.ndarray, ratio: float, left: float, top: float,
                 width: int, height: int) -> np.ndarray:
    """Mapping koordinat (x, y) dari ruang padded ke frame asli + clip.

    Sama dengan scale_boxes ultralytics: semua koordinat x (kolom 0, 2) dan
    y (kolom 1, 3) dikurangi padding lalu dibagi ratio."""
    out = arr.astype(np.float32).copy()
    if out.ndim == 1:
        out = out[None]
    out[..., [0, 2]] = np.clip((out[..., [0, 2]] - left) / ratio, 0, max(width - 1, 1))
    out[..., [1, 3]] = np.clip((out[..., [1, 3]] - top) / ratio, 0, max(height - 1, 1))
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if b.ndim == 1:
        b = b[None]
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Greedy NMS (bebas kelas — kita hanya memakai kelas person)."""
    order = np.argsort(scores)[::-1]
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        iou = box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][iou <= iou_thr]
    return keep


def _preprocess(frame: np.ndarray, imgsz: int):
    padded, ratio, left, top = letterbox(frame, imgsz)
    blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return blob, ratio, left, top


class _YoloONNX:
    def __init__(self, path: str, providers: Optional[Sequence[str]] = None):
        self.session = ort.InferenceSession(
            str(path), providers=list(providers) if providers else None)
        self.input_name = self.session.get_inputs()[0].name

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def close(self):
        try:
            self.session.run  # noqa: B018
        except Exception:
            pass


class YoloDetector(_YoloONNX):
    """Detektor person: output (1, 4+80, 5376), decode xywh -> xyxy + NMS."""

    def run(self, frame: np.ndarray, imgsz: int, conf: float, iou: float,
            max_det: int) -> List[Tuple[np.ndarray, float]]:
        blob, ratio, left, top = _preprocess(frame, imgsz)
        out = self.session.run(None, {self.input_name: blob})[0][0]  # (84, 5376)
        pred = out.T  # (5376, 84)
        scores = pred[:, 4]  # person class index 0
        keep = scores >= conf
        if not keep.any():
            return []
        pred, scores = pred[keep], scores[keep]
        xywh = pred[:, :4]
        xyxy = np.empty_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        idx = nms(xyxy, scores, iou)[:max_det]
        boxes = scale_coords(xyxy[idx], ratio, left, top, frame.shape[1], frame.shape[0])
        return [(boxes[i], float(scores[idx[i]])) for i in range(len(idx))]


class YoloPose(_YoloONNX):
    """Pose person: output NMS-ready (1, 300, 57) = xyxy+conf+cls+17x3 kpt."""

    def run(self, frame: np.ndarray, imgsz: int, conf: float,
            max_det: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        blob, ratio, left, top = _preprocess(frame, imgsz)
        out = self.session.run(None, {self.input_name: blob})[0][0]  # (300, 57)
        mask = out[:, 4] >= conf
        if not mask.any():
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0, NUM_KEYPOINTS, 3), np.float32)
        pred = out[mask][:max_det]
        boxes = scale_coords(pred[:, :4], ratio, left, top, frame.shape[1], frame.shape[0])
        kpts = pred[:, 6:].reshape(-1, NUM_KEYPOINTS, 3).astype(np.float32).copy()
        kpts[..., :2] = (kpts[..., :2] - np.array([left, top], np.float32)) / ratio
        kpts[..., 0] = np.clip(kpts[..., 0], 0, max(frame.shape[1] - 1, 1))
        kpts[..., 1] = np.clip(kpts[..., 1], 0, max(frame.shape[0] - 1, 1))
        return boxes, pred[:, 4].astype(np.float32), kpts


class IoUTracker:
    """Tracker greedy IoU (tanpa Kalman).

    ponytail: cukup untuk HAR (sequence 30 frame) dengan kamera yang relatif
    diam. Upgrade ke ByteTrack penuh (dengan prediksi Kalman) bila ID sering
    berpindah saat orang saling menutupi.
    """

    def __init__(self, iou_low: float = 0.3, max_misses: int = 30):
        self.iou_low = iou_low
        self.max_misses = max_misses
        self._tracks: List[dict] = []
        self._next_id = 1

    def update(self, boxes: Sequence[np.ndarray]) -> List[int]:
        """Kembalikan ID per box (sejajar urutan `boxes`)."""
        boxes = [np.asarray(b, dtype=np.float32) for b in boxes]
        ids = [0] * len(boxes)
        for track in self._tracks:
            track["misses"] += 1

        pairs = [
            (float(box_iou(track["box"], boxes[bi])), ti, bi)
            for ti, track in enumerate(self._tracks)
            for bi in range(len(boxes))
            if float(box_iou(track["box"], boxes[bi])) >= self.iou_low
        ]
        pairs.sort(key=lambda p: p[0], reverse=True)
        used_t, used_b = set(), set()
        for _, ti, bi in pairs:
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            self._tracks[ti]["box"] = boxes[bi]
            self._tracks[ti]["misses"] = 0
            ids[bi] = self._tracks[ti]["id"]

        for bi in range(len(boxes)):
            if bi in used_b:
                continue
            self._tracks.append({"id": self._next_id, "box": boxes[bi], "misses": 0})
            ids[bi] = self._next_id
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t["misses"] <= self.max_misses]
        return ids


def _self_check() -> None:
    """Cek mapping koordinat: bbox di ruang padded harus kembali tepat ke asli."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    padded, ratio, left, top = letterbox(frame, 512)
    assert padded.shape == (512, 512, 3), padded.shape
    box = np.array([[400.0, 300.0, 100.0, 150.0]])  # xyxy di ruang 512x512
    scaled = scale_coords(box, ratio, left, top, frame.shape[1], frame.shape[0])
    expect = (box - np.array([left, top, left, top], np.float32)) / ratio
    assert np.allclose(scaled, expect, atol=1e-3), (scaled, expect)
    assert scaled[0, 0] >= 0 and scaled[0, 3] <= 720, scaled
    print("yolo_onnx self-check OK (letterbox + scale_coords)")


if __name__ == "__main__":
    _self_check()
