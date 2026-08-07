"""YOLO deteksi + pose via ONNX Runtime — tanpa torch.

- yolov8s_512_fp32.onnx       : head mentah (1,84,5376) -> decode xywh + NMS manual
- yolo26s-pose_512_fp32.onnx : output NMS-ready (1,300,57) = xyxy+conf+cls+51 kpt

Semua inferensi model lewat ONNX Runtime. Detector dan pose tetap sama.
Tracker menggunakan ByteTrack ringan:
1) prediksi posisi track dengan Kalman filter,
2) asosiasi deteksi confidence tinggi,
3) asosiasi kedua memakai deteksi confidence rendah,
4) track hilang dipertahankan selama track_buffer frame.

Implementasi tracker tidak mengubah model HAR, Raw51, Body110, CNN-BiLSTM,
model ONNX, atau representasi fitur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # fallback bila scipy belum tersedia
    linear_sum_assignment = None


PERSON_CLASS = 0
NUM_KEYPOINTS = 17


# =====================================================================
# PREPROCESSING YOLO — TETAP
# =====================================================================
def letterbox(
    im: np.ndarray,
    new_shape: int = 512,
    color: Tuple[int, int, int] = (114, 114, 114),
):
    """Resize + center-pad."""
    h, w = im.shape[:2]
    ratio = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    dw, dh = (new_shape - nw) / 2, (new_shape - nh) / 2

    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))

    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, ratio, left, top


def scale_coords(
    arr: np.ndarray,
    ratio: float,
    left: float,
    top: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Mapping koordinat xyxy dari ruang padded ke frame asli."""
    out = np.asarray(arr, dtype=np.float32).copy()

    squeeze = False
    if out.ndim == 1:
        out = out[None]
        squeeze = True

    out[..., [0, 2]] = np.clip(
        (out[..., [0, 2]] - left) / ratio,
        0,
        max(width - 1, 1),
    )
    out[..., [1, 3]] = np.clip(
        (out[..., [1, 3]] - top) / ratio,
        0,
        max(height - 1, 1),
    )

    return out[0] if squeeze else out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU satu box terhadap banyak box."""
    a = np.asarray(a, dtype=np.float32).reshape(4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)

    if len(b) == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(
        b[:, 3] - b[:, 1], 0
    )

    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matriks untuk dua himpunan bbox xyxy."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = (
        np.clip(a[:, 2] - a[:, 0], 0.0, None)
        * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    )
    area_b = (
        np.clip(b[:, 2] - b[:, 0], 0.0, None)
        * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    )

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Greedy NMS; pipeline hanya memakai kelas person."""
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
    blob = (
        padded[:, :, ::-1]
        .transpose(2, 0, 1)[None]
        .astype(np.float32)
        / 255.0
    )
    return blob, ratio, left, top


class _YoloONNX:
    def __init__(
        self,
        path: str,
        providers: Optional[Sequence[str]] = None,
    ):
        self.session = ort.InferenceSession(
            str(path),
            providers=list(providers) if providers else None,
        )
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
    """Detektor person: output (1, 4+80, 5376)."""

    def run(
        self,
        frame: np.ndarray,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
    ) -> List[Tuple[np.ndarray, float]]:
        blob, ratio, left, top = _preprocess(frame, imgsz)
        out = self.session.run(
            None,
            {self.input_name: blob},
        )[0][0]

        pred = out.T
        scores = pred[:, 4]

        keep = scores >= conf
        if not keep.any():
            return []

        pred = pred[keep]
        scores = scores[keep]

        xywh = pred[:, :4]
        xyxy = np.empty_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

        idx = nms(xyxy, scores, iou)[:max_det]
        boxes = scale_coords(
            xyxy[idx],
            ratio,
            left,
            top,
            frame.shape[1],
            frame.shape[0],
        )

        return [
            (boxes[i], float(scores[idx[i]]))
            for i in range(len(idx))
        ]


class YoloPose(_YoloONNX):
    """Pose person: (1, 300, 57) = xyxy+conf+cls+17x3 keypoint."""

    def run(
        self,
        frame: np.ndarray,
        imgsz: int,
        conf: float,
        max_det: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        blob, ratio, left, top = _preprocess(frame, imgsz)
        out = self.session.run(
            None,
            {self.input_name: blob},
        )[0][0]

        mask = out[:, 4] >= conf

        if not mask.any():
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros(
                    (0, NUM_KEYPOINTS, 3),
                    np.float32,
                ),
            )

        pred = out[mask][:max_det]

        boxes = scale_coords(
            pred[:, :4],
            ratio,
            left,
            top,
            frame.shape[1],
            frame.shape[0],
        )

        kpts = (
            pred[:, 6:]
            .reshape(-1, NUM_KEYPOINTS, 3)
            .astype(np.float32)
            .copy()
        )

        kpts[..., :2] = (
            kpts[..., :2]
            - np.array([left, top], np.float32)
        ) / ratio

        kpts[..., 0] = np.clip(
            kpts[..., 0],
            0,
            max(frame.shape[1] - 1, 1),
        )
        kpts[..., 1] = np.clip(
            kpts[..., 1],
            0,
            max(frame.shape[0] - 1, 1),
        )

        return (
            boxes,
            pred[:, 4].astype(np.float32),
            kpts,
        )


# =====================================================================
# BYTETrack — PENGGANTI IoUTracker
# =====================================================================
class KalmanFilterXYAH:
    """Kalman filter 8D: x, y, aspect_ratio, height + kecepatannya."""

    ndim = 4
    dt = 1.0

    def __init__(self):
        self._motion_mat = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt

        self._update_mat = np.eye(
            self.ndim,
            2 * self.ndim,
            dtype=np.float32,
        )

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        measurement = np.asarray(
            measurement,
            dtype=np.float32,
        )

        mean = np.r_[
            measurement,
            np.zeros_like(measurement),
        ].astype(np.float32)

        h = max(float(measurement[3]), 1.0)

        std = np.array(
            [
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                1e-2,
                2 * self._std_weight_position * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                1e-5,
                10 * self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )

        covariance = np.diag(std * std)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        h = max(float(mean[3]), 1.0)

        std_pos = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-2,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )

        std_vel = np.array(
            [
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                1e-5,
                self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )

        motion_cov = np.diag(
            np.r_[std_pos, std_vel] ** 2
        )

        mean = self._motion_mat @ mean
        covariance = (
            self._motion_mat
            @ covariance
            @ self._motion_mat.T
            + motion_cov
        )

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        h = max(float(mean[3]), 1.0)

        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )

        innovation_cov = np.diag(std * std)

        projected_mean = self._update_mat @ mean
        projected_cov = (
            self._update_mat
            @ covariance
            @ self._update_mat.T
            + innovation_cov
        )

        return projected_mean, projected_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ):
        projected_mean, projected_cov = self.project(
            mean,
            covariance,
        )

        cross_cov = (
            covariance
            @ self._update_mat.T
        )

        kalman_gain = np.linalg.solve(
            projected_cov.T,
            cross_cov.T,
        ).T

        innovation = (
            np.asarray(measurement, dtype=np.float32)
            - projected_mean
        )

        new_mean = mean + kalman_gain @ innovation
        new_covariance = (
            covariance
            - kalman_gain
            @ projected_cov
            @ kalman_gain.T
        )

        return (
            new_mean.astype(np.float32),
            new_covariance.astype(np.float32),
        )


def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(
        float,
        np.asarray(box).reshape(4),
    )

    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)

    cx = x1 + w / 2
    cy = y1 + h / 2
    aspect = w / h

    return np.array(
        [cx, cy, aspect, h],
        dtype=np.float32,
    )


def xyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, aspect, h = map(
        float,
        np.asarray(state)[:4],
    )

    h = max(h, 1.0)
    w = max(aspect * h, 1.0)

    return np.array(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
        ],
        dtype=np.float32,
    )


@dataclass
class _Track:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    score: float
    state: str = "tracked"
    lost_frames: int = 0

    @property
    def box(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)


def _linear_assignment(
    cost_matrix: np.ndarray,
    max_cost: float,
):
    """Hungarian bila scipy ada; fallback greedy bila tidak ada."""
    cost_matrix = np.asarray(
        cost_matrix,
        dtype=np.float32,
    )

    n_rows, n_cols = cost_matrix.shape

    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    matches = []

    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost_matrix)

        used_rows = set()
        used_cols = set()

        for r, c in zip(rows, cols):
            if float(cost_matrix[r, c]) <= max_cost:
                matches.append((int(r), int(c)))
                used_rows.add(int(r))
                used_cols.add(int(c))

        unmatched_rows = [
            i for i in range(n_rows)
            if i not in used_rows
        ]
        unmatched_cols = [
            i for i in range(n_cols)
            if i not in used_cols
        ]

        return matches, unmatched_rows, unmatched_cols

    # Fallback tanpa scipy.
    pairs = [
        (float(cost_matrix[r, c]), r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if float(cost_matrix[r, c]) <= max_cost
    ]
    pairs.sort(key=lambda x: x[0])

    used_rows = set()
    used_cols = set()

    for _, r, c in pairs:
        if r in used_rows or c in used_cols:
            continue
        matches.append((r, c))
        used_rows.add(r)
        used_cols.add(c)

    unmatched_rows = [
        i for i in range(n_rows)
        if i not in used_rows
    ]
    unmatched_cols = [
        i for i in range(n_cols)
        if i not in used_cols
    ]

    return matches, unmatched_rows, unmatched_cols


class ByteTrack:
    """ByteTrack ringan untuk bbox person.

    Interface:
        ids = tracker.update(detections)

    `detections` harus berupa:
        [(bbox_xyxy, confidence), ...]

    Return:
        list Track ID sejajar urutan detections.
        Nilai 0 berarti deteksi belum menjadi track aktif.

    Prinsip ByteTrack yang dipakai:
    - deteksi high-confidence untuk asosiasi utama,
    - deteksi low-confidence untuk pemulihan track,
    - Kalman prediction,
    - track baru hanya dari deteksi >= new_track_thresh,
    - lost track dihapus setelah track_buffer frame.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.40,
        track_low_thresh: float = 0.10,
        new_track_thresh: Optional[float] = None,
        track_buffer: int = 20,
        match_thresh: float = 0.80,
        second_match_thresh: float = 0.50,
    ):
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(
            min(track_low_thresh, track_high_thresh)
        )
        self.new_track_thresh = float(
            track_high_thresh
            if new_track_thresh is None
            else new_track_thresh
        )
        self.track_buffer = int(track_buffer)
        self.match_thresh = float(match_thresh)
        self.second_match_thresh = float(
            second_match_thresh
        )

        self.kalman = KalmanFilterXYAH()
        self._tracks: List[_Track] = []
        self._next_id = 1

    @property
    def detector_min_conf(self) -> float:
        """Confidence minimum yang perlu dikeluarkan detector."""
        return self.track_low_thresh

    def _predict_all(self):
        for track in self._tracks:
            track.mean, track.covariance = self.kalman.predict(
                track.mean,
                track.covariance,
            )

    def _cost(
        self,
        track_indices: Sequence[int],
        detection_boxes: np.ndarray,
    ) -> np.ndarray:
        if not track_indices or len(detection_boxes) == 0:
            return np.zeros(
                (len(track_indices), len(detection_boxes)),
                dtype=np.float32,
            )

        track_boxes = np.asarray(
            [self._tracks[i].box for i in track_indices],
            dtype=np.float32,
        )

        return 1.0 - pairwise_iou(
            track_boxes,
            detection_boxes,
        )

    def _apply_matches(
        self,
        matches,
        track_indices,
        detection_indices,
        boxes,
        scores,
        output_ids,
    ):
        matched_tracks = set()
        matched_detections = set()

        for local_t, local_d in matches:
            track_index = track_indices[local_t]
            det_index = detection_indices[local_d]

            track = self._tracks[track_index]

            measurement = xyxy_to_xyah(
                boxes[det_index]
            )

            track.mean, track.covariance = (
                self.kalman.update(
                    track.mean,
                    track.covariance,
                    measurement,
                )
            )

            track.score = float(scores[det_index])
            track.state = "tracked"
            track.lost_frames = 0

            output_ids[det_index] = track.track_id
            matched_tracks.add(track_index)
            matched_detections.add(det_index)

        return matched_tracks, matched_detections

    def update(
        self,
        detections: Sequence[
            Tuple[np.ndarray, float]
        ],
    ) -> List[int]:
        detections = list(detections)

        if not detections:
            self._predict_all()

            for track in self._tracks:
                track.state = "lost"
                track.lost_frames += 1

            self._tracks = [
                t for t in self._tracks
                if t.lost_frames <= self.track_buffer
            ]

            return []

        boxes = np.asarray(
            [d[0] for d in detections],
            dtype=np.float32,
        ).reshape(-1, 4)

        scores = np.asarray(
            [d[1] for d in detections],
            dtype=np.float32,
        ).reshape(-1)

        output_ids = [0] * len(detections)

        high_indices = np.flatnonzero(
            scores >= self.track_high_thresh
        ).tolist()

        low_indices = np.flatnonzero(
            (scores >= self.track_low_thresh)
            & (scores < self.track_high_thresh)
        ).tolist()

        # Prediksi semua track ke frame sekarang.
        previous_states = [
            track.state for track in self._tracks
        ]
        self._predict_all()

        all_track_indices = list(
            range(len(self._tracks))
        )

        # -------------------------------------------------------------
        # Tahap 1: semua track existing vs deteksi high-confidence.
        # -------------------------------------------------------------
        high_boxes = (
            boxes[high_indices]
            if high_indices
            else np.zeros((0, 4), np.float32)
        )

        cost_high = self._cost(
            all_track_indices,
            high_boxes,
        )

        (
            matches_high,
            unmatched_track_local,
            unmatched_high_local,
        ) = _linear_assignment(
            cost_high,
            self.match_thresh,
        )

        matched_track_indices, matched_high_indices = (
            self._apply_matches(
                matches_high,
                all_track_indices,
                high_indices,
                boxes,
                scores,
                output_ids,
            )
        )

        unmatched_track_indices = [
            all_track_indices[i]
            for i in unmatched_track_local
        ]

        unmatched_high_indices = [
            high_indices[i]
            for i in unmatched_high_local
        ]

        # -------------------------------------------------------------
        # Tahap 2: hanya track yang sebelumnya tracked vs deteksi low.
        # Ini adalah inti ByteTrack: deteksi confidence rendah masih
        # boleh mempertahankan identitas objek yang sudah ada.
        # -------------------------------------------------------------
        second_track_indices = [
            idx
            for idx in unmatched_track_indices
            if previous_states[idx] == "tracked"
        ]

        low_boxes = (
            boxes[low_indices]
            if low_indices
            else np.zeros((0, 4), np.float32)
        )

        cost_low = self._cost(
            second_track_indices,
            low_boxes,
        )

        (
            matches_low,
            unmatched_second_local,
            _,
        ) = _linear_assignment(
            cost_low,
            self.second_match_thresh,
        )

        matched_second_tracks, _ = (
            self._apply_matches(
                matches_low,
                second_track_indices,
                low_indices,
                boxes,
                scores,
                output_ids,
            )
        )

        # Track yang tidak mendapat pasangan menjadi lost.
        matched_all_tracks = (
            matched_track_indices
            | matched_second_tracks
        )

        for track_index, track in enumerate(self._tracks):
            if track_index in matched_all_tracks:
                continue

            track.state = "lost"
            track.lost_frames += 1

        # -------------------------------------------------------------
        # Track baru HANYA dari unmatched high detections.
        # -------------------------------------------------------------
        for det_index in unmatched_high_indices:
            if scores[det_index] < self.new_track_thresh:
                continue

            mean, covariance = self.kalman.initiate(
                xyxy_to_xyah(boxes[det_index])
            )

            track = _Track(
                track_id=self._next_id,
                mean=mean,
                covariance=covariance,
                score=float(scores[det_index]),
                state="tracked",
                lost_frames=0,
            )

            self._tracks.append(track)
            output_ids[det_index] = self._next_id
            self._next_id += 1

        # Buang track yang hilang terlalu lama.
        self._tracks = [
            t for t in self._tracks
            if t.lost_frames <= self.track_buffer
        ]

        return output_ids


# Alias sengaja TIDAK dipakai di pipeline final.
# Dibiarkan agar kode lama yang mengimpor IoUTracker langsung gagal terlihat,
# bukan diam-diam memakai tracker yang berbeda dari metodologi skripsi.


def _self_check() -> None:
    """Cek mapping koordinat dan tracker sederhana."""
    frame = np.zeros((720, 1280, 3), np.uint8)

    padded, ratio, left, top = letterbox(
        frame,
        512,
    )

    assert padded.shape == (512, 512, 3)

    box = np.array(
        [[400.0, 300.0, 500.0, 450.0]],
        dtype=np.float32,
    )

    scaled = scale_coords(
        box,
        ratio,
        left,
        top,
        frame.shape[1],
        frame.shape[0],
    )

    expect = (
        box
        - np.array(
            [left, top, left, top],
            np.float32,
        )
    ) / ratio

    expect[:, [0, 2]] = np.clip(
        expect[:, [0, 2]],
        0,
        frame.shape[1] - 1,
    )
    expect[:, [1, 3]] = np.clip(
        expect[:, [1, 3]],
        0,
        frame.shape[0] - 1,
    )

    assert np.allclose(
        scaled,
        expect,
        atol=1e-3,
    ), (scaled, expect)

    tracker = ByteTrack(
        track_high_thresh=0.4,
        track_low_thresh=0.1,
        track_buffer=20,
    )

    ids1 = tracker.update(
        [
            (
                np.array(
                    [100, 100, 200, 300],
                    np.float32,
                ),
                0.9,
            )
        ]
    )

    ids2 = tracker.update(
        [
            (
                np.array(
                    [104, 101, 204, 301],
                    np.float32,
                ),
                0.35,
            )
        ]
    )

    assert ids1[0] > 0
    assert ids2[0] == ids1[0]

    print(
        "yolo_onnx self-check OK "
        "(letterbox + scale_coords + ByteTrack)"
    )


if __name__ == "__main__":
    _self_check()
