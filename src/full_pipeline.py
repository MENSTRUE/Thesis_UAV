"""Pipeline penuh HAR UAV off-board.

Alur:
Tello/video/webcam -> YOLO person + ByteTrack -> YOLO Pose -> Raw51 ->
Body110 -> CNN-BiLSTM ONNX -> InsightFace -> MiniFASNetV2 -> overlay/report.

Catatan penting: profil ``nano`` adalah profil penghematan komputasi. Profil ini
tidak mengubah model HAR, tetapi menurunkan resolusi inferensi dan frekuensi
pose/wajah. Ukur akurasi dan FPS kembali sebelum membuat klaim penelitian.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from body110 import MODEL_FEATURE_DIM, RAW_FEATURE_DIM, prepare_sequence


SEQUENCE_LENGTH = 30
STEP_SIZE = 10
NUM_KEYPOINTS = 17
KEYPOINT_CONF = 0.15
MIN_VALID_KEYPOINTS = 5
TRACK_STALE_FRAMES = 45
CROP_PAD_X = 0.25
CROP_PAD_Y = 0.35
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]


@dataclass(frozen=True)
class RuntimeProfile:
    detector_imgsz: int
    pose_imgsz: int
    pose_interval: int
    face_interval: int
    max_people: int


PROFILES = {
    # Acuan kualitas/hasil sebelum optimasi.
    "quality": RuntimeProfile(960, 960, 1, 3, 10),
    # Titik awal realistis untuk laptop RTX.
    "laptop": RuntimeProfile(640, 640, 1, 5, 10),
    # Titik awal Jetson Nano lama. Bukan jaminan 25-30 FPS.
    "nano": RuntimeProfile(512, 512, 2, 10, 1),
    # Titik awal Jetson Orin Nano.
    "orin": RuntimeProfile(640, 640, 1, 5, 5),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline penuh HAR UAV off-board")
    parser.add_argument("--source", choices=["video", "webcam", "tello"], default="video")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="laptop")
    parser.add_argument("--detector", type=Path, default=Path("models/yolov8s.pt"))
    parser.add_argument("--pose", type=Path, default=Path("models/yolo26s-pose.pt"))
    parser.add_argument("--har", type=Path, default=Path("models/har_window_30_representative.onnx"))
    parser.add_argument("--mean", type=Path, default=Path("models/feature_mean.npy"))
    parser.add_argument("--std", type=Path, default=Path("models/feature_std.npy"))
    parser.add_argument("--mapping", type=Path, default=Path("models/class_mapping.json"))
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ...")
    parser.add_argument("--detector-conf", type=float, default=0.15)
    parser.add_argument("--pose-conf", type=float, default=0.05)
    parser.add_argument("--detector-imgsz", type=int)
    parser.add_argument("--pose-imgsz", type=int)
    parser.add_argument("--pose-interval", type=int)
    parser.add_argument("--face-interval", type=int)
    parser.add_argument("--max-people", type=int)
    parser.add_argument("--enable-face", action="store_true")
    parser.add_argument("--face-assets", type=Path, default=Path("face_assets"))
    parser.add_argument("--face-model", default="buffalo_sc")
    parser.add_argument("--face-det-size", type=int, default=640)
    parser.add_argument("--face-threshold", type=float, default=0.39)
    parser.add_argument("--liveness-threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path, default=Path("output/full_pipeline.mp4"))
    parser.add_argument("--report", type=Path, default=Path("output/benchmark.json"))
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--allow-takeoff", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 berarti sampai sumber habis")
    return parser.parse_args()


def require_file(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path.resolve()}")


def load_mapping(path: Path) -> Dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "class_mapping" in raw:
        raw = raw["class_mapping"]
    result = {}
    for key, value in raw.items():
        if str(key).isdigit():
            result[int(key)] = str(value)
        elif str(value).isdigit():
            result[int(value)] = str(key)
    if not result:
        raise ValueError(f"Format class mapping tidak dikenali: {path}")
    return result


def choose_yolo_device(requested):
    if requested != "auto":
        return int(requested) if str(requested).isdigit() else requested
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def make_ort_session(path: Path):
    available = ort.get_available_providers()
    order = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    providers = [provider for provider in order if provider in available]
    session = ort.InferenceSession(str(path), providers=providers)
    names = [item.name for item in session.get_inputs()]
    if "input" not in names or "frame_mask" not in names:
        raise RuntimeError(f"Input ONNX wajib input dan frame_mask; ditemukan {names}")
    return session


class OpenCVSource:
    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Sumber tidak dapat dibuka: {source}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0

    def read(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def takeoff(self):
        print("Takeoff hanya tersedia pada --source tello")

    def land(self):
        return None

    def close(self):
        self.capture.release()


class TelloSource:
    def __init__(self, allow_takeoff=False):
        try:
            from djitellopy import Tello
        except ImportError as exc:
            raise RuntimeError("Pasang djitellopy untuk memakai Tello") from exc
        self.tello = Tello()
        self.allow_takeoff = allow_takeoff
        self.airborne = False
        self.tello.connect()
        print(f"Baterai Tello: {self.tello.get_battery()}%")
        try:
            self.tello.streamoff()
        except Exception:
            pass
        self.tello.streamon()
        self.reader = self.tello.get_frame_read()
        self.fps = 30.0
        time.sleep(1.0)

    def read(self):
        frame = self.reader.frame
        if frame is None or frame.size == 0:
            return None
        return frame.copy()

    def takeoff(self):
        if not self.allow_takeoff:
            print("Takeoff diblokir; tambahkan --allow-takeoff setelah area aman")
        elif not self.airborne:
            self.tello.takeoff()
            self.airborne = True

    def land(self):
        if self.airborne:
            self.tello.land()
            self.airborne = False

    def close(self):
        try:
            self.land()
        finally:
            try:
                self.tello.streamoff()
            except Exception:
                pass
            self.tello.end()


def make_source(args):
    if args.source == "tello":
        return TelloSource(args.allow_takeoff)
    if args.source == "webcam":
        return OpenCVSource(args.camera)
    if args.video is None:
        raise ValueError("--video wajib diberikan ketika --source video")
    return OpenCVSource(str(args.video))


def padded_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    return (
        max(0, int(x1 - bw * CROP_PAD_X)),
        max(0, int(y1 - bh * CROP_PAD_Y)),
        min(width, int(x2 + bw * CROP_PAD_X)),
        min(height, int(y2 + bh * CROP_PAD_Y)),
    )


def pose_to_raw51(frame, box, pose_model, pose_imgsz, pose_conf, device):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = padded_box(box, width, height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(RAW_FEATURE_DIM, np.float32), False
    result = pose_model.predict(
        crop, imgsz=pose_imgsz, conf=pose_conf, iou=0.50, classes=[0],
        max_det=5, device=device, verbose=False,
    )[0]
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return np.zeros(RAW_FEATURE_DIM, np.float32), False
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    keypoints = result.keypoints.data.detach().cpu().numpy().astype(np.float32)
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    best = int(np.argmax(areas * np.maximum(scores, 1e-6)))
    pose = keypoints[best, :NUM_KEYPOINTS, :3].copy()
    pose[:, 0] = (pose[:, 0] + x1) / max(width, 1)
    pose[:, 1] = (pose[:, 1] + y1) / max(height, 1)
    pose[:, :2] = np.clip(pose[:, :2], 0.0, 1.0)
    valid = pose[:, 2] >= KEYPOINT_CONF
    pose[~valid] = 0.0
    frame_valid = int(valid.sum()) >= MIN_VALID_KEYPOINTS
    if not frame_valid:
        pose[:] = 0.0
    return pose.reshape(RAW_FEATURE_DIM).astype(np.float32), frame_valid


def softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def infer_har(session, raw_buffer, mask_buffer, mean, std):
    model_input, mask = prepare_sequence(raw_buffer, mask_buffer, mean, std)
    logits = session.run(None, {"input": model_input, "frame_mask": mask})[0]
    return softmax(logits)[0].astype(np.float32)


def normalize_embedding(value):
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    return value / max(float(np.linalg.norm(value)), 1e-8)


class MiniFASNetV2:
    def __init__(self, model_path: Path):
        self.session = make_single_input_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict_real_score(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        scale = min((h - 1) / bh, (w - 1) / bw, 3.0)
        cx, cy = x1 + bw / 2, y1 + bh / 2
        nw, nh = bw * scale, bh * scale
        ax1, ay1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
        ax2, ay2 = min(w - 1, int(cx + nw / 2)), min(h - 1, int(cy + nh / 2))
        crop = frame[ay1:ay2 + 1, ax1:ax2 + 1]
        if crop.size == 0:
            return 0.0
        tensor = cv2.resize(crop, (80, 80)).astype(np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        return float(softmax(logits[None, ...])[0, 1])


def make_single_input_session(path: Path):
    available = ort.get_available_providers()
    order = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [item for item in order if item in available]
    return ort.InferenceSession(str(path), providers=providers)


class FaceSystem:
    def __init__(self, assets: Path, model_name: str, det_size: int, similarity: float, liveness: float):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("Mode wajah memerlukan insightface") from exc
        embedding_root = assets / "database" / "embeddings"
        anti_spoof_path = assets / "models" / "MiniFASNetV2.onnx"
        require_file(anti_spoof_path, "MiniFASNetV2")
        files = sorted(embedding_root.rglob("emb_*.npy"))
        if not files:
            raise FileNotFoundError(f"Database embedding kosong: {embedding_root}")
        self.database = defaultdict(list)
        for file in files:
            self.database[file.parent.name].append(normalize_embedding(np.load(file)))
        available = ort.get_available_providers()
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
        self.app = FaceAnalysis(
            name=model_name, allowed_modules=["detection", "recognition"], providers=providers,
        )
        self.app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(det_size, det_size))
        self.anti_spoof = MiniFASNetV2(anti_spoof_path)
        self.similarity_threshold = similarity
        self.liveness_threshold = liveness

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
            live_score = self.anti_spoof.predict_real_score(frame, face.bbox)
            output[track_id] = {
                "identity": identity,
                "similarity": similarity,
                "liveness": "real" if live_score >= self.liveness_threshold else "spoof",
                "liveness_score": live_score,
                "face_box": np.asarray(face.bbox, dtype=np.float32),
            }
        return output


class ModuleTimer:
    def __init__(self):
        self.values = defaultdict(list)

    def record(self, name, seconds):
        self.values[name].append(float(seconds) * 1000.0)

    def summary(self):
        result = {}
        for name, values in self.values.items():
            arr = np.asarray(values, dtype=np.float64)
            result[name] = {
                "count": int(arr.size),
                "mean_ms": float(arr.mean()),
                "p95_ms": float(np.percentile(arr, 95)),
                "max_ms": float(arr.max()),
            }
        return result


def draw_pose(frame, raw51):
    h, w = frame.shape[:2]
    pose = np.asarray(raw51).reshape(NUM_KEYPOINTS, 3)
    valid = pose[:, 2] > 0
    points = np.stack([pose[:, 0] * w, pose[:, 1] * h], axis=-1).astype(int)
    for a, b in SKELETON_EDGES:
        if valid[a] and valid[b]:
            cv2.line(frame, tuple(points[a]), tuple(points[b]), (0, 255, 255), 2)
    for index, point in enumerate(points):
        if valid[index]:
            cv2.circle(frame, tuple(point), 3, (0, 80, 255), -1)


def draw_box_and_text(frame, box, lines):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    y = max(18, y1 - 8)
    for line in reversed(lines):
        cv2.putText(frame, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        y -= 20


def resolved_profile(args):
    base = PROFILES[args.profile]
    return RuntimeProfile(
        args.detector_imgsz or base.detector_imgsz,
        args.pose_imgsz or base.pose_imgsz,
        max(args.pose_interval or base.pose_interval, 1),
        max(args.face_interval or base.face_interval, 1),
        max(args.max_people or base.max_people, 1),
    )


def main():
    args = parse_args()
    profile = resolved_profile(args)
    for path, label in [
        (args.detector, "YOLO detector"), (args.pose, "YOLO Pose"),
        (args.har, "model HAR ONNX"), (args.mean, "feature_mean"),
        (args.std, "feature_std"), (args.mapping, "class_mapping"),
    ]:
        require_file(path, label)

    device = choose_yolo_device(args.device)
    detector = YOLO(str(args.detector))
    pose_model = YOLO(str(args.pose))
    har_session = make_ort_session(args.har)
    mean = np.load(args.mean).astype(np.float32).reshape(-1)
    std = np.load(args.std).astype(np.float32).reshape(-1)
    if mean.shape != (MODEL_FEATURE_DIM,) or std.shape != (MODEL_FEATURE_DIM,):
        raise ValueError(f"Scaler Body110 wajib (110,), ditemukan {mean.shape}/{std.shape}")
    mapping = load_mapping(args.mapping)
    face_system = None
    if args.enable_face:
        face_system = FaceSystem(
            args.face_assets, args.face_model, args.face_det_size,
            args.face_threshold, args.liveness_threshold,
        )
    source = make_source(args)

    print("=" * 72)
    print("ARSITEKTUR : OFF-BOARD (Tello adalah kamera; komputasi di ground station)")
    print("PROFILE    :", args.profile, profile)
    print("YOLO DEVICE:", device)
    print("HAR ORT    :", har_session.get_providers())
    print("FACE       :", "aktif" if face_system else "nonaktif")
    print("=" * 72)

    raw_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
    mask_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
    probability_history = defaultdict(lambda: deque(maxlen=5))
    samples_seen = defaultdict(int)
    last_seen, last_pose, last_prediction, last_face = {}, {}, {}, {}
    timers = ModuleTimer()
    writer = None
    frame_index = 0
    fps_ema = 0.0
    started_all = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            frame_started = time.perf_counter()
            frame = source.read()
            if frame is None:
                if args.source == "video":
                    break
                time.sleep(0.005)
                continue
            h, w = frame.shape[:2]
            if writer is None and not args.no_save_video:
                writer = cv2.VideoWriter(
                    str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                    max(min(source.fps, 30.0), 1.0), (w, h),
                )

            tick = time.perf_counter()
            tracked = detector.track(
                frame, persist=True, tracker=args.tracker, classes=[0],
                conf=args.detector_conf, iou=0.50, imgsz=profile.detector_imgsz,
                max_det=profile.max_people, device=device, verbose=False,
            )[0]
            timers.record("detector_bytetrack", time.perf_counter() - tick)

            current_boxes = {}
            detections = []
            if tracked.boxes is not None and len(tracked.boxes) > 0:
                boxes = tracked.boxes.xyxy.detach().cpu().numpy()
                confs = tracked.boxes.conf.detach().cpu().numpy()
                ids_tensor = tracked.boxes.id
                ids = ids_tensor.detach().cpu().numpy().astype(int) if ids_tensor is not None else np.arange(len(boxes))
                detections = list(zip(boxes, confs, ids))
                current_boxes = {int(track_id): box for box, _, track_id in detections}

            if face_system is not None and frame_index % profile.face_interval == 0:
                tick = time.perf_counter()
                last_face.update(face_system.process(frame, current_boxes))
                timers.record("face_liveness", time.perf_counter() - tick)

            for box, det_conf, track_id in detections:
                track_id = int(track_id)
                run_pose = frame_index % profile.pose_interval == 0 or track_id not in last_pose
                if run_pose:
                    tick = time.perf_counter()
                    raw51, valid = pose_to_raw51(
                        frame, box, pose_model, profile.pose_imgsz,
                        args.pose_conf, device,
                    )
                    timers.record("pose", time.perf_counter() - tick)
                    last_pose[track_id] = raw51
                else:
                    # Zero-order hold untuk profil hemat. Ini menurunkan resolusi gerak
                    # temporal dan harus dievaluasi terhadap profil quality.
                    raw51 = last_pose[track_id].copy()
                    valid = bool(np.count_nonzero(raw51.reshape(17, 3)[:, 2]) >= MIN_VALID_KEYPOINTS)

                raw_buffers[track_id].append(raw51)
                mask_buffers[track_id].append(valid)
                samples_seen[track_id] += 1
                last_seen[track_id] = frame_index
                if len(raw_buffers[track_id]) == SEQUENCE_LENGTH and samples_seen[track_id] % STEP_SIZE == 0:
                    tick = time.perf_counter()
                    probability = infer_har(
                        har_session, np.asarray(raw_buffers[track_id]),
                        np.asarray(mask_buffers[track_id]), mean, std,
                    )
                    timers.record("body110_har", time.perf_counter() - tick)
                    probability_history[track_id].append(probability)
                    smooth = np.mean(probability_history[track_id], axis=0)
                    class_id = int(np.argmax(smooth))
                    last_prediction[track_id] = (mapping.get(class_id, str(class_id)), float(smooth[class_id]))

                draw_pose(frame, raw51)
                activity, activity_score = last_prediction.get(track_id, ("collecting", 0.0))
                lines = [
                    f"T{track_id} | {activity} {activity_score:.2f} | det {float(det_conf):.2f}",
                    f"pose {int(sum(mask_buffers[track_id]))}/{len(mask_buffers[track_id])}",
                ]
                if track_id in last_face:
                    face = last_face[track_id]
                    lines.append(
                        f"{face['identity']} {face['similarity']:.2f} | "
                        f"{face['liveness']} {face['liveness_score']:.2f}"
                    )
                draw_box_and_text(frame, box, lines)

            stale = [track_id for track_id, seen in last_seen.items() if frame_index - seen > TRACK_STALE_FRAMES]
            for track_id in stale:
                for store in [raw_buffers, mask_buffers, probability_history, samples_seen, last_seen, last_pose, last_prediction, last_face]:
                    store.pop(track_id, None)

            elapsed = max(time.perf_counter() - frame_started, 1e-6)
            timers.record("total_frame", elapsed)
            fps = 1.0 / elapsed
            fps_ema = fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * fps
            cv2.putText(
                frame, f"OFF-BOARD | {args.profile} | {fps_ema:.1f} FPS | {args.source}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )
            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Full Off-board HAR UAV", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("t"):
                    source.takeoff()
                if key == ord("l"):
                    source.land()
            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    finally:
        source.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    wall_time = max(time.perf_counter() - started_all, 1e-6)
    report = {
        "architecture": "off-board",
        "source": args.source,
        "profile": args.profile,
        "profile_values": profile.__dict__,
        "frames": frame_index,
        "wall_time_seconds": wall_time,
        "throughput_fps": frame_index / wall_time,
        "yolo_device": str(device),
        "onnx_providers": har_session.get_providers(),
        "face_enabled": bool(face_system),
        "platform": platform.platform(),
        "module_timing": timers.summary(),
        "warning": "FPS adalah hasil perangkat dan konfigurasi ini; bukan spesifikasi tetap model.",
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if writer is not None:
        print("Video :", args.output.resolve())
    print("Report:", args.report.resolve())


if __name__ == "__main__":
    main()
