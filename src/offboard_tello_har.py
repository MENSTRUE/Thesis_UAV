"""Off-board HAR untuk video/file/webcam/DJI Tello.

Drone bertugas sebagai platform akuisisi video. Seluruh inferensi berjalan pada
laptop: YOLOv8s -> ByteTrack -> YOLO Pose -> Raw51 -> Body110 -> ONNX.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from body110 import RAW_FEATURE_DIM, prepare_sequence


SEQUENCE_LENGTH = 30
STEP_SIZE = 10
NUM_KEYPOINTS = 17
KEYPOINT_CONF = 0.15
MIN_VALID_KEYPOINTS = 5
TRACK_STALE_FRAMES = 45
CROP_PAD_X = 0.25
CROP_PAD_Y = 0.35
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def parse_args():
    p = argparse.ArgumentParser(description="Off-board HAR UAV/Tello")
    p.add_argument("--source", choices=["tello", "webcam", "video"], default="webcam")
    p.add_argument("--video", type=Path, help="Wajib bila --source video")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--detector", type=Path, default=Path("models/yolov8s.pt"))
    p.add_argument("--pose", type=Path, default=Path("models/yolo26s-pose.pt"))
    p.add_argument("--har", type=Path, default=Path("models/har_window_30_representative.onnx"))
    p.add_argument("--mean", type=Path, default=Path("models/feature_mean.npy"))
    p.add_argument("--std", type=Path, default=Path("models/feature_std.npy"))
    p.add_argument("--mapping", type=Path, default=Path("models/class_mapping.json"))
    p.add_argument("--output", type=Path, default=Path("output/offboard_result.mp4"))
    p.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ...")
    p.add_argument("--detector-imgsz", type=int, default=960)
    p.add_argument("--pose-imgsz", type=int, default=960)
    p.add_argument("--detector-conf", type=float, default=0.15)
    p.add_argument("--pose-conf", type=float, default=0.05)
    p.add_argument("--max-people", type=int, default=10)
    p.add_argument("--no-display", action="store_true")
    p.add_argument(
        "--allow-takeoff",
        action="store_true",
        help="Izinkan tombol T untuk takeoff. Default hanya menerima stream.",
    )
    return p.parse_args()


def check_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path.resolve()}")


def load_mapping(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "class_mapping" in raw:
        raw = raw["class_mapping"]
    mapping = {}
    for key, value in raw.items():
        if str(key).isdigit():
            mapping[int(key)] = str(value)
        elif str(value).isdigit():
            mapping[int(value)] = str(key)
    if not mapping:
        raise ValueError(f"Format class_mapping tidak dikenali: {path}")
    return mapping


def choose_yolo_device(requested):
    if requested != "auto":
        return int(requested) if str(requested).isdigit() else requested
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def make_ort_session(model_path: Path):
    available = ort.get_available_providers()
    preferred = [
        name for name in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        if name in available
    ]
    session = ort.InferenceSession(str(model_path), providers=preferred)
    inputs = {item.name: item for item in session.get_inputs()}
    if "input" not in inputs or "frame_mask" not in inputs:
        raise RuntimeError(f"Input ONNX harus 'input' dan 'frame_mask'; tersedia={list(inputs)}")
    print("ONNX providers:", session.get_providers())
    return session


class OpenCVSource:
    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Sumber video tidak dapat dibuka: {source}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0

    def read(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def takeoff(self):
        print("Takeoff hanya tersedia untuk source=tello")

    def land(self):
        pass

    def close(self):
        self.capture.release()


class TelloSource:
    def __init__(self, allow_takeoff=False):
        try:
            from djitellopy import Tello
        except ImportError as exc:
            raise RuntimeError("Pasang djitellopy: pip install djitellopy") from exc

        self.tello = Tello()
        self.allow_takeoff = allow_takeoff
        self.airborne = False
        self.tello.connect()
        print(f"Tello battery: {self.tello.get_battery()}%")
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
            print("Takeoff diblokir. Jalankan dengan --allow-takeoff bila area sudah aman.")
            return
        if not self.airborne:
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
    if args.source == "video":
        if args.video is None:
            raise ValueError("--video wajib diberikan untuk --source video")
        return OpenCVSource(str(args.video))
    return OpenCVSource(args.camera)


def padded_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1 = max(0, int(round(x1 - bw * CROP_PAD_X)))
    y1 = max(0, int(round(y1 - bh * CROP_PAD_Y)))
    x2 = min(width, int(round(x2 + bw * CROP_PAD_X)))
    y2 = min(height, int(round(y2 + bh * CROP_PAD_Y)))
    return x1, y1, x2, y2


def pose_to_raw51(frame, box, pose_model, args, device):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = padded_box(box, width, height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(RAW_FEATURE_DIM, np.float32), False

    result = pose_model.predict(
        crop,
        imgsz=args.pose_imgsz,
        conf=args.pose_conf,
        iou=0.50,
        classes=[0],
        max_det=5,
        device=device,
        verbose=False,
    )[0]
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return np.zeros(RAW_FEATURE_DIM, np.float32), False

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confs = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    keypoints = result.keypoints.data.detach().cpu().numpy().astype(np.float32)
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0
    )
    best = int(np.argmax(areas * np.maximum(confs, 1e-6)))
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
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def infer_har(session, raw_buffer, mask_buffer, mean, std):
    model_input, mask_input = prepare_sequence(raw_buffer, mask_buffer, mean, std)
    logits = session.run(None, {"input": model_input, "frame_mask": mask_input})[0]
    return softmax(logits)[0].astype(np.float32)


def draw_pose(frame, raw51):
    h, w = frame.shape[:2]
    pose = np.asarray(raw51).reshape(NUM_KEYPOINTS, 3)
    valid = pose[:, 2] > 0
    points = np.stack([pose[:, 0] * w, pose[:, 1] * h], axis=-1).astype(int)
    for a, b in SKELETON_EDGES:
        if valid[a] and valid[b]:
            cv2.line(frame, tuple(points[a]), tuple(points[b]), (0, 255, 255), 2)
    for idx, point in enumerate(points):
        if valid[idx]:
            cv2.circle(frame, tuple(point), 3, (0, 80, 255), -1)


def draw_label(frame, box, text, color=(0, 220, 0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - 25)), (min(frame.shape[1], x1 + 430), y1), color, -1)
    cv2.putText(
        frame, text, (x1 + 4, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (0, 0, 0), 2, cv2.LINE_AA,
    )


def main():
    args = parse_args()
    for path, label in [
        (args.detector, "YOLO detector"), (args.pose, "YOLO Pose"),
        (args.har, "model HAR ONNX"), (args.mean, "feature_mean"),
        (args.std, "feature_std"), (args.mapping, "class_mapping"),
    ]:
        check_file(path, label)

    device = choose_yolo_device(args.device)
    print("YOLO device:", device)
    detector = YOLO(str(args.detector))
    pose_model = YOLO(str(args.pose))
    session = make_ort_session(args.har)
    mean = np.load(args.mean).astype(np.float32).reshape(-1)
    std = np.load(args.std).astype(np.float32).reshape(-1)
    mapping = load_mapping(args.mapping)
    source = make_source(args)

    raw_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
    mask_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
    probability_history = defaultdict(lambda: deque(maxlen=5))
    samples_seen = defaultdict(int)
    last_seen = {}
    last_pose = {}
    last_prediction = {}

    writer = None
    frame_index = 0
    fps_ema = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Kontrol: Q=keluar, T=takeoff (harus --allow-takeoff), L=land")
    try:
        while True:
            started = time.perf_counter()
            frame = source.read()
            if frame is None:
                if args.source == "video":
                    break
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(
                    str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                    max(min(source.fps, 30.0), 1.0), (w, h),
                )

            tracked = detector.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=args.detector_conf,
                iou=0.50,
                imgsz=args.detector_imgsz,
                max_det=args.max_people,
                device=device,
                verbose=False,
            )[0]

            if tracked.boxes is not None and len(tracked.boxes) > 0:
                boxes = tracked.boxes.xyxy.detach().cpu().numpy()
                confs = tracked.boxes.conf.detach().cpu().numpy()
                ids_tensor = tracked.boxes.id
                ids = (
                    ids_tensor.detach().cpu().numpy().astype(int)
                    if ids_tensor is not None
                    else np.arange(len(boxes), dtype=int)
                )

                for box, det_conf, track_id in zip(boxes, confs, ids):
                    track_id = int(track_id)
                    raw51, frame_valid = pose_to_raw51(
                        frame, box, pose_model, args, device
                    )
                    raw_buffers[track_id].append(raw51)
                    mask_buffers[track_id].append(frame_valid)
                    samples_seen[track_id] += 1
                    last_seen[track_id] = frame_index
                    last_pose[track_id] = raw51

                    if (
                        len(raw_buffers[track_id]) == SEQUENCE_LENGTH
                        and samples_seen[track_id] % STEP_SIZE == 0
                    ):
                        probabilities = infer_har(
                            session,
                            np.asarray(raw_buffers[track_id]),
                            np.asarray(mask_buffers[track_id]),
                            mean,
                            std,
                        )
                        probability_history[track_id].append(probabilities)
                        smooth = np.mean(probability_history[track_id], axis=0)
                        class_id = int(np.argmax(smooth))
                        last_prediction[track_id] = (
                            mapping.get(class_id, str(class_id)), float(smooth[class_id])
                        )

                    draw_pose(frame, last_pose[track_id])
                    activity, har_conf = last_prediction.get(track_id, ("collecting", 0.0))
                    valid_count = int(sum(mask_buffers[track_id]))
                    label = (
                        f"T{track_id} | {activity} {har_conf:.2f} | "
                        f"pose {valid_count}/{len(mask_buffers[track_id])} | det {det_conf:.2f}"
                    )
                    draw_label(frame, box, label)

            stale = [
                tid for tid, seen in last_seen.items()
                if frame_index - seen > TRACK_STALE_FRAMES
            ]
            for tid in stale:
                for store in [
                    raw_buffers, mask_buffers, probability_history, samples_seen,
                    last_seen, last_pose, last_prediction,
                ]:
                    store.pop(tid, None)

            elapsed = max(time.perf_counter() - started, 1e-6)
            current_fps = 1.0 / elapsed
            fps_ema = current_fps if fps_ema == 0 else 0.90 * fps_ema + 0.10 * current_fps
            cv2.putText(
                frame,
                f"OFF-BOARD | {fps_ema:.1f} FPS | compute: laptop | source: {args.source}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Off-board HAR UAV", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("t"):
                    source.takeoff()
                if key == ord("l"):
                    source.land()
            frame_index += 1
    finally:
        source.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()

