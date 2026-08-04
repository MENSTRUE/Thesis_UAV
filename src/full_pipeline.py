"""Pipeline penuh HAR UAV off-board.

Alur:
Tello/video/webcam -> YOLO person + ByteTrack -> YOLO Pose -> Raw51 ->
Body110 -> CNN-BiLSTM ONNX -> InsightFace -> MiniFASNetV2 -> overlay/report.

Source ``tello`` kini memakai DroneControl dari dji-tello (PyAV low-latency,
keyboard + gamepad, HUD, foto, rekaman). Mode wajah memakai database anchor
centroid (threshold EER 0.275) dari drone_e99_face_recognition.

Catatan: profil ``nano`` adalah profil penghematan komputasi. Profil ini
tidak mengubah model HAR, tetapi menurunkan resolusi inferensi dan frekuensi
pose/wajah.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from body110 import MODEL_FEATURE_DIM, RAW_FEATURE_DIM, prepare_sequence

try:
    from runtime_utils import choose_yolo_device, resolve_ort_providers, setup_cuda_paths
except ImportError:
    from src.runtime_utils import choose_yolo_device, resolve_ort_providers, setup_cuda_paths

try:
    from tello_control import (BATTERY_CRITICAL, TRIM_MAX, TRIM_STEP,
                               DroneControl, InputHandler, SPEED_MODES,
                               VideoHandler, load_config, rc_from_state,
                               save_config)
except ImportError:
    from src.tello_control import (BATTERY_CRITICAL, TRIM_MAX, TRIM_STEP,
                                   DroneControl, InputHandler, SPEED_MODES,
                                   VideoHandler, load_config, rc_from_state,
                                   save_config)


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
    "quality": RuntimeProfile(960, 960, 1, 3, 10),
    "laptop": RuntimeProfile(512, 512, 1, 5, 10),
    "nano": RuntimeProfile(512, 512, 2, 10, 1),
    "orin": RuntimeProfile(640, 640, 1, 5, 5),
}


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Pipeline penuh HAR UAV off-board")
    parser.add_argument("--source", choices=["video", "webcam", "tello"], default="video")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="laptop")
    parser.add_argument("--detector", type=Path,
                        default=Path("models/yolov8s_512_fp32.onnx"))
    parser.add_argument("--pose", type=Path,
                        default=Path("models/yolo26s-pose_512_fp32.onnx"))
    parser.add_argument("--har", type=Path, default=Path("models/har_window_30_representative.onnx"))
    parser.add_argument("--mean", type=Path, default=Path("models/feature_mean.npy"))
    parser.add_argument("--std", type=Path, default=Path("models/feature_std.npy"))
    parser.add_argument("--mapping", type=Path, default=Path("models/pipeline_metadata.json"))
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
    parser.add_argument("--face-threshold", type=float, default=0.275,
                        help="cosine similarity (EER centroid = 0.275)")
    parser.add_argument("--liveness-threshold", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=Path("output/full_pipeline.mp4"))
    parser.add_argument("--report", type=Path, default=Path("output/benchmark.json"))
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--allow-takeoff", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 berarti sampai sumber habis")
    return parser.parse_args(argv)


def require_file(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path.resolve()}")


def load_mapping(path: Path) -> Dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # pipeline_metadata.json memakai array "class_names"
    if isinstance(raw, dict) and isinstance(raw.get("class_names"), list):
        return {int(i): str(name) for i, name in enumerate(raw["class_names"])}
    if isinstance(raw, dict) and "class_mapping" in raw:
        raw = raw["class_mapping"]
    result = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if str(key).isdigit():
                result[int(key)] = str(value)
            elif str(value).isdigit():
                result[int(value)] = str(key)
    elif isinstance(raw, list):
        result = {int(i): str(name) for i, name in enumerate(raw)}
    if not result:
        raise ValueError(f"Format class mapping tidak dikenali: {path}")
    return result


def make_ort_session(path: Path):
    setup_cuda_paths()
    providers = resolve_ort_providers()
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
        pass

    def close(self):
        self.capture.release()

    def process_control(self, key, frame):
        return False

    def drive(self):
        pass

    def battery_check(self):
        pass

    def overlay(self, frame):
        return frame


class DroneControlSource:
    """TelloSource yang menggabungkan driver PyAV low-latency + kontrol manual
    keyboard/gamepad + HUD + foto/record (dari dji-tello)."""

    def __init__(self, allow_takeoff=False):
        self.control = DroneControl(allow_takeoff=allow_takeoff)
        self.inputs = InputHandler()
        self.video = VideoHandler()
        self.trim_lr, self.speed_idx = load_config()
        self.show_grid = False
        self.fps = 30.0
        self._st = None
        self._last_rc = (0, 0, 0, 0)
        self.control.connect()
        if self.inputs.has_gamepad():
            print("[OK] Gamepad terdeteksi")
        else:
            print("[!] Gamepad tidak terdeteksi - keyboard saja")

    def read(self):
        self._frame = self.control.read()
        return self._frame

    def takeoff(self):
        self.control.takeoff()

    def land(self):
        self.control.land()

    def close(self):
        try:
            save_config(self.trim_lr, self.speed_idx)
        finally:
            self.control.close()

    def process_control(self, key, frame):
        """Mengembalikan True bila harus keluar."""
        st = self.inputs.poll(key)
        self._st = st
        if st.quit:
            return True
        if st.switch_mode:
            self.inputs.switch_mode()
        if st.takeoff_land:
            try:
                self.control.toggle_flight()
            except Exception as exc:
                print(f"[!] Takeoff/Land gagal: {exc}")
        if st.emergency_land and self.control.is_flying:
            try:
                self.control.land()
                print("[EMERGENCY] Landed")
            except Exception as exc:
                print(f"[!] Emergency land gagal: {exc}")
        if st.photo and frame is not None and frame.size:
            self.video.capture_photo(frame)
        if st.record_toggle and frame is not None and frame.size:
            self.video.toggle_recording(frame.shape)
        if st.trim_left:
            self.trim_lr = max(self.trim_lr - TRIM_STEP, -TRIM_MAX)
        if st.trim_right:
            self.trim_lr = min(self.trim_lr + TRIM_STEP, TRIM_MAX)
        if st.trim_reset:
            self.trim_lr = 0
        if st.speed_up:
            self.speed_idx = (self.speed_idx + 1) % len(SPEED_MODES)
        if st.speed_down:
            self.speed_idx = (self.speed_idx - 1) % len(SPEED_MODES)
        return False

    def drive(self):
        """Kirim RC tiap frame (hover saat tidak ada input)."""
        st = self._st
        lr, fb, ud, yaw = 0, 0, 0, 0
        if st is not None:
            lr, fb, ud, yaw = rc_from_state(st, SPEED_MODES[self.speed_idx], self.trim_lr)
        self.control.send_rc(lr, fb, ud, yaw)
        self._last_rc = (st.lr if st else 0.0, st.fb if st else 0.0,
                         st.ud if st else 0.0, st.yaw if st else 0.0)

    def battery_check(self):
        if self.control.get_battery() <= BATTERY_CRITICAL and self.control.is_flying:
            try:
                self.control.land()
                print("[AUTO-LAND] Baterai kritis - mendarat")
            except Exception:
                pass

    def overlay(self, frame):
        lr, fb, ud, yaw = self._last_rc
        return self.video.render(
            frame, self.control.get_battery(), self.control.is_flying,
            self.inputs.mode, self.video.recording, self.trim_lr,
            SPEED_MODES[self.speed_idx], self.show_grid,
            lr=lr, fb=fb, ud=ud, yaw=yaw,
        )


def make_source(args):
    if args.source == "tello":
        return DroneControlSource(args.allow_takeoff)
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


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    setup_cuda_paths()
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
        from face_system import FaceSystem
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

            if args.source == "tello":
                source.drive()
                source.battery_check()

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
                    # Zero-order hold untuk profil hemat.
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
                if face_system is not None:
                    face_system.forget_track(track_id)

            elapsed = max(time.perf_counter() - frame_started, 1e-6)
            timers.record("total_frame", elapsed)
            fps = 1.0 / elapsed
            fps_ema = fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * fps
            text_y = 74 if args.source == "tello" else 28
            cv2.putText(
                frame, f"OFF-BOARD | {args.profile} | {fps_ema:.1f} FPS | {args.source}",
                (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )
            if args.source == "tello":
                frame = source.overlay(frame)
            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Full Off-board HAR UAV", frame)
            key = cv2.waitKey(1) & 0xFF
            if args.source == "tello":
                quit_now = source.process_control(key, frame)
                if quit_now:
                    break
            else:
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