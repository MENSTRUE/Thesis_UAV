"""Pipeline penuh HAR UAV off-board.

Alur:
Tello/video/webcam -> YOLO person -> ByteTrack -> YOLO Pose -> Raw51 ->
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
from typing import Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from body110 import MODEL_FEATURE_DIM, RAW_FEATURE_DIM, prepare_sequence

try:
    from runtime_utils import resolve_ort_providers, setup_cuda_paths
except ImportError:
    from src.runtime_utils import resolve_ort_providers, setup_cuda_paths

try:
    from yolo_onnx import ByteTrack, YoloDetector, YoloPose
except ImportError:
    from src.yolo_onnx import ByteTrack, YoloDetector, YoloPose

try:
    from full_reporting import (ACCURACY_COLUMNS, ACTIVITY_COLUMNS,
                                CONFUSION_COLUMNS, IDENTITY_COLUMNS,
                                SECOND_COLUMNS, SEGMENT_STATS_COLUMNS,
                                build_activity_rows, build_identity_rows,
                                build_second_rows, build_segment_stats_row,
                                compute_accuracy, load_ground_truth,
                                write_rows_csv, write_run_manifest)
except ImportError:
    from src.full_reporting import (ACCURACY_COLUMNS, ACTIVITY_COLUMNS,
                                    CONFUSION_COLUMNS, IDENTITY_COLUMNS,
                                    SECOND_COLUMNS, SEGMENT_STATS_COLUMNS,
                                    build_activity_rows, build_identity_rows,
                                    build_second_rows, build_segment_stats_row,
                                    compute_accuracy, load_ground_truth,
                                    write_rows_csv, write_run_manifest)

try:
    from tello_control import (BATTERY_CRITICAL, TRIM_MAX, TRIM_STEP,
                               DroneControl, H264Mp4Writer, InputHandler,
                               SPEED_MODES, VideoHandler, load_config,
                               rc_from_state, save_config)
except ImportError:
    from src.tello_control import (BATTERY_CRITICAL, TRIM_MAX, TRIM_STEP,
                                   DroneControl, H264Mp4Writer, InputHandler,
                                   SPEED_MODES, VideoHandler, load_config,
                                   rc_from_state, save_config)

try:
    from track_reid import ReIdFaceSystem, TrackReId
except ImportError:
    from src.track_reid import ReIdFaceSystem, TrackReId


SEQUENCE_LENGTH = 30
HAR_MIN_VALID_FRAMES = 20
HAR_UPDATE_INTERVAL = 4
NUM_KEYPOINTS = 17
KEYPOINT_CONF = 0.15
MIN_VALID_KEYPOINTS = 5
MAX_TRACK_GAP = 2
TRACK_STALE_FRAMES = 20
CROP_PAD_X = 0.25
CROP_PAD_Y = 0.35
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]

DEFAULT_BOX_COLOR = (0, 220, 0)
FACE_BOX_COLORS = {
    ("spoof", False): (0, 0, 255),      # spoof + unknown -> merah
    ("real", False): (0, 255, 255),     # real + unknown -> kuning
    ("spoof", True): (0, 165, 255),     # spoof + enrolled -> oranye
    ("real", True): DEFAULT_BOX_COLOR,  # real + enrolled -> hijau
}


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
    parser.add_argument("--detector-conf", type=float, default=0.40)
    # confidence nya naikin kalau bisa
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
    parser.add_argument("--reid-cos", type=float, default=0.25,
                        help="cosine similarity face embedding utk match subject aktif")
    parser.add_argument("--reid-iou", type=float, default=0.35,
                        help="IoU bbox utk match subject aktif")
    parser.add_argument("--reid-retired-cos", type=float, default=0.30,
                        help="cosine utk re-ID subjek yang keluar-masuk frame")
    parser.add_argument("--reid-max-missed", type=int, default=30,
                        help="frame tanpa match sebelum subject dipindah ke pool retired")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path video output. Jika kosong, dibuat otomatis per run.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Path report JSON. Jika kosong, dibuat otomatis per run.")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--allow-takeoff", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 berarti sampai sumber habis")
    parser.add_argument("--labels", type=Path, default=None,
                        help="CSV ground truth per detik (second,activity[,segment_index]) "
                             "untuk akurasi + confusion matrix per segmen.")
    return parser.parse_args(argv)


def resolve_output_paths(args):
    """Buat output unik per eksekusi agar video/report lama tidak tertimpa."""
    run_id = time.strftime("%Y%m%d_%H%M%S")
    base_dir = Path("output") / f"run_{run_id}"

    # Hindari tabrakan bila dua run dimulai pada detik yang sama.
    suffix = 1
    while base_dir.exists():
        base_dir = Path("output") / f"run_{run_id}_{suffix:02d}"
        suffix += 1

    if args.output is None and args.report is None:
        base_dir.mkdir(parents=True, exist_ok=True)
        args.output = base_dir / "full_pipeline.mp4"
        args.report = base_dir / "benchmark.json"
    elif args.output is None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.output = args.report.parent / "full_pipeline.mp4"
    elif args.report is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.report = args.output.parent / "benchmark.json"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    return args.output.parent.name



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
            self.video.close()
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
        frame = self.video.render(
            frame, self.control.get_battery(), self.control.is_flying,
            self.inputs.mode, self.video.recording, self.trim_lr,
            SPEED_MODES[self.speed_idx], self.show_grid,
            lr=lr, fb=fb, ud=ud, yaw=yaw,
        )
        # write_frame tidak pernah dipanggil sebelumnya -> rekaman selalu
        # menghasilkan MP4 kosong (258 byte). Rekam frame + HUD yang tampil.
        if self.video.recording:
            self.video.write_frame(frame)
        return frame


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


def pose_to_raw51(frame, box, pose_model, pose_imgsz, pose_conf):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = padded_box(box, width, height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(RAW_FEATURE_DIM, np.float32), False
    boxes, scores, keypoints = pose_model.run(crop, pose_imgsz, pose_conf, max_det=5)
    if len(boxes) == 0:
        return np.zeros(RAW_FEATURE_DIM, np.float32), False
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
    # Window < 30 frame (prediksi pertama dimulai setelah 20 frame valid
    # terkumpul): pad ke belakang dengan nol + frame_mask False, mengikuti
    # konvensi partial-prefix training (x[observed_length:] = 0, mask False).
    raw = np.asarray(raw_buffer, dtype=np.float32)
    mask = np.asarray(mask_buffer, dtype=bool)
    if raw.shape[0] < SEQUENCE_LENGTH:
        padded_raw = np.zeros((SEQUENCE_LENGTH, RAW_FEATURE_DIM), np.float32)
        padded_mask = np.zeros(SEQUENCE_LENGTH, dtype=bool)
        n = raw.shape[0]
        padded_raw[:n] = raw
        padded_mask[:n] = mask
        raw, mask = padded_raw, padded_mask
    model_input, frame_mask = prepare_sequence(raw, mask, mean, std)
    logits = session.run(None, {"input": model_input, "frame_mask": frame_mask})[0]
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


def draw_box_and_text(frame, box, lines, color=DEFAULT_BOX_COLOR):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
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


def record_timing(timers, segment_timers, segment_active, name, seconds):
    """Catat timing sesi penuh + segmen aktif, kembalikan detik untuk CSV."""
    timers.record(name, seconds)
    if segment_active and segment_timers is not None:
        segment_timers.record(name, seconds)
    return seconds


class CsvLogger:
    """Log deteksi per track (detections.csv) + agregat per frame (frames.csv).

    Ditulis streaming per frame dan difinalisasi saat program berhenti.
    Sekaligus mengumpulkan agregat per segmen (dict `agg`) untuk laporan
    per detik/pr segmen/per aktivitas/per identitas (full_reporting.py).
    """

    DETECTION_COLUMNS = [
        "frame_index", "t_run_s", "byte_track_id", "subject_id",
        "x1", "y1", "x2", "y2",
        "det_conf", "activity", "activity_score", "pose_valid",
        "face_identity", "face_similarity", "liveness", "liveness_score",
    ]
    FRAME_COLUMNS = [
        "frame_index", "t_run_s", "n_people", "dominant_activity",
        "fps_ema", "ms_detector", "ms_bytetrack", "ms_pose",
        "ms_body110_har", "ms_face_liveness", "ms_total",
    ]
    MODULE_KEYS = ["detector", "bytetrack", "pose", "body110_har",
                   "face_liveness", "total"]

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self._det_file = open(directory / "detections.csv", "w", newline="", encoding="utf-8")
        self._det_writer = csv.writer(self._det_file)
        self._det_writer.writerow(self.DETECTION_COLUMNS)
        self._frame_file = open(directory / "frames.csv", "w", newline="", encoding="utf-8")
        self._frame_writer = csv.writer(self._frame_file)
        self._frame_writer.writerow(self.FRAME_COLUMNS)
        self.agg = None

    def begin_segment(self, segment_index: int = 1):
        self.agg = {
            "segment_index": segment_index,
            "start_t": None, "end_t": None,
            "n_frames": 0, "people_sum": 0, "people_max": 0,
            "fps_samples": [],
            "seconds": {},
            "activity": {},            # {nama: count}
            "activities": {},          # {nama: (count, score_sum, [scores])}
            "faces": {},               # {nama: count}
            "liveness": {"real": 0, "spoof": 0, "unknown": 0},
            "identity_sims": {},       # {nama: [similarities]}
            "pose_valid": 0, "pose_total": 0,
            "n_detections": 0,
            "ms_samples": {m: [] for m in self.MODULE_KEYS},
            "first_t": {}, "last_t": {},
        }

    def end_segment(self) -> dict:
        agg = self.agg
        self.agg = None
        return agg if agg is not None else {}

    def detection(self, frame_index, t_sec, byte_track_id, subject_id, box,
                  det_conf, activity, activity_score, pose_valid, face):
        x1, y1, x2, y2 = map(int, box)
        if face is None:
            identity, similarity, liveness, liveness_score = "", "", "", ""
        else:
            identity, similarity = face["identity"], f"{face['similarity']:.3f}"
            liveness, liveness_score = face["liveness"], f"{face['liveness_score']:.3f}"
        self._det_writer.writerow([
            frame_index, f"{t_sec:.3f}", byte_track_id, subject_id,
            x1, y1, x2, y2,
            f"{float(det_conf):.3f}", activity, f"{float(activity_score):.3f}",
            pose_valid, identity, similarity, liveness, liveness_score,
        ])
        if self.agg is None:
            return
        a = self.agg
        if a["start_t"] is None:
            a["start_t"] = t_sec
        a["end_t"] = t_sec
        a["n_detections"] += 1
        a["activity"][activity] = a["activity"].get(activity, 0) + 1
        acc = a["activities"].setdefault(activity, {"count": 0, "score_sum": 0.0, "scores": []})
        acc["count"] += 1
        acc["score_sum"] += float(activity_score)
        acc["scores"].append(float(activity_score))
        a["pose_valid"] += int(bool(pose_valid))
        a["pose_total"] += 1
        a["first_t"].setdefault(subject_id, t_sec)
        a["last_t"][subject_id] = t_sec

        sec = int(t_sec)
        s = a["seconds"].setdefault(sec, {
            "n_frames": 0, "people_sum": 0, "n_detections": 0,
            "activity": {}, "faces": {}, "ms": {},
        })
        s["n_detections"] += 1
        s["activity"][activity] = s["activity"].get(activity, 0) + 1
        if face is not None:
            ident = face["identity"]
            a["faces"][ident] = a["faces"].get(ident, 0) + 1
            s["faces"][ident] = s["faces"].get(ident, 0) + 1
            a["identity_sims"].setdefault(ident, []).append(float(face["similarity"]))
            if face["liveness"] in a["liveness"]:
                a["liveness"][face["liveness"]] += 1
            else:
                a["liveness"]["unknown"] += 1

    def frame(self, frame_index, t_sec, n_people, dominant_activity,
              fps_ema, module_ms):
        self._frame_writer.writerow([
            frame_index, f"{t_sec:.3f}", n_people, dominant_activity,
            f"{fps_ema:.1f}",
            f"{module_ms['detector']:.1f}", f"{module_ms['bytetrack']:.1f}",
            f"{module_ms['pose']:.1f}", f"{module_ms['body110_har']:.1f}",
f"{module_ms['face_liveness']:.1f}", f"{module_ms['total']:.1f}",
        ])
        if self.agg is None:
            return
        a = self.agg
        if a["start_t"] is None:
            a["start_t"] = t_sec
        a["end_t"] = t_sec
        a["n_frames"] += 1
        a["people_sum"] += n_people
        a["people_max"] = max(a["people_max"], n_people)
        total_ms = float(module_ms["total"])
        if total_ms > 0:
            a["fps_samples"].append(1000.0 / total_ms)
        sec = int(t_sec)
        s = a["seconds"].setdefault(sec, {
            "n_frames": 0, "people_sum": 0, "n_detections": 0,
            "activity": {}, "faces": {}, "ms": {},
        })
        s["n_frames"] += 1
        s["people_sum"] += n_people
        for key in self.MODULE_KEYS:
            value = float(module_ms[key])
            # Modul yang tidak dijalankan frame ini (pose/face interval,
            # tanpa deteksi) menghasilkan 0.0; jangan cemari mean timing.
            if value > 0:
                a["ms_samples"][key].append(value)
                s["ms"].setdefault(key, []).append(value)

    def close(self):
        self._det_file.close()
        self._frame_file.close()


def make_tracker(args):
    """Instance ByteTrack baru dengan konfigurasi eksperimen yang sama."""
    return ByteTrack(
        track_high_thresh=args.detector_conf,
        track_low_thresh=min(0.10, args.detector_conf),
        new_track_thresh=args.detector_conf,
        track_buffer=20,
        match_thresh=0.80,
        second_match_thresh=0.50,
    )


def write_recording_report(report_path, video_path, recording_index,
                           recording_frames, recording_started, profile,
                           tracker, detector, pose_model, har_session,
                           face_system, segment_timers, args):
    """Benchmark JSON per segmen: 1 segmen = 1 percobaan eksperimen."""
    wall_time = max(time.perf_counter() - recording_started, 1e-6)
    report = {
        "run_kind": "tello_segment",
        "recording_index": recording_index,
        "video_output": str(video_path),
        "report_output": str(report_path),
        "architecture": "off-board",
        "source": "tello",
        "profile": args.profile,
        "profile_values": profile.__dict__,
        "tracker": {
            "name": "ByteTrack",
            "track_high_thresh": tracker.track_high_thresh,
            "track_low_thresh": tracker.track_low_thresh,
            "new_track_thresh": tracker.new_track_thresh,
            "track_buffer": tracker.track_buffer,
            "match_thresh": tracker.match_thresh,
            "second_match_thresh": tracker.second_match_thresh,
        },
        "frames": recording_frames,
        "wall_time_seconds": round(float(wall_time), 6),
        "throughput_fps": round(recording_frames / wall_time, 3),
        "yolo_ort": [detector.provider, pose_model.provider],
        "onnx_providers": har_session.get_providers(),
        "face_enabled": bool(face_system),
        "platform": platform.platform(),
        "module_timing": segment_timers.summary() if segment_timers is not None else {},
        "warning": "FPS adalah hasil perangkat dan konfigurasi ini; bukan spesifikasi tetap model.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print("=" * 72)
    print(f"[REC STOP] Segmen #{recording_index:03d} selesai.")
    print("Video :", video_path.resolve())
    print("Report:", report_path.resolve())
    print(json.dumps(report, indent=2))
    print("=" * 72)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    run_id = resolve_output_paths(args)
    setup_cuda_paths()
    profile = resolved_profile(args)
    for path, label in [
        (args.detector, "YOLO detector"), (args.pose, "YOLO Pose"),
        (args.har, "model HAR ONNX"), (args.mean, "feature_mean"),
        (args.std, "feature_std"), (args.mapping, "class_mapping"),
    ]:
        require_file(path, label)

    providers = resolve_ort_providers()
    detector = YoloDetector(str(args.detector), providers)
    pose_model = YoloPose(str(args.pose), providers)
    tracker = make_tracker(args)
    har_session = make_ort_session(args.har)
    mean = np.load(args.mean).astype(np.float32).reshape(-1)
    std = np.load(args.std).astype(np.float32).reshape(-1)
    if mean.shape != (MODEL_FEATURE_DIM,) or std.shape != (MODEL_FEATURE_DIM,):
        raise ValueError(f"Scaler Body110 wajib (110,), ditemukan {mean.shape}/{std.shape}")
    mapping = load_mapping(args.mapping)
    labels = load_ground_truth(args.labels)
    if args.labels is not None:
        require_file(args.labels, "ground truth labels")
    face_system = None
    reid = None
    if args.enable_face:
        face_system = ReIdFaceSystem(
            args.face_assets, args.face_model, args.face_det_size,
            args.face_threshold, args.liveness_threshold,
        )
        reid = TrackReId(
            cos_thresh=args.reid_cos, iou_thresh=args.reid_iou,
            retired_cos=args.reid_retired_cos, max_missed=args.reid_max_missed,
        )
    source = make_source(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    session_dir = args.output.parent
    session_report_path = (
        session_dir / "session_benchmark.json"
        if args.source == "tello"
        else args.report
    )

    print("=" * 72)
    print("ARSITEKTUR : OFF-BOARD (Tello adalah kamera; komputasi di ground station)")
    print("PROFILE    :", args.profile, profile)
    print("YOLO ORT   :", detector.provider, "/", pose_model.provider)
    print("HAR ORT    :", har_session.get_providers())
    print("TRACKER    : ByteTrack | high=", tracker.track_high_thresh,
          "| low=", tracker.track_low_thresh,
          "| buffer=", tracker.track_buffer)
    print("FACE       :", "aktif" if face_system else "nonaktif")
    print("RUN        :", run_id)
    if args.source == "tello":
        print("SESSION OUT:", session_dir)
        print("RECORD     : E = START/STOP segmen (recording_NNN.mp4 + benchmark_NNN.json)")
        print("LOG CSV    :", (session_dir / "detections.csv").resolve())
    else:
        print("VIDEO OUT  :", args.output)
        print("REPORT OUT :", args.report)
    print("=" * 72)

    raw_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
    mask_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
    probability_history = defaultdict(lambda: deque(maxlen=5))
    samples_seen = defaultdict(int)
    last_seen, last_pose, last_prediction, last_face = {}, {}, {}, {}
    timers = ModuleTimer()
    writer = None
    csv_logger = CsvLogger(session_dir)

    # Display ID: subject_id (stabil antar re-ID) yang ditampilkan di
    # overlay/HAR/face/CSV adalah 1, 2, 3, ... Mapping di-reset tiap START
    # rekaman (bersamaan dengan reset tracker).
    display_used = {}
    next_display = 1

    frame_index = 0
    fps_ema = 0.0
    started_all = time.perf_counter()

    # Rekaman multi-segmen Tello: E = START/STOP, tanpa restart program/Tello.
    pipeline_recording = False
    recording_index = 0
    recording_frames = 0
    recording_started = None
    segment_timers = None
    segment_writer = None
    segment_video_path = None
    segment_report_path = None

    # Laporan eksperimen (mirror reports/ project face-recognition):
    # CSV per detik / per segmen / per aktivitas / per identitas + akurasi.
    reports_session_dir = Path("reports") / "sessions" / session_dir.name
    segment_stat_rows = []
    activity_rows = []
    identity_rows = []
    accuracy_rows = []
    confusion_rows = []
    segment_entries = []
    reported_files = []

    def finalize_segment(seg_index, multi):
        """Tutup agregat segmen, tulis CSV laporan, akumulasi baris agregat."""
        agg = csv_logger.end_segment()
        stat_row = build_segment_stats_row(agg, len(display_used), labels)
        segment_stat_rows.append(stat_row)
        activity_rows.extend(build_activity_rows(agg))
        identity_rows.extend(build_identity_rows(agg))
        acc_rows, conf_rows, _, _ = compute_accuracy(agg, labels)
        accuracy_rows.extend(acc_rows)
        confusion_rows.extend(conf_rows)

        suffix = f"_{seg_index:03d}" if multi else ""
        per_second_path = reports_session_dir / f"per_second_{session_dir.name}{suffix}.csv"
        write_rows_csv(per_second_path, build_second_rows(agg), SECOND_COLUMNS)
        reported_files.append(str(per_second_path))

        for rows, cols, kind in [
            (segment_stat_rows, SEGMENT_STATS_COLUMNS, "segment_stats"),
            (activity_rows, ACTIVITY_COLUMNS, "activity_stats"),
            (identity_rows, IDENTITY_COLUMNS, "identity_stats"),
            (accuracy_rows, ACCURACY_COLUMNS, "accuracy"),
            (confusion_rows, CONFUSION_COLUMNS, "confusion"),
        ]:
            if not rows:
                continue
            path = reports_session_dir / f"{kind}_{session_dir.name}.csv"
            write_rows_csv(path, rows, cols)
            reported_files.append(str(path))

        entry = {
            "segment_index": seg_index,
            "frames": agg.get("n_frames", 0),
            "n_detections": agg.get("n_detections", 0),
            "duration_s": round(agg.get("end_t", 0) - agg.get("start_t", 0), 3)
            if agg.get("start_t") else 0,
            "throughput_fps": stat_row.get("throughput_fps", 0),
            "dominant_activity": stat_row.get("dominant_activity", ""),
            "accuracy_pct": stat_row.get("accuracy_pct", ""),
            "detection_ok": bool(agg.get("n_detections", 0) > 0),
        }
        if segment_video_path is not None:
            entry["video"] = str(segment_video_path)
            entry["report"] = str(segment_report_path)
        segment_entries.append(entry)
        return agg

    if args.source != "tello":
        csv_logger.begin_segment(1)

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
            if args.source != "tello" and writer is None and not args.no_save_video:
                writer = H264Mp4Writer(
                    str(args.output), w, h,
                    fps=max(min(source.fps, 30.0), 1.0),
                )

            if args.source == "tello":
                source.drive()
                source.battery_check()

            # Detector dijalankan sampai confidence rendah yang dibutuhkan
            # ByteTrack. Track baru tetap hanya dibuat pada detector_conf.
            tick = time.perf_counter()
            dets = detector.run(
                frame,
                profile.detector_imgsz,
                tracker.detector_min_conf,
                0.50,
                profile.max_people,
            )
            ms_detector = record_timing(timers, segment_timers, pipeline_recording, "detector", time.perf_counter() - tick) * 1000.0

            tick = time.perf_counter()
            ids = tracker.update(dets)
            ms_bytetrack = record_timing(timers, segment_timers, pipeline_recording, "bytetrack", time.perf_counter() - tick) * 1000.0

            current_boxes = {}
            detections = [
                (det[0], det[1], int(track_id))
                for det, track_id in zip(dets, ids)
                if int(track_id) > 0
            ]
            current_boxes = {
                int(track_id): box
                for box, _, track_id in detections
            }

            ms_pose = ms_har = ms_face = 0.0
            if face_system is not None and frame_index % profile.face_interval == 0:
                tick = time.perf_counter()
                last_face.update(face_system.process(frame, current_boxes))
                ms_face = record_timing(timers, segment_timers, pipeline_recording, "face_liveness", time.perf_counter() - tick) * 1000.0

            # Re-ID ByteTrack -> subject_id stabil: track lama yang keluar-masuk
            # frame tetap dianggap subjek yang sama (embedding wajah).
            if reid is not None and current_boxes:
                face_map = {
                    tid: face for tid, face in last_face.items()
                    if tid in current_boxes
                }
                subject_of = reid.update(face_map, current_boxes)
            else:
                subject_of = {tid: tid for tid in current_boxes}

            appended_this_frame = set()
            for box, det_conf, track_id in detections:
                track_id = int(track_id)
                # Satu Track ID hanya boleh append satu pose per frame.
                if track_id in appended_this_frame:
                    continue
                appended_this_frame.add(track_id)

                # Track absen terlalu lama: window HAR-nya rusak (pose tidak
                # berurutan), reset state HAR track ini agar prediksi baru
                # dibangun dari window yang kontinu.
                if track_id in last_seen and frame_index - last_seen[track_id] > MAX_TRACK_GAP:
                    for store in (raw_buffers, mask_buffers, probability_history):
                        store.pop(track_id, None)
                    samples_seen.pop(track_id, None)
                    last_pose.pop(track_id, None)
                    last_prediction.pop(track_id, None)

                # Display/reporting memakai subject_id (stabil antar re-ID);
                # HAR temporal tetap keyed ByteTrack + reset saat gap.
                subject_id = subject_of.get(track_id, track_id)
                if subject_id not in display_used:
                    display_used[subject_id] = next_display
                    next_display += 1
                display = display_used[subject_id]

                run_pose = frame_index % profile.pose_interval == 0 or track_id not in last_pose
                if run_pose:
                    tick = time.perf_counter()
                    raw51, valid = pose_to_raw51(
                        frame, box, pose_model, profile.pose_imgsz,
                        args.pose_conf,
                    )
                    ms_pose = record_timing(timers, segment_timers, pipeline_recording, "pose", time.perf_counter() - tick) * 1000.0
                    last_pose[track_id] = raw51
                else:
                    # Zero-order hold untuk profil hemat.
                    raw51 = last_pose[track_id].copy()
                    valid = bool(np.count_nonzero(raw51.reshape(17, 3)[:, 2]) >= MIN_VALID_KEYPOINTS)

                raw_buffers[track_id].append(raw51)
                mask_buffers[track_id].append(valid)
                samples_seen[track_id] += 1
                last_seen[track_id] = frame_index

                valid_pose_frames = int(sum(mask_buffers[track_id]))
                if valid_pose_frames >= HAR_MIN_VALID_FRAMES and samples_seen[track_id] % HAR_UPDATE_INTERVAL == 0:
                    tick = time.perf_counter()
                    probability = infer_har(
                        har_session, np.asarray(raw_buffers[track_id]),
                        np.asarray(mask_buffers[track_id]), mean, std,
                    )
                    ms_har = record_timing(timers, segment_timers, pipeline_recording, "body110_har", time.perf_counter() - tick) * 1000.0
                    probability_history[track_id].append(probability)
                    smooth = np.mean(probability_history[track_id], axis=0)
                    class_id = int(np.argmax(smooth))
                    last_prediction[track_id] = (mapping.get(class_id, str(class_id)), float(smooth[class_id]))

                draw_pose(frame, raw51)
                prediction_ready = track_id in last_prediction
                activity, activity_score = last_prediction.get(track_id, ("collecting", 0.0))
                if prediction_ready:
                    activity_line = f"ID {display} | {activity.upper()} | {activity_score * 100.0:.1f}%"
                else:
                    activity_line = f"ID {display} | HAR: MENUNGGU {valid_pose_frames}/{HAR_MIN_VALID_FRAMES}"
                lines = [
                    activity_line,
                    f"Pose valid {valid_pose_frames}/{len(mask_buffers[track_id])} | det {float(det_conf) * 100.0:.1f}%",
                ]
                face = last_face.get(track_id)
                if face is not None:
                    lines.append(
                        f"{face['identity']} {face['similarity']:.2f} | "
                        f"{face['liveness']} {face['liveness_score']:.2f}"
                    )
                box_color = DEFAULT_BOX_COLOR
                if face is not None:
                    box_color = FACE_BOX_COLORS.get(
                        (face["liveness"], face["identity"] != "unknown"),
                        DEFAULT_BOX_COLOR,
                    )
                draw_box_and_text(frame, box, lines, box_color)
                csv_logger.detection(
                    frame_index, time.perf_counter() - started_all,
                    track_id, subject_id, box, det_conf, activity,
                    activity_score, valid_pose_frames, last_face.get(track_id),
                )

            stale = [track_id for track_id, seen in last_seen.items() if frame_index - seen > TRACK_STALE_FRAMES]
            for track_id in stale:
                for store in [raw_buffers, mask_buffers, probability_history, samples_seen, last_seen, last_pose, last_prediction, last_face]:
                    store.pop(track_id, None)
                if face_system is not None:
                    face_system.forget_track(track_id)

            elapsed = max(time.perf_counter() - frame_started, 1e-6)
            ms_total = record_timing(timers, segment_timers, pipeline_recording, "total_frame", elapsed) * 1000.0
            fps = 1.0 / elapsed
            fps_ema = fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * fps
            dominant_activity = "collecting"
            best_score = 0.0
            for track_id in current_boxes:
                if track_id in last_prediction:
                    name, score = last_prediction[track_id]
                    if score > best_score:
                        best_score, dominant_activity = score, name
            csv_logger.frame(
                frame_index, time.perf_counter() - started_all,
                len(current_boxes), dominant_activity, fps_ema,
                {"detector": ms_detector, "bytetrack": ms_bytetrack,
                 "pose": ms_pose, "body110_har": ms_har,
                 "face_liveness": ms_face, "total": ms_total},
            )
            text_y = 74 if args.source == "tello" else 28
            cv2.putText(
                frame, f"OFF-BOARD | {args.profile} | {fps_ema:.1f} FPS | {args.source}",
                (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )
            if args.source == "tello":
                frame = source.overlay(frame)
            if writer is not None:
                writer.write(frame)
            if segment_writer is not None:
                segment_writer.write(frame)
                recording_frames += 1
            if not args.no_display:
                cv2.imshow("Full Off-board HAR UAV", frame)
            key = cv2.waitKey(1) & 0xFF
            if args.source == "tello":
                # E = START/STOP segmen. START mereset seluruh state eksperimen
                # (tracker + mapping ID + buffer HAR + pose + prediksi + wajah)
                # tanpa memutus koneksi Tello. Selama segmen berjalan tracker
                # tidak di-reset sehingga ID stabil.
                if key in (ord("e"), ord("E")):
                    if not pipeline_recording:
                        recording_index += 1
                        pipeline_recording = True
                        recording_frames = 0
                        recording_started = time.perf_counter()
                        segment_timers = ModuleTimer()
                        segment_video_path = (
                            session_dir / f"recording_{recording_index:03d}.mp4"
                        )
                        segment_report_path = (
                            session_dir / f"benchmark_{recording_index:03d}.json"
                        )

                        old_tracks = set(last_seen) | set(last_face)
                        tracker = make_tracker(args)
                        display_used.clear()
                        next_display = 1
                        if reid is not None:
                            reid.reset()
                        for store in (raw_buffers, mask_buffers, probability_history):
                            store.clear()
                        samples_seen.clear()
                        last_seen.clear()
                        last_pose.clear()
                        last_prediction.clear()
                        last_face.clear()
                        if face_system is not None:
                            for tid in old_tracks:
                                face_system.forget_track(tid)

                        segment_writer = H264Mp4Writer(
                            str(segment_video_path), w, h,
                            fps=max(min(source.fps, 30.0), 1.0),
                        )
                        csv_logger.begin_segment(recording_index)

                        print()
                        print("=" * 72)
                        print(f"[REC START] Rekaman #{recording_index:03d}")
                        print("Video :", segment_video_path.resolve())
                        print("Report:", segment_report_path.resolve())
                        print("State eksperimen di-reset; Tello tetap terhubung.")
                        print("Tekan E lagi untuk STOP dan menyimpan segmen ini.")
                        print("=" * 72)
                    else:
                        pipeline_recording = False
                        if segment_writer is not None:
                            segment_writer.close()
                            segment_writer = None
                        write_recording_report(
                            segment_report_path,
                            segment_video_path,
                            recording_index,
                            recording_frames,
                            recording_started,
                            profile,
                            tracker,
                            detector,
                            pose_model,
                            har_session,
                            face_system,
                            segment_timers,
                            args,
                        )
                        finalize_segment(recording_index, multi=True)
                        recording_started = None
                        segment_timers = None
                        segment_video_path = None
                        segment_report_path = None
                        recording_frames = 0

                    # Rekaman dikelola pipeline; jangan diteruskan ke
                    # VideoHandler (tello_control.py) supaya tidak rekam ganda.
                    key = 0
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
            writer.close()
        if segment_writer is not None:
            segment_writer.close()
        csv_logger.close()
        cv2.destroyAllWindows()

    # Jika keluar (q) saat rekaman masih aktif, finalisasi segmen terakhir
    # agar video + report tidak hilang.
    if (args.source == "tello" and pipeline_recording
            and segment_video_path is not None and segment_report_path is not None):
        write_recording_report(
            segment_report_path, segment_video_path, recording_index,
            recording_frames, recording_started, profile, tracker,
            detector, pose_model, har_session, face_system, segment_timers,
            args,
        )
        finalize_segment(recording_index, multi=True)

    # Non-Tello: seluruh run = satu segmen laporan.
    if args.source != "tello" and csv_logger.agg is not None:
        finalize_segment(1, multi=False)

    wall_time = max(time.perf_counter() - started_all, 1e-6)
    report = {
        "run_id": run_id,
        "video_output": str(args.output),
        "report_output": str(args.report),
        "architecture": "off-board",
        "source": args.source,
        "profile": args.profile,
        "profile_values": profile.__dict__,
        "tracker": {
            "name": "ByteTrack",
            "track_high_thresh": tracker.track_high_thresh,
            "track_low_thresh": tracker.track_low_thresh,
            "new_track_thresh": tracker.new_track_thresh,
            "track_buffer": tracker.track_buffer,
            "match_thresh": tracker.match_thresh,
            "second_match_thresh": tracker.second_match_thresh,
        },
        "frames": frame_index,
        "wall_time_seconds": wall_time,
        "throughput_fps": frame_index / wall_time,
        "yolo_ort": [detector.provider, pose_model.provider],
        "onnx_providers": har_session.get_providers(),
        "face_enabled": bool(face_system),
        "platform": platform.platform(),
        "module_timing": timers.summary(),
        "warning": "FPS adalah hasil perangkat dan konfigurasi ini; bukan spesifikasi tetap model.",
    }
    session_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.source == "tello":
        print("Session folder:", session_dir.resolve())
        print("Session report:", session_report_path.resolve())
        print("Log CSV      :", (session_dir / "detections.csv").resolve())
    else:
        if writer is not None:
            print("Video :", args.output.resolve())
        print("Report:", session_report_path.resolve())

    if segment_stat_rows:
        warnings = [
            f"Segmen {e['segment_index']}: tidak ada deteksi - "
            f"cek video, detector-conf ({args.detector_conf}), atau provider ONNX"
            for e in segment_entries if not e.get("detection_ok")
        ]
        manifest_path = write_run_manifest(
            session_dir.name, session_dir.name, args, profile, mapping,
            segment_entries, reported_files, warnings,
        )
        print("[OK] Manifest:", manifest_path.resolve())
        print("[OK] Reports :", reports_session_dir.resolve())

        try:
            import importlib.util as _ilu

            export_script = Path(__file__).resolve().parents[1] / "scripts" / "export_pipeline.py"
            spec = _ilu.spec_from_file_location("export_pipeline", export_script)
            if spec is None or spec.loader is None:
                raise RuntimeError("export_pipeline.py tidak ditemukan")
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)

            summary_path = mod.build()
            print("[OK] Summary :", summary_path.resolve())
        except Exception as exc:
            print(f"[!] Summary generator gagal: {exc}")


if __name__ == "__main__":
    main()