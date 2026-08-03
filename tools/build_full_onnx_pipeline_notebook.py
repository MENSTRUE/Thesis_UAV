from pathlib import Path

import nbformat as nbf


ROOT = Path("/workspace/scratch/82d934242f4c")
OUT = ROOT / "deliverables" / "FULL_ONNX_HAR_UAV_Kaggle.ipynb"
BODY110_SOURCE = (ROOT / "deliverables" / "offboard_tello_har" / "body110.py").read_text(encoding="utf-8")

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kaggle": {"accelerator": "gpu", "dataSources": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

cells = []
cells.append(nbf.v4.new_markdown_cell("""# Pengujian Pipeline Lengkap ONNX HAR UAV

Pipeline yang diuji:

`video → YOLOv8s ONNX → ByteTrack → crop → YOLO26s-Pose ONNX → Raw51 → Body110 → CNN-BiLSTM ONNX → wajah/liveness opsional → video dan laporan`

Notebook memakai ONNX CPU untuk menghindari konflik CUDA/ONNX Runtime pada Kaggle. Angka FPS Kaggle bukan prediksi FPS Jetson. Tujuan tahap ini adalah membuktikan integrasi, bentuk data, hasil pose, dan keluaran HAR.

Mode:

- `CORE`: detector, tracker, pose, Body110, dan HAR.
- `FULL`: menambahkan InsightFace dan MiniFASNetV2.

Jalankan `CORE` terlebih dahulu. Setelah berhasil, ubah `PIPELINE_MODE = "FULL"` dan jalankan ulang dari awal."""))

cells.append(nbf.v4.new_code_cell("""# Dependensi. ONNX dijalankan pada CPU agar stabil di semua accelerator Kaggle.
!pip install -q -U ultralytics onnxruntime onnx lap

# InsightFace hanya diperlukan ketika PIPELINE_MODE='FULL'. Kegagalan instalasi
# tidak menghalangi mode CORE.
!pip install -q insightface || true

print("Dependensi selesai.")"""))

cells.append(nbf.v4.new_code_cell("""import gc
import json
import math
import platform
import shutil
import subprocess
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import ultralytics
from IPython.display import Video, display
from ultralytics import YOLO

warnings.filterwarnings("ignore")

print("Python       :", platform.python_version())
print("Torch        :", torch.__version__)
print("GPU tersedia :", torch.cuda.is_available())
print("GPU          :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Ultralytics  :", ultralytics.__version__)
print("ORT          :", ort.__version__)
print("ORT providers:", ort.get_available_providers())
"""))

cells.append(nbf.v4.new_markdown_cell("""## Konfigurasi utama

Konfigurasi awal dibatasi satu manusia dan 300 frame agar pengujian pertama selesai lebih cepat. Setelah pipeline lulus, ubah `MAX_FRAMES = 0` untuk memproses video penuh."""))

cells.append(nbf.v4.new_code_cell("""# ============================================================
# KONFIGURASI
# ============================================================
PIPELINE_MODE = "CORE"       # "CORE" dahulu, kemudian "FULL"

YOLO_DIR_REQUESTED = Path("/kaggle/input/datasets/wafabila/yolo-onnx")
YOLO_DATASET_HANDLE = "wafabila/yolo-onnx"
DETECTOR_FILENAME = "yolov8s_512_fp32.onnx"
POSE_FILENAME = "yolo26s-pose_512_fp32.onnx"

IMGSZ = 512
DETECTOR_CONF = 0.15
POSE_CONF = 0.05
KEYPOINT_CONF = 0.15
MIN_VALID_KEYPOINTS = 5
MAX_PEOPLE = 1
POSE_INTERVAL = 2
FACE_INTERVAL = 10
SEQUENCE_LENGTH = 30
STEP_SIZE = 10
MAX_FRAMES = 300           # 0 = seluruh video
TRACK_STALE_FRAMES = 45
GESTURE_SMOOTHING = 5

FACE_MODEL_NAME = "buffalo_sc"
FACE_DET_SIZE = (640, 640)
FACE_SIMILARITY_THRESHOLD = 0.39
LIVENESS_THRESHOLD = 0.85

ONNX_DEVICE = "cpu"
WORK_ROOT = Path("/kaggle/working/FULL_ONNX_HAR_UAV")
OUTPUT_DIR = WORK_ROOT / "outputs"
FRAME_DIR = OUTPUT_DIR / "sample_frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

print("Mode:", PIPELINE_MODE)
print("Work:", WORK_ROOT)
"""))

cells.append(nbf.v4.new_code_cell("""# ============================================================
# PENEMUAN FILE OTOMATIS
# ============================================================
INPUT_ROOT = Path("/kaggle/input")


def find_exact(filename, preferred_text=None, required=True):
    candidates = sorted(INPUT_ROOT.rglob(filename))
    if preferred_text:
        preferred = [path for path in candidates if preferred_text.lower() in str(path).lower()]
        if preferred:
            candidates = preferred
    if candidates:
        return candidates[0]
    if required:
        raise FileNotFoundError(f"{filename} tidak ditemukan di /kaggle/input")
    return None


def find_under(root, filename):
    if root is None or not root.exists():
        return None
    direct = root / filename
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def resolve_yolo_models():
    candidate_roots = [
        YOLO_DIR_REQUESTED,
        Path("/kaggle/input/yolo-onnx"),
        Path("/kaggle/input/yolo_onnx"),
        INPUT_ROOT,
    ]

    for root in candidate_roots:
        detector = find_under(root, DETECTOR_FILENAME)
        pose = find_under(root, POSE_FILENAME)
        if detector is not None and pose is not None:
            print("[YOLO FOUND]", root)
            return detector, pose

    print("Model YOLO belum terlihat pada mount /kaggle/input.")
    print("Mencoba dataset_download:", YOLO_DATASET_HANDLE)
    try:
        import kagglehub
        downloaded_root = Path(kagglehub.dataset_download(YOLO_DATASET_HANDLE))
    except Exception as exc:
        visible = [str(path) for path in sorted(INPUT_ROOT.glob("*"))]
        raise FileNotFoundError(
            "Dataset YOLO tidak terlihat dan pengunduhan otomatis gagal. "
            "Restart Session setelah Add Input. Root yang terlihat: "
            f"{visible}. Penyebab download: {exc}"
        ) from exc

    detector = find_under(downloaded_root, DETECTOR_FILENAME)
    pose = find_under(downloaded_root, POSE_FILENAME)
    if detector is None or pose is None:
        available_onnx = [str(path) for path in downloaded_root.rglob("*.onnx")]
        raise FileNotFoundError(
            f"Dataset terunduh ke {downloaded_root}, tetapi file target tidak ditemukan. "
            f"ONNX tersedia: {available_onnx}"
        )
    print("[YOLO DOWNLOADED]", downloaded_root)
    return detector, pose


detector_path, pose_path = resolve_yolo_models()

har_path = find_exact("har_window_30_representative.onnx", "model")
mean_path = find_exact("feature_mean.npy", "model")
std_path = find_exact("feature_std.npy", "model")
metadata_path = find_exact("pipeline_metadata.json", "model")

video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v"}
video_candidates = sorted(
    path for path in INPUT_ROOT.rglob("*")
    if path.is_file() and path.suffix.lower() in video_extensions
)
preferred_videos = [path for path in video_candidates if "video_test" in str(path).lower()]
video_paths = preferred_videos or video_candidates
if not video_paths:
    raise FileNotFoundError("Tidak ada video pengujian di /kaggle/input")
video_path = video_paths[0]

for path, label in [
    (detector_path, "detector"), (pose_path, "pose"), (har_path, "HAR"),
    (mean_path, "mean"), (std_path, "std"), (metadata_path, "metadata"),
    (video_path, "video"),
]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
class_names = metadata.get("class_names", [])
if not class_names:
    raise KeyError("class_names tidak ditemukan di pipeline_metadata.json")

print("Detector :", detector_path)
print("Pose     :", pose_path)
print("HAR      :", har_path)
print("Mean/std :", mean_path, std_path)
print("Metadata :", metadata_path)
print("Video    :", video_path)
print("Classes  :", class_names)
"""))

cells.append(nbf.v4.new_markdown_cell("""## Transformasi Body110

Sel berikut disalin dari implementasi yang sama dengan training final. Raw51 `(30,51)` diubah menjadi Body85 dan 25 fitur biomekanik, sehingga masukan HAR menjadi `(1,30,110)`."""))
cells.append(nbf.v4.new_code_cell(BODY110_SOURCE))

cells.append(nbf.v4.new_code_cell("""# ============================================================
# PEMERIKSAAN MODEL DAN SCALER
# ============================================================
feature_mean = np.load(mean_path).astype(np.float32).reshape(-1)
feature_std = np.load(std_path).astype(np.float32).reshape(-1)
assert feature_mean.shape == (110,), feature_mean.shape
assert feature_std.shape == (110,), feature_std.shape
assert np.isfinite(feature_mean).all() and np.isfinite(feature_std).all()

har_session = ort.InferenceSession(str(har_path), providers=["CPUExecutionProvider"])
har_inputs = {item.name: item for item in har_session.get_inputs()}
har_outputs = har_session.get_outputs()
assert "input" in har_inputs and "frame_mask" in har_inputs, list(har_inputs)

detector_model = YOLO(str(detector_path), task="detect")
pose_model = YOLO(str(pose_path), task="pose")

dummy_raw = np.zeros((30, 51), dtype=np.float32)
dummy_mask = np.ones(30, dtype=bool)
dummy_x, dummy_m = prepare_sequence(dummy_raw, dummy_mask, feature_mean, feature_std)
dummy_logits = har_session.run(None, {"input": dummy_x, "frame_mask": dummy_m})[0]

assert dummy_x.shape == (1, 30, 110)
assert dummy_m.shape == (1, 30)
assert dummy_logits.shape == (1, len(class_names)), (dummy_logits.shape, len(class_names))

print("Detector task:", detector_model.task)
print("Pose task    :", pose_model.task)
print("HAR inputs   :", [(item.name, item.shape, item.type) for item in har_session.get_inputs()])
print("HAR outputs  :", [(item.name, item.shape, item.type) for item in har_outputs])
print("Scaler       :", feature_mean.shape, feature_std.shape)
print("Dummy parity : PASS", dummy_x.shape, dummy_logits.shape)
"""))

cells.append(nbf.v4.new_markdown_cell("""## Modul wajah dan liveness opsional

Pada mode `CORE`, sel ini hanya menyiapkan kelas tanpa memuat model. Pada mode `FULL`, database embedding, InsightFace, dan MiniFASNetV2 dicari dan diaktifkan."""))

cells.append(nbf.v4.new_code_cell("""def normalize_embedding(value):
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    return value / max(float(np.linalg.norm(value)), 1e-8)


class MiniFASNetV2:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
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
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        logits = logits - np.max(logits)
        probabilities = np.exp(logits) / np.maximum(np.exp(logits).sum(), 1e-8)
        return float(probabilities[1])


class FaceSystem:
    def __init__(self, embedding_root, anti_spoof_path):
        from insightface.app import FaceAnalysis

        files = sorted(embedding_root.rglob("emb_*.npy"))
        if not files:
            raise FileNotFoundError(f"Embedding wajah kosong: {embedding_root}")
        self.database = defaultdict(list)
        for file in files:
            self.database[file.parent.name].append(normalize_embedding(np.load(file)))

        self.app = FaceAnalysis(
            name=FACE_MODEL_NAME,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=FACE_DET_SIZE)
        self.anti_spoof = MiniFASNetV2(anti_spoof_path)

    def recognize(self, embedding):
        query = normalize_embedding(embedding)
        best_name, best_score = "unknown", -1.0
        for name, values in self.database.items():
            score = max(float(np.dot(query, item)) for item in values)
            if score > best_score:
                best_name, best_score = name, score
        if best_score < FACE_SIMILARITY_THRESHOLD:
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
                candidates.append((max((x2 - x1) * (y2 - y1), 1.0), track_id))
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
                "liveness": "real" if live_score >= LIVENESS_THRESHOLD else "spoof",
                "liveness_score": live_score,
            }
        return output


face_system = None
if PIPELINE_MODE.upper() == "FULL":
    embedding_candidates = sorted(INPUT_ROOT.rglob("emb_*.npy"))
    anti_spoof_path = find_exact("MiniFASNetV2.onnx", "face", required=False)
    if not embedding_candidates or anti_spoof_path is None:
        raise FileNotFoundError("Mode FULL membutuhkan emb_*.npy dan MiniFASNetV2.onnx")
    # parent dari folder identitas pertama -> folder embeddings
    embedding_root = embedding_candidates[0].parent.parent
    face_system = FaceSystem(embedding_root, anti_spoof_path)
    print("Face FULL aktif:", embedding_root, anti_spoof_path)
else:
    print("Mode CORE: face dan liveness dilewati.")
"""))

cells.append(nbf.v4.new_markdown_cell("""## Fungsi pipeline video"""))

cells.append(nbf.v4.new_code_cell("""SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]


def padded_box(box, width, height, pad_x=0.25, pad_y=0.35):
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    return (
        max(0, int(x1 - bw * pad_x)), max(0, int(y1 - bh * pad_y)),
        min(width, int(x2 + bw * pad_x)), min(height, int(y2 + bh * pad_y)),
    )


def pose_to_raw51(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = padded_box(box, width, height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(51, np.float32), False

    result = pose_model.predict(
        crop, imgsz=IMGSZ, conf=POSE_CONF, iou=0.50,
        classes=[0], max_det=5, device=ONNX_DEVICE, verbose=False,
    )[0]
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return np.zeros(51, np.float32), False

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    keypoints = result.keypoints.data.detach().cpu().numpy().astype(np.float32)
    assert keypoints.shape[1] == 17 and keypoints.shape[2] >= 3, keypoints.shape
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    best = int(np.argmax(areas * np.maximum(scores, 1e-6)))

    pose = keypoints[best, :17, :3].copy()
    pose[:, 0] = (pose[:, 0] + x1) / max(width, 1)
    pose[:, 1] = (pose[:, 1] + y1) / max(height, 1)
    pose[:, :2] = np.clip(pose[:, :2], 0.0, 1.0)
    valid = pose[:, 2] >= KEYPOINT_CONF
    pose[~valid] = 0.0
    frame_valid = int(valid.sum()) >= MIN_VALID_KEYPOINTS
    if not frame_valid:
        pose[:] = 0.0
    return pose.reshape(51).astype(np.float32), frame_valid


def softmax(logits):
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-8)


def infer_har(raw_buffer, mask_buffer):
    model_input, mask = prepare_sequence(
        np.asarray(raw_buffer), np.asarray(mask_buffer), feature_mean, feature_std
    )
    logits = har_session.run(None, {"input": model_input, "frame_mask": mask})[0]
    probability = softmax(logits)[0]
    assert probability.shape == (len(class_names),)
    return probability.astype(np.float32)


def draw_pose(frame, raw51):
    h, w = frame.shape[:2]
    pose = np.asarray(raw51).reshape(17, 3)
    valid = pose[:, 2] > 0
    points = np.stack([pose[:, 0] * w, pose[:, 1] * h], axis=-1).astype(int)
    for a, b in SKELETON_EDGES:
        if valid[a] and valid[b]:
            cv2.line(frame, tuple(points[a]), tuple(points[b]), (0, 255, 255), 2)
    for index, point in enumerate(points):
        if valid[index]:
            cv2.circle(frame, tuple(point), 3, (0, 80, 255), -1)


def draw_track(frame, box, lines):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    y = max(18, y1 - 8)
    for line in reversed(lines):
        cv2.putText(frame, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y -= 20
"""))

cells.append(nbf.v4.new_markdown_cell("""## Jalankan pipeline

Cell ini menghasilkan video beranotasi, prediksi HAR per window, audit frame, serta waktu setiap modul."""))

cells.append(nbf.v4.new_code_cell("""capture = cv2.VideoCapture(str(video_path))
if not capture.isOpened():
    raise RuntimeError(f"Video gagal dibuka: {video_path}")

source_fps = float(capture.get(cv2.CAP_PROP_FPS))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
if not np.isfinite(source_fps) or source_fps <= 0:
    source_fps = 30.0

temporary_video = OUTPUT_DIR / f"{video_path.stem}_{PIPELINE_MODE.lower()}_onnx_mp4v.mp4"
output_video = OUTPUT_DIR / f"{video_path.stem}_{PIPELINE_MODE.lower()}_onnx_h264.mp4"
writer = cv2.VideoWriter(
    str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"),
    min(source_fps, 30.0), (width, height),
)
if not writer.isOpened():
    capture.release()
    raise RuntimeError("VideoWriter gagal")

raw_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
mask_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
probability_history = defaultdict(lambda: deque(maxlen=GESTURE_SMOOTHING))
samples_seen = defaultdict(int)
last_seen, last_pose, last_prediction, last_face = {}, {}, {}, {}

timings = defaultdict(list)
prediction_rows = []
frame_rows = []
frame_index = 0
started_all = time.perf_counter()

while True:
    frame_started = time.perf_counter()
    ok, frame = capture.read()
    if not ok:
        break
    if MAX_FRAMES > 0 and frame_index >= MAX_FRAMES:
        break

    tick = time.perf_counter()
    tracked = detector_model.track(
        frame, persist=True, tracker="bytetrack.yaml", classes=[0],
        conf=DETECTOR_CONF, iou=0.50, imgsz=IMGSZ,
        max_det=MAX_PEOPLE, device=ONNX_DEVICE, verbose=False,
    )[0]
    timings["detector_bytetrack_ms"].append((time.perf_counter() - tick) * 1000)

    detections = []
    current_boxes = {}
    if tracked.boxes is not None and len(tracked.boxes) > 0:
        boxes = tracked.boxes.xyxy.detach().cpu().numpy()
        confs = tracked.boxes.conf.detach().cpu().numpy()
        ids_tensor = tracked.boxes.id
        ids = ids_tensor.detach().cpu().numpy().astype(int) if ids_tensor is not None else np.arange(len(boxes))
        detections = list(zip(boxes, confs, ids))
        current_boxes = {int(track_id): box for box, _, track_id in detections}

    if face_system is not None and frame_index % FACE_INTERVAL == 0:
        tick = time.perf_counter()
        last_face.update(face_system.process(frame, current_boxes))
        timings["face_liveness_ms"].append((time.perf_counter() - tick) * 1000)

    for box, detector_score, track_id in detections:
        track_id = int(track_id)
        run_pose = frame_index % POSE_INTERVAL == 0 or track_id not in last_pose
        if run_pose:
            tick = time.perf_counter()
            raw51, frame_valid = pose_to_raw51(frame, box)
            timings["pose_ms"].append((time.perf_counter() - tick) * 1000)
            last_pose[track_id] = raw51
        else:
            raw51 = last_pose[track_id].copy()
            frame_valid = int(np.count_nonzero(raw51.reshape(17, 3)[:, 2])) >= MIN_VALID_KEYPOINTS

        raw_buffers[track_id].append(raw51)
        mask_buffers[track_id].append(frame_valid)
        samples_seen[track_id] += 1
        last_seen[track_id] = frame_index

        if len(raw_buffers[track_id]) == SEQUENCE_LENGTH and samples_seen[track_id] % STEP_SIZE == 0:
            tick = time.perf_counter()
            probabilities = infer_har(raw_buffers[track_id], mask_buffers[track_id])
            timings["body110_har_ms"].append((time.perf_counter() - tick) * 1000)
            probability_history[track_id].append(probabilities)
            smooth = np.mean(probability_history[track_id], axis=0)
            class_id = int(np.argmax(smooth))
            last_prediction[track_id] = (class_names[class_id], float(smooth[class_id]))
            prediction_rows.append({
                "frame_index": frame_index,
                "time_seconds": frame_index / source_fps,
                "track_id": track_id,
                "activity": class_names[class_id],
                "confidence": float(smooth[class_id]),
                "valid_pose_frames": int(sum(mask_buffers[track_id])),
            })

        draw_pose(frame, raw51)
        activity, activity_score = last_prediction.get(track_id, ("collecting", 0.0))
        lines = [
            f"T{track_id} | {activity} {activity_score:.2f} | det {float(detector_score):.2f}",
            f"pose {int(sum(mask_buffers[track_id]))}/{len(mask_buffers[track_id])}",
        ]
        if track_id in last_face:
            face = last_face[track_id]
            lines.append(
                f"{face['identity']} {face['similarity']:.2f} | "
                f"{face['liveness']} {face['liveness_score']:.2f}"
            )
        draw_track(frame, box, lines)

    stale = [track_id for track_id, seen in last_seen.items() if frame_index - seen > TRACK_STALE_FRAMES]
    for track_id in stale:
        for store in [raw_buffers, mask_buffers, probability_history, samples_seen,
                      last_seen, last_pose, last_prediction, last_face]:
            store.pop(track_id, None)

    frame_ms = (time.perf_counter() - frame_started) * 1000
    timings["total_frame_ms"].append(frame_ms)
    fps_now = 1000.0 / max(frame_ms, 1e-6)
    cv2.putText(
        frame, f"ONNX {PIPELINE_MODE} | {fps_now:.2f} FPS | frame {frame_index}",
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
        cv2.LINE_AA,
    )
    writer.write(frame)

    frame_rows.append({
        "frame_index": frame_index,
        "time_seconds": frame_index / source_fps,
        "people": len(detections),
        "frame_ms": frame_ms,
        "instant_fps": fps_now,
    })

    if frame_index in {0, 29, 59, 119, 299}:
        cv2.imwrite(str(FRAME_DIR / f"frame_{frame_index:06d}.jpg"), frame)

    frame_index += 1

capture.release()
writer.release()
wall_seconds = time.perf_counter() - started_all

# H.264 lebih mudah diputar pada browser/Kaggle dibanding MP4V.
ffmpeg_command = [
    "ffmpeg", "-y", "-loglevel", "error", "-i", str(temporary_video),
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-pix_fmt", "yuv420p", str(output_video),
]
ffmpeg_result = subprocess.run(ffmpeg_command, check=False)
if ffmpeg_result.returncode != 0 or not output_video.is_file():
    print("Peringatan: konversi H.264 gagal; memakai MP4V sebagai fallback.")
    output_video = temporary_video

print("Selesai memproses", frame_index, "frame")
print("Video output:", output_video)
print("Wall time   :", wall_seconds)
print("Throughput  :", frame_index / max(wall_seconds, 1e-6), "FPS")
"""))

cells.append(nbf.v4.new_markdown_cell("""## Laporan, visualisasi, dan ZIP"""))

cells.append(nbf.v4.new_code_cell("""prediction_df = pd.DataFrame(prediction_rows)
frame_df = pd.DataFrame(frame_rows)
prediction_csv = OUTPUT_DIR / "har_predictions.csv"
frame_csv = OUTPUT_DIR / "frame_audit.csv"
prediction_df.to_csv(prediction_csv, index=False)
frame_df.to_csv(frame_csv, index=False)


def summarize(values):
    if not values:
        return {"count": 0, "mean_ms": None, "p95_ms": None, "max_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(array.max()),
    }


report = {
    "pipeline_mode": PIPELINE_MODE,
    "video": str(video_path),
    "frames_processed": frame_index,
    "source_fps": source_fps,
    "source_total_frames": total_source_frames,
    "wall_seconds": wall_seconds,
    "throughput_fps": frame_index / max(wall_seconds, 1e-6),
    "configuration": {
        "imgsz": IMGSZ,
        "max_people": MAX_PEOPLE,
        "pose_interval": POSE_INTERVAL,
        "face_interval": FACE_INTERVAL,
        "window": SEQUENCE_LENGTH,
        "step": STEP_SIZE,
        "onnx_device": ONNX_DEVICE,
    },
    "models": {
        "detector": str(detector_path),
        "pose": str(pose_path),
        "har": str(har_path),
    },
    "prediction_count": len(prediction_rows),
    "module_timing": {name: summarize(values) for name, values in timings.items()},
    "warning": "Throughput Kaggle ONNX CPU bukan FPS TensorRT Jetson.",
}

report_path = OUTPUT_DIR / "pipeline_benchmark.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
display(prediction_df.head(20))
display(frame_df.describe(include="all"))

sample_images = sorted(FRAME_DIR.glob("*.jpg"))
if sample_images:
    columns = 2
    rows = math.ceil(len(sample_images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, image_path in zip(axes, sample_images):
        image = cv2.imread(str(image_path))
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(image_path.name)
        axis.axis("off")
    for axis in axes[len(sample_images):]:
        axis.axis("off")
    plt.tight_layout()
    plt.show()

zip_path = Path(shutil.make_archive(
    "/kaggle/working/FULL_ONNX_HAR_UAV_RESULT", "zip", root_dir=OUTPUT_DIR
))

print("=" * 80)
print("OUTPUT VIDEO:", output_video)
print("PREDICTIONS :", prediction_csv)
print("FRAME AUDIT :", frame_csv)
print("BENCHMARK   :", report_path)
print("ZIP         :", zip_path)
print("=" * 80)
print("Klik Save Version untuk mempertahankan output Kaggle.")
"""))

cells.append(nbf.v4.new_code_cell("""# Preview video di notebook
if output_video.is_file():
    display(Video(str(output_video), embed=True, width=900))
"""))

cells.append(nbf.v4.new_markdown_cell("""## Interpretasi hasil

Pipeline dinyatakan terintegrasi jika:

1. video output terbentuk dan dapat diputar;
2. bounding box memiliki Track ID stabil;
3. skeleton memiliki 17 keypoint;
4. buffer pose mencapai 30 frame;
5. `har_predictions.csv` berisi kelas aktivitas;
6. `pipeline_benchmark.json` tidak mengandung error atau NaN.

Jika mode `CORE` berhasil, ubah ke `FULL` untuk mengukur tambahan beban wajah dan liveness. Jangan membandingkan FPS ONNX CPU Kaggle dengan TensorRT Jetson sebagai angka yang setara."""))

nb["cells"] = cells
nbf.write(nb, OUT)
print(OUT)
