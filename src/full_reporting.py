"""Laporan eksperimen pipeline HAR UAV (adaptasi reporting.py / runlog.py
dari project drone_e99_face_recognition).

CsvLogger (full_pipeline.py) mengumpulkan agregat streaming dalam dict `agg`;
modul ini mengolahnya jadi baris CSV per detik / per segmen / per aktivitas /
per identitas + akurasi (ground truth), lalu manifest run JSON.

Workbook Excel dibuat terpisah oleh scripts/export_pipeline.py.
"""
import csv
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SECOND_COLUMNS = [
    "segment_index", "second", "n_frames", "n_people_avg", "n_detections",
    "dominant_activity", "dominant_activity_pct", "activity_counts",
    "dominant_face", "dominant_face_count",
    "ms_detector", "ms_bytetrack", "ms_pose", "ms_body110_har",
    "ms_face_liveness", "ms_total", "fps_avg",
]
# (nama kolom, nama modul ModuleTimer)
SEGMENT_MODULES = [
    ("ms_detector", "detector"),
    ("ms_bytetrack", "bytetrack"),
    ("ms_pose", "pose"),
    ("ms_body110_har", "body110_har"),
    ("ms_face_liveness", "face_liveness"),
    ("ms_total", "total"),
]
SEGMENT_STATS_COLUMNS = [
    "segment_index", "expected_people", "activity_plan",
    "duration_s", "processed_frames", "n_people_avg", "n_people_max",
    "people_match_frames", "people_match_ratio_pct",
    "throughput_fps", "fps_mean", "fps_p95",
    "n_detections", "detection_rate",
    "dominant_activity", "dominant_activity_pct", "activity_dist",
    "n_tracks", "mean_track_duration_s", "pose_valid_ratio",
    "n_faces", "mean_faces_per_frame", "dominant_face", "dominant_face_pct",
    "real", "spoof", "unknown",
]
for col, _ in SEGMENT_MODULES:
    SEGMENT_STATS_COLUMNS += [f"{col}_mean", f"{col}_p95"]
SEGMENT_STATS_COLUMNS += ["accuracy_pct", "n_labeled_seg", "detection_ok"]

ACTIVITY_COLUMNS = [
    "segment_index", "activity", "n_detections", "pct_time", "mean_score",
    "p95_score", "max_score",
]
IDENTITY_COLUMNS = [
    "segment_index", "identity", "n", "mean_sim", "median_sim", "std_sim",
    "min_sim", "max_sim",
]
ACCURACY_COLUMNS = [
    "segment_index", "activity", "support", "correct", "recall_pct",
    "predicted", "precision_pct",
]
CONFUSION_COLUMNS = ["segment_index", "label", "predicted", "count"]


def _r(value: float) -> float:
    return round(float(value), 2)


def _dominant(counts: Dict[str, int]) -> Tuple[str, float, int]:
    """Nama, persen, dan jumlah terbesar dari dict counter (kosong = '')."""
    total = sum(counts.values())
    if not total:
        return "", 0.0, 0
    name = max(counts, key=lambda k: counts[k])
    return name, counts[name] / total * 100.0, counts[name]


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def pkg_versions() -> Dict[str, str]:
    from importlib.metadata import version
    out: Dict[str, str] = {"python": platform.python_version()}
    for name in ("onnxruntime", "numpy", "opencv-python", "openpyxl"):
        try:
            out[name] = str(version(name))
        except Exception:
            out[name] = ""
    return out


def load_ground_truth(path: Optional[Path]) -> Dict[Tuple[int, int], str]:
    """CSV ground truth: kolom `second,activity` (+ `segment_index` opsional).

    Satu baris = label aktivitas untuk 1 detik dalam segmen tertentu. Dipakai
    menghitung akurasi + confusion matrix per segmen.
    """
    labels: Dict[Tuple[int, int], str] = {}
    if path is None:
        return labels
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            activity = (row.get("activity") or "").strip()
            if not activity:
                continue
            try:
                sec = int(row["second"])
            except (KeyError, ValueError):
                continue
            seg = 1
            raw_seg = row.get("segment_index", "").strip()
            if raw_seg:
                try:
                    seg = int(raw_seg)
                except ValueError:
                    pass
            labels[(seg, sec)] = activity
    return labels


def write_rows_csv(path: Path, rows: List[dict], columns: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def build_second_rows(agg: dict) -> List[dict]:
    rows = []
    for sec in sorted(agg.get("seconds", {})):
        s = agg["seconds"][sec]
        n_frames = s["n_frames"]
        act_name, act_pct, _ = _dominant(s["activity"])
        face_name, _, face_count = _dominant(s["faces"])
        activity_counts = ";".join(
            f"{k}:{v}" for k, v in
            sorted(s["activity"].items(), key=lambda kv: -kv[1]))
        mod_pairs = [("ms_detector", "detector"), ("ms_bytetrack", "bytetrack"),
                     ("ms_pose", "pose"), ("ms_body110_har", "body110_har"),
                     ("ms_face_liveness", "face_liveness"), ("ms_total", "total")]
        vals = {}
        for col, mod in mod_pairs:
            samples = s["ms"].get(mod)
            vals[col] = _r(np.mean(samples)) if samples else ""
        fps = 1000.0 / vals["ms_total"] if vals["ms_total"] else 0.0
        rows.append({
            "segment_index": agg.get("segment_index", 1),
            "second": sec,
            "n_frames": n_frames,
            "n_people_avg": _r(s["people_sum"] / n_frames) if n_frames else 0,
            "n_detections": s["n_detections"],
            "dominant_activity": act_name,
            "dominant_activity_pct": _r(act_pct),
            "activity_counts": activity_counts,
            "dominant_face": face_name,
            "dominant_face_count": face_count,
            **vals,
            "fps_avg": _r(fps),
        })
    return rows


def build_segment_stats_row(agg: dict, n_tracks: int,
                            labels: Optional[Dict[Tuple[int, int], str]] = None) -> dict:
    act_name, act_pct, _ = _dominant(agg.get("activity", {}))
    face_name, face_pct, _ = _dominant(agg.get("faces", {}))
    dist_total = sum(agg.get("activity", {}).values())
    activity_dist = "".join(
        f"{k}:{v / dist_total * 100:.1f}%"
        for k, v in sorted(agg["activity"].items(), key=lambda kv: -kv[1])
    ) if dist_total else ""

    duration = max(agg.get("end_t", 0) - agg.get("start_t", 0), 1e-6) if agg.get("start_t") is not None else 0.0
    n_frames = agg.get("n_frames", 0)
    track_durs = [agg["last_t"][t] - agg["first_t"][t] for t in agg.get("first_t", {})]

    expected_people = int(agg.get("expected_people", 0) or 0)
    people_match_frames = int(agg.get("people_match_frames", 0) or 0)
    row = {
        "segment_index": agg.get("segment_index", 1),
        "expected_people": expected_people if expected_people else "",
        "activity_plan": agg.get("activity_plan", ""),
        "duration_s": _r(duration),
        "processed_frames": n_frames,
        "n_people_avg": _r(agg.get("people_sum", 0) / n_frames) if n_frames else 0,
        "n_people_max": agg.get("people_max", 0),
        "people_match_frames": people_match_frames if expected_people else "",
        "people_match_ratio_pct": _r(people_match_frames / n_frames * 100)
        if expected_people and n_frames else "",
        "throughput_fps": _r(n_frames / duration) if duration else 0,
        "fps_mean": _r(np.mean(agg["fps_samples"])) if agg.get("fps_samples") else 0,
        "fps_p95": _r(np.percentile(agg["fps_samples"], 95)) if agg.get("fps_samples") else 0,
        "n_detections": agg.get("n_detections", 0),
        "detection_rate": _r(agg.get("n_detections", 0) / duration) if duration else 0,
        "dominant_activity": act_name,
        "dominant_activity_pct": _r(act_pct),
        "activity_dist": activity_dist,
        "n_tracks": n_tracks,
        "mean_track_duration_s": _r(np.mean(track_durs)) if track_durs else 0,
        "pose_valid_ratio": _r(agg["pose_valid"] / agg["pose_total"] * 100)
        if agg.get("pose_total") else 0,
        "n_faces": sum(agg.get("faces", {}).values()),
        "mean_faces_per_frame": _r(sum(agg.get("faces", {}).values()) / n_frames)
        if n_frames else 0,
        "dominant_face": face_name,
        "dominant_face_pct": face_pct,
        "real": agg["liveness"].get("real", 0),
        "spoof": agg["liveness"].get("spoof", 0),
        "unknown": agg["liveness"].get("unknown", 0),
        "accuracy_pct": "",
        "n_labeled_seg": 0,
        "detection_ok": 1 if agg.get("n_detections", 0) > 0 else 0,
    }
    for col, mod in SEGMENT_MODULES:
        samples = agg.get("ms_samples", {}).get(mod, [])
        if samples:
            row[f"{col}_mean"] = _r(np.mean(samples))
            row[f"{col}_p95"] = _r(np.percentile(samples, 95))
        else:
            row[f"{col}_mean"] = ""
            row[f"{col}_p95"] = ""

    if labels and agg.get("seconds"):
        _, _, accuracy, n_labeled = compute_accuracy(agg, labels)
        if accuracy is not None:
            row["accuracy_pct"] = _r(accuracy)
        row["n_labeled_seg"] = n_labeled
    return row


def build_activity_rows(agg: dict) -> List[dict]:
    rows = []
    activities = agg.get("activities", {})
    total = sum(v["count"] for v in activities.values())
    for activity, acc in activities.items():
        count, score_sum, scores = acc["count"], acc["score_sum"], np.asarray(acc["scores"])
        rows.append({
            "segment_index": agg.get("segment_index", 1),
            "activity": activity,
            "n_detections": count,
            "pct_time": _r(count / total * 100) if total else 0,
            "mean_score": _r(score_sum / count) if count else 0,
            "p95_score": _r(np.percentile(scores, 95)) if scores.size else 0,
            "max_score": _r(scores.max()) if scores.size else 0,
        })
    return sorted(rows, key=lambda r: -r["n_detections"])


def build_identity_rows(agg: dict) -> List[dict]:
    rows = []
    for identity, sims in agg.get("identity_sims", {}).items():
        arr = np.asarray(sims, dtype=np.float64)
        rows.append({
            "segment_index": agg.get("segment_index", 1),
            "identity": identity,
            "n": int(arr.size),
            "mean_sim": _r(arr.mean()) if arr.size else 0,
            "median_sim": _r(np.median(arr)) if arr.size else 0,
            "std_sim": _r(arr.std()) if arr.size else 0,
            "min_sim": _r(arr.min()) if arr.size else 0,
            "max_sim": _r(arr.max()) if arr.size else 0,
        })
    return sorted(rows, key=lambda r: -r["n"])


def compute_accuracy(agg: dict, labels: Dict[Tuple[int, int], str]
                     ) -> Tuple[List[dict], List[dict], Optional[float], int]:
    """Akurasi per detik vs ground truth untuk satu segmen.

    Prediksi = aktivitas dominan pada detik tsb; detik tanpa deteksi dihitung
    sebagai salah. Return (accuracy_rows, confusion_rows, acc_pct, n_label).
    """
    seg = agg.get("segment_index", 1)
    predicted = {}
    for sec, s in agg.get("seconds", {}).items():
        name, _, _ = _dominant(s["activity"])
        predicted[sec] = name

    label_secs = {sec: act for (s, sec), act in labels.items() if s == seg}
    if not label_secs:
        return [], [], None, 0

    confusion = {}
    correct_by = {}
    for sec, act in label_secs.items():
        pred = predicted.get(sec, "")
        confusion[(act, pred)] = confusion.get((act, pred), 0) + 1
        correct_by[act] = correct_by.get(act, 0) + int(pred == act)

    n = len(label_secs)
    n_correct = sum(correct_by.values())
    acc_rows = [{
        "segment_index": seg, "activity": "ALL", "support": n,
        "correct": n_correct,
        "recall_pct": _r(n_correct / n * 100),
        "predicted": n,
        "precision_pct": _r(n_correct / n * 100),
    }]
    for act in sorted(set(label_secs.values())):
        support = sum(1 for sec in label_secs if label_secs[sec] == act)
        c = correct_by.get(act, 0)
        p_tot = sum(cnt for (la, pr), cnt in confusion.items() if la == act)
        acc_rows.append({
            "segment_index": seg, "activity": act, "support": support,
            "correct": c,
            "recall_pct": _r(c / support * 100) if support else 0,
            "predicted": p_tot,
            "precision_pct": _r(c / p_tot * 100) if p_tot else 0,
        })
    conf_rows = [
        {"segment_index": seg, "label": la, "predicted": pr, "count": cnt}
        for (la, pr), cnt in sorted(confusion.items())
    ]
    return acc_rows, conf_rows, n_correct / n * 100.0, n


def write_run_manifest(run_id: str, session_id: str, args, profile,
                       mapping: Dict[int, str], segments: List[dict],
                       files: List[str],
                       warnings: Optional[List[str]] = None) -> Path:
    """Manifest run (mirror runlog.py): config + env + hasil ringkas."""
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "session_id": session_id,
        "source": args.source,
        "profile": args.profile,
        "profile_values": profile.__dict__,
        "labels": str(args.labels) if args.labels else None,
        "class_mapping": mapping,
        "config": {
            "detector_conf": args.detector_conf,
            "pose_conf": args.pose_conf,
            "pose_interval": profile.pose_interval,
            "face_interval": profile.face_interval,
            "max_people": profile.max_people,
            "tracker": {
                "track_high_thresh": args.detector_conf,
                "track_low_thresh": min(0.10, args.detector_conf),
                "new_track_thresh": args.detector_conf,
                "track_buffer": 20,
                "match_thresh": 0.80,
                "second_match_thresh": 0.50,
            },
            "face_enabled": bool(getattr(args, "enable_face", False)),
            "face_threshold": args.face_threshold,
            "liveness_threshold": args.liveness_threshold,
            "reid": {
                "enabled": bool(getattr(args, "enable_face", False)),
                "reid_cos": args.reid_cos,
                "reid_iou": args.reid_iou,
                "reid_retired_cos": args.reid_retired_cos,
                "reid_max_missed": args.reid_max_missed,
            },
            "experiment": {
                "enabled": bool(getattr(args, "experiment_people", 0)),
                "expected_people": int(getattr(args, "experiment_people", 0) or 0),
                "repetitions": int(getattr(args, "experiment_repetitions", 0) or 0),
                "warmup_seconds": float(getattr(args, "experiment_warmup", 0.0) or 0.0),
                "measurement_seconds": float(getattr(args, "experiment_duration", 0.0) or 0.0),
                "activity_plan": str(getattr(args, "experiment_activity", "") or ""),
            },
        },
        "env": {**pkg_versions(), "git_commit": git_commit()},
        "segments": segments,
        "files": files,
        "warnings": warnings or [],
    }
    path = Path("reports") / "runs" / f"manifest_{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
