"""Workbook summary eksperimen (adaptasi scripts/export_summary.py pada
project drone_e99_face_recognition).

Satu file Excel, 1 baris per session/run: `reports/summary/summary_pipeline.xlsx`.

Membaca hanya `reports/runs/manifest_*.json` (satu folder) sehingga tidak perlu
mencari-cari file di seluruh `reports/sessions/**`. Detail mentah tetap di CSV
per-session (`reports/sessions/<session_id>/...`) dan tidak digabung di sini.

Baris session lama dipertahankan (tidak ditimpa) saat regenerate.
"""
import json
import sys
from glob import glob
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS_DIR = ROOT / "reports" / "runs"
OUT = ROOT / "reports" / "summary" / "summary_pipeline.xlsx"

EXPERIMENT_HEADERS = [
    "timestamp", "session_id", "source", "profile", "n_segments",
    "n_frames", "duration_s", "throughput_fps", "dominant_activity",
    "accuracy_pct", "detection_ok", "labels",
    "detector_conf", "pose_conf", "pose_interval", "face_interval",
    "max_people", "face_enabled", "face_threshold", "liveness_threshold",
    "git_commit", "manifest",
]


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"[WARN] Manifest korup, dilewati: {path}")
        return None


def load_manifests():
    out = []
    for path in sorted(glob(str(RUNS_DIR / "manifest_*.json"))):
        data = _load_json(path)
        if data:
            out.append(data)
    return out


def _aggregate(segments):
    """Ringkasan 1 baris dari daftar segmen manifest."""
    if not segments:
        return {
            "n_segments": 0, "n_frames": 0, "duration_s": 0.0,
            "throughput_fps": None, "dominant_activity": "",
            "accuracy_pct": None, "detection_ok": False,
        }
    n_frames = sum(s.get("frames", 0) for s in segments)
    duration = sum(s.get("duration_s", 0) for s in segments)
    fps_vals = [s["throughput_fps"] for s in segments if s.get("throughput_fps")]
    acc_vals = [s["accuracy_pct"] for s in segments
                if isinstance(s.get("accuracy_pct"), (int, float))]
    longest = max(segments, key=lambda s: s.get("duration_s", 0))
    return {
        "n_segments": len(segments),
        "n_frames": n_frames,
        "duration_s": round(duration, 2),
        "throughput_fps": round(sum(fps_vals) / len(fps_vals), 2) if fps_vals else None,
        "dominant_activity": longest.get("dominant_activity", ""),
        "accuracy_pct": round(sum(acc_vals) / len(acc_vals), 2) if acc_vals else None,
        "detection_ok": all(s.get("detection_ok") for s in segments),
    }


def experiment_rows(manifests):
    rows = []
    for m in manifests:
        cfg = m.get("config", {})
        env = m.get("env", {})
        agg = _aggregate(m.get("segments", []))
        rows.append({
            "timestamp": m.get("timestamp"),
            "session_id": m.get("session_id"),
            "source": m.get("source"),
            "profile": m.get("profile"),
            **agg,
            "labels": m.get("labels"),
            "detector_conf": cfg.get("detector_conf"),
            "pose_conf": cfg.get("pose_conf"),
            "pose_interval": cfg.get("pose_interval"),
            "face_interval": cfg.get("face_interval"),
            "max_people": cfg.get("max_people"),
            "face_enabled": cfg.get("face_enabled"),
            "face_threshold": cfg.get("face_threshold"),
            "liveness_threshold": cfg.get("liveness_threshold"),
            "git_commit": env.get("git_commit"),
            "manifest": m.get("run_id"),
        })
    return rows


def write_table(wb, title, rows):
    ws = wb.create_sheet(title)
    for c, h in enumerate(EXPERIMENT_HEADERS, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9E1F2")
    for r, row in enumerate(rows, 2):
        for c, h in enumerate(EXPERIMENT_HEADERS):
            ws.cell(row=r, column=c + 1, value=row.get(h))
    ci = EXPERIMENT_HEADERS.index("throughput_fps") + 1
    vals = [ws.cell(row=r, column=ci).value for r in range(2, len(rows) + 2)]
    vals = [v for v in vals if isinstance(v, (int, float))]
    if vals:
        best = max(vals)
        fill = PatternFill("solid", fgColor="C6EFCE")
        for r in range(2, len(rows) + 2):
            if ws.cell(row=r, column=ci).value == best:
                ws.cell(row=r, column=ci).fill = fill
    for c in range(1, len(EXPERIMENT_HEADERS) + 1):
        width = len(str(EXPERIMENT_HEADERS[c - 1]))
        for r in range(2, len(rows) + 2):
            val = ws.cell(row=r, column=c).value
            if val is not None:
                width = max(width, len(str(val)))
        ws.column_dimensions[get_column_letter(c)].width = min(width + 2, 40)
    ws.freeze_panes = "A2"
    return ws


def _load_existing_log():
    """Baca baris ExperimentLog dari workbook lama supaya tidak hilang."""
    if not OUT.exists():
        return []
    try:
        wb = load_workbook(OUT, read_only=True)
    except Exception:
        return []
    if "ExperimentLog" not in wb.sheetnames:
        wb.close()
        return []
    ws = wb["ExperimentLog"]
    headers = [c.value for c in ws[1]]
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        rows.append(dict(zip(headers, row)))
    wb.close()
    return rows


def merge_unique_by_session(existing, new_rows):
    seen = {r.get("session_id") for r in existing}
    for row in new_rows:
        if row.get("session_id") and row["session_id"] in seen:
            continue
        existing.append(row)
        seen.add(row.get("session_id"))
    return existing


def build():
    manifests = load_manifests()
    new_rows = experiment_rows(manifests)
    exp_rows = merge_unique_by_session(_load_existing_log(), new_rows)

    wb = Workbook()
    default = wb.active
    if default is not None:
        wb.remove(default)
    write_table(wb, "ExperimentLog", exp_rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"[OK] {OUT} ({len(manifests)} run, {len(exp_rows)} baris experiment log)")
    return OUT


if __name__ == "__main__":
    build()