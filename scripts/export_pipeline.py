"""Generator report BAB IV untuk eksperimen UAV 1/2/3 orang + stress test 3+.

Tidak menyentuh model atau modul face. Script ini hanya membaca hasil yang sudah
tersimpan di ``output/uav_final`` lalu membuat report yang mudah dipakai di skripsi:

- REPORT_BAB4.html         : report visual yang bisa dibuka di browser
- REPORT_BAB4.json         : data ringkasan terstruktur
- BAB4_REPETITIONS.csv     : satu baris per pengulangan
- BAB4_SUBJECTS.csv        : satu baris per subject_id per pengulangan
- BAB4_ACTIVITY_TIMELINE.csv: aktivitas dominan aktual per subject_id per detik

Fungsi ``build()`` sengaja tanpa argumen karena dipanggil otomatis oleh
``full_pipeline.py`` setelah satu sesi eksperimen selesai.
"""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
UAV_ROOT = ROOT / "output" / "uav_final"

def _scenario_dirs():
    """Temukan otomatis 1_orang, 2_orang, 3_orang, 4_orang, dst."""
    found = []
    if not UAV_ROOT.is_dir():
        return found
    for folder in UAV_ROOT.iterdir():
        if not folder.is_dir():
            continue
        m = __import__("re").fullmatch(r"(\d+)_orang", folder.name)
        if m:
            found.append((int(m.group(1)), folder.name))
    return sorted(found)


def _read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _float(value, default=None):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(vals) if vals else None


def _sd(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.stdev(vals) if len(vals) >= 2 else (0.0 if vals else None)


def _fmt(value, digits=2, suffix="") -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_mean_sd(values: Iterable[float], digits=2, suffix="") -> str:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return "-"
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return f"{m:.{digits}f} ± {s:.{digits}f}{suffix}"


def _latest_run(scenario_dir: Path) -> Optional[Path]:
    if not scenario_dir.is_dir():
        return None
    runs = sorted(
        [p for p in scenario_dir.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.name,
    )
    return runs[-1] if runs else None


def _dominant(counter: Counter) -> tuple[str, int, float]:
    total = sum(counter.values())
    if not total:
        return "", 0, 0.0
    name, count = counter.most_common(1)[0]
    return name, count, count / total * 100.0


def _rep_subject_rows(people: int, run_id: str, rep_idx: int, rep_dir: Path) -> List[dict]:
    rows = _read_csv(rep_dir / "detections.csv")
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        sid = str(row.get("subject_id") or row.get("byte_track_id") or "").strip()
        if sid:
            grouped[sid].append(row)

    out = []
    for sid, items in sorted(grouped.items(), key=lambda kv: (_int(kv[0], 10**9), kv[0])):
        activities = Counter()
        identities = Counter()
        liveness = Counter()
        activity_scores = []
        face_sims = []
        byte_ids = set()

        for row in items:
            activity = str(row.get("activity") or "").strip()
            if activity and activity not in {"collecting", ""}:
                activities[activity] += 1
            identity = str(row.get("face_identity") or "").strip()
            if identity:
                identities[identity] += 1
            live = str(row.get("liveness") or "").strip()
            if live:
                liveness[live] += 1
            score = _float(row.get("activity_score"))
            if score is not None and activity not in {"collecting", ""}:
                activity_scores.append(score * 100.0)
            sim = _float(row.get("face_similarity"))
            if sim is not None:
                face_sims.append(sim)
            bid = str(row.get("byte_track_id") or "").strip()
            if bid:
                byte_ids.add(bid)

        act_name, _, act_pct = _dominant(activities)
        total_act = sum(activities.values())
        activity_distribution = "; ".join(
            f"{name}:{count / total_act * 100.0:.1f}%"
            for name, count in activities.most_common()
        ) if total_act else ""

        # Urutan label ketika prediksi dominan berubah (tanpa spam label yang sama).
        activity_sequence = []
        last_act = None
        for row in sorted(items, key=lambda r: _float(r.get("t_segment_s"), 0.0) or 0.0):
            a = str(row.get("activity") or "").strip()
            if not a or a == "collecting" or a == last_act:
                continue
            activity_sequence.append(a)
            last_act = a
        id_name, _, id_pct = _dominant(identities)
        live_name, _, live_pct = _dominant(liveness)
        out.append({
            "scenario_people": people,
            "scenario_type": "utama" if people <= 3 else "stress_test_3plus",
            "run_id": run_id,
            "repetition": rep_idx,
            "subject_id": sid,
            "byte_track_ids": ";".join(sorted(byte_ids, key=lambda x: (_int(x, 10**9), x))),
            "dominant_activity": act_name or "-",
            "dominant_activity_pct": round(act_pct, 2),
            "activity_distribution": activity_distribution,
            "activity_sequence": " -> ".join(activity_sequence),
            "mean_activity_confidence_pct": round(_mean(activity_scores), 2) if activity_scores else "",
            "face_identity": id_name or "-",
            "face_identity_pct": round(id_pct, 2),
            "mean_face_similarity": round(_mean(face_sims), 3) if face_sims else "",
            "dominant_liveness": live_name or "-",
            "dominant_liveness_pct": round(live_pct, 2),
            "detection_rows": len(items),
        })
    return out


def _rep_activity_timeline_rows(people: int, run_id: str, rep_idx: int, rep_dir: Path) -> List[dict]:
    """Ringkas prediksi aktual menjadi satu baris per subject_id per detik.

    Ini adalah OBSERVED/PREDICTED activity, bukan ground truth. Sangat berguna
    ketika subjek melakukan gerakan campuran/random selama eksperimen.
    """
    rows = _read_csv(rep_dir / "detections.csv")
    grouped = defaultdict(list)
    for row in rows:
        sid = str(row.get("subject_id") or row.get("byte_track_id") or "").strip()
        activity = str(row.get("activity") or "").strip()
        if not sid or not activity or activity == "collecting":
            continue
        sec = _int(row.get("t_segment_s"), -1)
        if sec < 0:
            continue
        grouped[(sid, sec)].append(row)

    out = []
    for (sid, sec), items in sorted(grouped.items(), key=lambda kv: (_int(kv[0][0], 10**9), kv[0][1])):
        acts = Counter(str(r.get("activity") or "").strip() for r in items)
        acts.pop("", None); acts.pop("collecting", None)
        act_name, _, act_pct = _dominant(acts)
        scores = []
        byte_ids = set()
        identities = Counter()
        liveness = Counter()
        for r in items:
            sc = _float(r.get("activity_score"))
            if sc is not None:
                scores.append(sc * 100.0)
            bid = str(r.get("byte_track_id") or "").strip()
            if bid:
                byte_ids.add(bid)
            ident = str(r.get("face_identity") or "").strip()
            if ident:
                identities[ident] += 1
            live = str(r.get("liveness") or "").strip()
            if live:
                liveness[live] += 1
        id_name, _, _ = _dominant(identities)
        live_name, _, _ = _dominant(liveness)
        out.append({
            "scenario_people": people,
            "scenario_type": "utama" if people <= 3 else "stress_test_3plus",
            "run_id": run_id,
            "repetition": rep_idx,
            "subject_id": sid,
            "second": sec,
            "byte_track_ids": ";".join(sorted(byte_ids, key=lambda x: (_int(x, 10**9), x))),
            "detected_activity": act_name or "-",
            "activity_vote_pct": round(act_pct, 2),
            "mean_activity_confidence_pct": round(_mean(scores), 2) if scores else "",
            "face_identity": id_name or "-",
            "liveness": live_name or "-",
            "n_detection_rows": len(items),
        })
    return out


def _collect() -> tuple[List[dict], List[dict], List[dict], List[dict]]:
    repetitions: List[dict] = []
    subjects: List[dict] = []
    timelines: List[dict] = []
    scenarios: List[dict] = []

    for people, folder_name in _scenario_dirs():
        run_dir = _latest_run(UAV_ROOT / folder_name)
        if run_dir is None:
            continue

        stats_rows = _read_csv(run_dir / "segment_stats.csv")
        stats_by_rep = {_int(r.get("segment_index")): r for r in stats_rows}
        session = _read_json(run_dir / "session_benchmark.json")
        activity_plan = (
            session.get("experiment", {}).get("activity_plan")
            or next((r.get("activity_plan") for r in stats_rows if r.get("activity_plan")), "")
        )

        rep_dirs = sorted([p for p in run_dir.glob("rep_*") if p.is_dir()])
        for rep_dir in rep_dirs:
            try:
                rep_idx = int(rep_dir.name.split("_")[-1])
            except ValueError:
                continue
            stat = stats_by_rep.get(rep_idx, {})
            bench = _read_json(rep_dir / "benchmark.json")
            rep = {
                "scenario_people": people,
                "scenario_type": "utama" if people <= 3 else "stress_test_3plus",
                "run_id": run_dir.name,
                "repetition": rep_idx,
                "activity_plan": activity_plan,
                "duration_s": _float(stat.get("duration_s"), _float(bench.get("measurement_seconds"))),
                "processed_frames": _int(stat.get("processed_frames"), _int(bench.get("measurement_frames"))),
                "n_people_avg": _float(stat.get("n_people_avg")),
                "n_people_max": _int(stat.get("n_people_max")),
                "people_match_ratio_pct": _float(stat.get("people_match_ratio_pct")),
                "fps_mean": _float(stat.get("fps_mean")),
                "fps_p95": _float(stat.get("fps_p95")),
                "throughput_fps": _float(stat.get("throughput_fps")),
                "ms_total_mean": _float(stat.get("ms_total_mean")),
                "ms_detector_mean": _float(stat.get("ms_detector_mean")),
                "ms_bytetrack_mean": _float(stat.get("ms_bytetrack_mean")),
                "ms_pose_mean": _float(stat.get("ms_pose_mean")),
                "ms_body110_har_mean": _float(stat.get("ms_body110_har_mean")),
                "ms_face_liveness_mean": _float(stat.get("ms_face_liveness_mean")),
                "pose_valid_ratio": _float(stat.get("pose_valid_ratio")),
                "dominant_activity": stat.get("dominant_activity", ""),
                "dominant_activity_pct": _float(stat.get("dominant_activity_pct")),
                "dominant_face": stat.get("dominant_face", ""),
                "dominant_face_pct": _float(stat.get("dominant_face_pct")),
                "real": _int(stat.get("real")),
                "spoof": _int(stat.get("spoof")),
                "unknown": _int(stat.get("unknown")),
                "video": str(rep_dir / "recording.mp4"),
                "benchmark": str(rep_dir / "benchmark.json"),
            }
            repetitions.append(rep)
            subjects.extend(_rep_subject_rows(people, run_dir.name, rep_idx, rep_dir))
            timelines.extend(_rep_activity_timeline_rows(people, run_dir.name, rep_idx, rep_dir))

        reps_this = [r for r in repetitions if r["scenario_people"] == people and r["run_id"] == run_dir.name]
        if reps_this:
            scenarios.append({
                "scenario_people": people,
                "scenario_type": "utama" if people <= 3 else "stress_test_3plus",
                "run_id": run_dir.name,
                "n_repetitions": len(reps_this),
                "activity_plan": activity_plan,
                "fps_mean_sd": _fmt_mean_sd([r["fps_mean"] for r in reps_this if r["fps_mean"] is not None]),
                "throughput_mean_sd": _fmt_mean_sd([r["throughput_fps"] for r in reps_this if r["throughput_fps"] is not None]),
                "latency_mean_sd_ms": _fmt_mean_sd([r["ms_total_mean"] for r in reps_this if r["ms_total_mean"] is not None], suffix=" ms"),
                "people_mean_sd": _fmt_mean_sd([r["n_people_avg"] for r in reps_this if r["n_people_avg"] is not None]),
                "people_match_mean_sd_pct": _fmt_mean_sd([r["people_match_ratio_pct"] for r in reps_this if r["people_match_ratio_pct"] is not None], suffix="%"),
                "pose_mean_sd_ms": _fmt_mean_sd([r["ms_pose_mean"] for r in reps_this if r["ms_pose_mean"] is not None], suffix=" ms"),
                "har_mean_sd_ms": _fmt_mean_sd([r["ms_body110_har_mean"] for r in reps_this if r["ms_body110_har_mean"] is not None], suffix=" ms"),
                "face_mean_sd_ms": _fmt_mean_sd([r["ms_face_liveness_mean"] for r in reps_this if r["ms_face_liveness_mean"] is not None], suffix=" ms"),
            })

    repetitions.sort(key=lambda r: (r["scenario_people"], r["repetition"]))
    subjects.sort(key=lambda r: (r["scenario_people"], r["repetition"], _int(r["subject_id"], 10**9)))
    scenarios.sort(key=lambda r: r["scenario_people"])
    timelines.sort(key=lambda r: (r["scenario_people"], r["repetition"], _int(r["subject_id"], 10**9), r["second"]))
    return scenarios, repetitions, subjects, timelines


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _td(value) -> str:
    return f"<td>{html.escape(str(value if value not in (None, '') else '-'))}</td>"


def _table(headers: List[str], rows: List[List[object]]) -> str:
    h = "".join(f"<th>{html.escape(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(_td(v) for v in row) + "</tr>" for row in rows)
    if not body:
        body = f'<tr><td colspan="{len(headers)}" class="empty">Belum ada data.</td></tr>'
    return f"<div class='table-wrap'><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>"


def _build_html(scenarios: List[dict], repetitions: List[dict], subjects: List[dict], timelines: List[dict]) -> str:
    scenario_rows = [[
        f"{r['scenario_people']} orang", r["n_repetitions"], r["activity_plan"] or "-",
        r["people_mean_sd"], r["people_match_mean_sd_pct"], r["fps_mean_sd"],
        r["latency_mean_sd_ms"], r["pose_mean_sd_ms"], r["har_mean_sd_ms"],
        r["face_mean_sd_ms"],
    ] for r in scenarios]

    rep_rows = [[
        f"{r['scenario_people']} orang", r["repetition"], r["activity_plan"] or "-",
        _fmt(r["duration_s"]), _fmt(r["n_people_avg"]), _fmt(r["people_match_ratio_pct"], suffix="%"),
        _fmt(r["fps_mean"]), _fmt(r["ms_total_mean"], suffix=" ms"),
        _fmt(r["ms_detector_mean"], suffix=" ms"), _fmt(r["ms_bytetrack_mean"], suffix=" ms"),
        _fmt(r["ms_pose_mean"], suffix=" ms"), _fmt(r["ms_body110_har_mean"], suffix=" ms"),
        _fmt(r["ms_face_liveness_mean"], suffix=" ms"), r["dominant_activity"] or "-",
        _fmt(r["dominant_activity_pct"], suffix="%"), r["dominant_face"] or "-",
    ] for r in repetitions]

    subject_rows = [[
        f"{r['scenario_people']} orang", r["repetition"], r["subject_id"], r["byte_track_ids"] or "-",
        r["dominant_activity"], _fmt(r["dominant_activity_pct"], suffix="%"),
        r["activity_distribution"] or "-", r["activity_sequence"] or "-",
        _fmt(r["mean_activity_confidence_pct"], suffix="%"), r["face_identity"],
        _fmt(r["mean_face_similarity"], digits=3), r["dominant_liveness"],
        _fmt(r["dominant_liveness_pct"], suffix="%"), r["detection_rows"],
    ] for r in subjects]

    timeline_rows = [[
        f"{r['scenario_people']} orang", r["repetition"], r["subject_id"], r["second"],
        r["detected_activity"], _fmt(r["activity_vote_pct"], suffix="%"),
        _fmt(r["mean_activity_confidence_pct"], suffix="%"), r["face_identity"],
        r["liveness"], r["byte_track_ids"] or "-",
    ] for r in timelines]

    identity_names = sorted({r["face_identity"] for r in subjects if r["face_identity"] not in {"", "-", "unknown"}})
    activity_names = sorted({r["dominant_activity"] for r in subjects if r["dominant_activity"] not in {"", "-", "collecting"}})
    scenarios_done = len(scenarios)

    return f"""<!doctype html>
<html lang='id'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Report BAB IV + Stress Test - Pengujian UAV</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#657089;--line:#dfe5ef;--accent:#214f86;--good:#e9f7ef}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Arial,sans-serif}}
main{{max-width:1500px;margin:28px auto;padding:0 20px 48px}} h1{{font-size:28px;margin:0 0 6px}} h2{{margin-top:32px;border-bottom:2px solid var(--line);padding-bottom:8px}}
.subtitle{{color:var(--muted);margin-bottom:20px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}} .card b{{display:block;font-size:22px;margin-top:4px;color:var(--accent)}}
.table-wrap{{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}} table{{border-collapse:collapse;width:100%;min-width:900px}}
th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}} th{{position:sticky;top:0;background:#eef3f9;color:#1c3656}} tr:nth-child(even) td{{background:#fafbfd}}
.note{{background:#fff9e8;border:1px solid #ead79c;border-radius:10px;padding:12px 14px;margin:12px 0}} .empty{{text-align:center;color:var(--muted)}} code{{background:#eef1f5;padding:2px 5px;border-radius:4px}}
@media print{{body{{background:white}} main{{max-width:none;margin:0;padding:0}} .table-wrap{{overflow:visible}} th{{position:static}}}}
</style>
</head>
<body><main>
<h1>Report Pengujian Sistem UAV — BAB IV</h1>
<div class='subtitle'>Dibuat otomatis dari measurement 1/2/3 orang + stress test. Warm-up tidak dihitung. Prediksi aktivitas aktual tetap disimpan meskipun gerakan campuran/random.</div>
<div class='cards'>
  <div class='card'>Skenario tersedia<b>{scenarios_done}/3</b></div>
  <div class='card'>Total pengulangan<b>{len(repetitions)}</b></div>
  <div class='card'>Label aktivitas teramati<b>{len(activity_names)}</b></div>
  <div class='card'>Identitas terdaftar teramati<b>{len(identity_names)}</b></div>
</div>
<div class='note'><b>Interpretasi:</b> <code>activity_plan</code> = rencana/ground-truth protokol bila gerakan tetap; <code>detected_activity</code> = hasil prediksi model yang benar-benar terdeteksi. Pada mode mixed/random, detected_activity tetap disimpan tetapi tidak boleh dianggap sebagai ground truth akurasi. <code>byte_track_id</code> = ID ByteTrack sementara; <code>subject_id</code> = ID subjek stabil; <code>face_identity</code> = nama dari database InsightFace atau <code>unknown</code>.</div>

<h2>A. Ringkasan 1 vs 2 vs 3 Orang</h2>
{_table(['Skenario','Rep','Aktivitas Plan','Rata-rata Orang','Frame Sesuai Skenario','FPS Mean ± SD','Latency Mean ± SD','Pose Mean ± SD','Body110+HAR Mean ± SD','Face+Liveness Mean ± SD'], scenario_rows)}

<h2>B. Detail Setiap Pengulangan</h2>
{_table(['Skenario','Rep','Aktivitas Plan','Durasi (s)','Orang Avg','Sesuai Skenario','FPS Mean','Total/frame','Detector','ByteTrack','Pose total/frame','Body110+HAR total/frame','Face+Liveness','Aktivitas Dominan','Aktivitas Dominan %','Wajah Dominan'], rep_rows)}

<h2>C. Aktivitas dan Identitas per Subjek</h2>
{_table(['Skenario','Rep','Subject ID','ByteTrack ID','Aktivitas Dominan','Dominan %','Distribusi Semua Aktivitas','Urutan Perubahan Label','Confidence Avg','Identitas Wajah','Similarity Avg','Liveness Dominan','Liveness %','Baris Deteksi'], subject_rows)}

<h2>D. Timeline Aktivitas Aktual per Subjek per Detik</h2>
{_table(['Skenario','Rep','Subject ID','Detik','Aktivitas Terdeteksi','Vote %','Confidence Avg','Identitas Wajah','Liveness','ByteTrack ID'], timeline_rows)}

<h2>E. File yang Dipakai untuk BAB IV</h2>
<div class='card'>
<b style='font-size:16px'>Gambar 4.4–4.6</b>: ambil frame representatif dari <code>rep_XX/recording.mp4</code>.<br>
<b style='font-size:16px'>Tabel performa</b>: gunakan bagian A dan B pada report ini.<br>
<b style='font-size:16px'>Bukti label gerakan + identitas</b>: gunakan bagian C/D, <code>BAB4_ACTIVITY_TIMELINE.csv</code>, dan <code>rep_XX/detections.csv</code>.<br>
<b style='font-size:16px'>Data mentah</b>: tetap tersedia pada <code>detections.csv</code>, <code>frames.csv</code>, dan <code>benchmark.json</code>.
</div>
</main></body></html>"""


def _filter_current_run(scenarios, repetitions, subjects, timelines, current_run: Path, current_people: int):
    run_id = current_run.name
    reps = [r for r in repetitions if r.get("scenario_people") == current_people and r.get("run_id") == run_id]
    subs = [r for r in subjects if r.get("scenario_people") == current_people and r.get("run_id") == run_id]
    tls = [r for r in timelines if r.get("scenario_people") == current_people and r.get("run_id") == run_id]
    scs = [r for r in scenarios if r.get("scenario_people") == current_people and r.get("run_id") == run_id]
    return scs, reps, subs, tls


def _write_current_run_summary(current_run: Path, current_people: int, scenarios, repetitions, subjects, timelines) -> Path:
    current_run.mkdir(parents=True, exist_ok=True)
    scs, reps, subs, tls = _filter_current_run(
        scenarios, repetitions, subjects, timelines, current_run, current_people
    )
    prefix = f"SUMMARY_{current_people}_ORANG"

    _write_csv(current_run / f"{prefix}_REPETITIONS.csv", reps)
    _write_csv(current_run / f"{prefix}_SUBJECTS.csv", subs)
    _write_csv(current_run / f"{prefix}_ACTIVITY_TIMELINE.csv", tls)

    payload = {
        "scenario_people": current_people,
        "run_id": current_run.name,
        "scenarios": scs,
        "repetitions": reps,
        "subjects": subs,
        "activity_timeline": tls,
        "notes": {
            "generated_incrementally": True,
            "face_system_modified": False,
            "activity_field": "activity",
            "activity_score_field": "activity_score",
            "stable_subject_field": "subject_id",
            "temporary_track_field": "byte_track_id",
            "face_identity_field": "face_identity",
            "face_similarity_field": "face_similarity",
            "liveness_fields": ["liveness", "liveness_score"],
        },
    }
    (current_run / f"{prefix}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    html_path = current_run / f"{prefix}.html"
    html_path.write_text(_build_html(scs, reps, subs, tls), encoding="utf-8")

    # Shortcut pada folder skenario agar tidak perlu masuk ke run_... untuk mencari summary terbaru.
    scenario_dir = current_run.parent
    latest_prefix = f"SUMMARY_TERBARU_{current_people}_ORANG"
    (scenario_dir / f"{latest_prefix}.html").write_text(
        _build_html(scs, reps, subs, tls), encoding="utf-8"
    )
    (scenario_dir / f"{latest_prefix}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(scenario_dir / f"{latest_prefix}_REPETITIONS.csv", reps)
    _write_csv(scenario_dir / f"{latest_prefix}_SUBJECTS.csv", subs)
    _write_csv(scenario_dir / f"{latest_prefix}_ACTIVITY_TIMELINE.csv", tls)
    return html_path


def build(current_run: Optional[Path] = None, current_people: Optional[int] = None) -> Path:
    """Bangun report global BAB IV dan, bila diberikan, summary run saat ini.

    Report global selalu ditulis ke ``output/uav_final``. Summary skenario/run
    ditulis segera agar setelah REP 1 selesai pengguna sudah dapat melihat hasil,
    tanpa harus menutup program terlebih dahulu.
    """
    UAV_ROOT.mkdir(parents=True, exist_ok=True)
    scenarios, repetitions, subjects, timelines = _collect()

    _write_csv(UAV_ROOT / "BAB4_REPETITIONS.csv", repetitions)
    _write_csv(UAV_ROOT / "BAB4_SUBJECTS.csv", subjects)
    _write_csv(UAV_ROOT / "BAB4_ACTIVITY_TIMELINE.csv", timelines)

    payload = {
        "scenarios": scenarios,
        "repetitions": repetitions,
        "subjects": subjects,
        "activity_timeline": timelines,
        "notes": {
            "face_system_modified": False,
            "activity_field": "activity",
            "activity_score_field": "activity_score",
            "stable_subject_field": "subject_id",
            "temporary_track_field": "byte_track_id",
            "face_identity_field": "face_identity",
            "face_similarity_field": "face_similarity",
            "liveness_fields": ["liveness", "liveness_score"],
        },
    }
    (UAV_ROOT / "REPORT_BAB4.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    html_path = UAV_ROOT / "REPORT_BAB4.html"
    html_path.write_text(_build_html(scenarios, repetitions, subjects, timelines), encoding="utf-8")

    if current_run is not None and current_people is not None:
        _write_current_run_summary(
            Path(current_run), int(current_people), scenarios, repetitions, subjects, timelines
        )
    return html_path


if __name__ == "__main__":
    print(build().resolve())
