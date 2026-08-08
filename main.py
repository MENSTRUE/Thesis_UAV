"""Entrypoint realtime HAR UAV.

Mode utama yang disarankan untuk pengujian skripsi adalah ``experiment``.
Mode ini TIDAK mengubah model HAR maupun modul wajah. Ia hanya membuat
pengujian 1/2/3 orang + stress test 3+ lebih terstruktur: folder output jelas, warm-up otomatis,
measurement otomatis, dan 3 pengulangan tidak tercampur.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DEFAULT_VIDEOS = [ROOT / "videos", ROOT / "data" / "videos"]


def _ensure_src_path() -> None:
    value = str(SRC)
    if value not in sys.path:
        sys.path.insert(0, value)


def _pick_video():
    candidates = []
    for folder in DEFAULT_VIDEOS:
        if folder.is_dir():
            candidates.extend(sorted(p for p in folder.rglob("*.mp4")))
    if not candidates:
        print("[!] Tidak ada video di:", ", ".join(str(p) for p in DEFAULT_VIDEOS))
        print("    Simpan video ke folder videos/ atau data/videos/.")
        return None

    print("\nPilih video:")
    for index, video in enumerate(candidates, 1):
        print(f"  {index}. {video.relative_to(ROOT)}")
    while True:
        raw = input("Nomor (Enter = pertama): ").strip()
        if not raw:
            return str(candidates[0])
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return str(candidates[int(raw) - 1])
        print("Nomor tidak valid.")


def _ort_providers():
    _ensure_src_path()
    from runtime_utils import resolve_ort_providers, setup_cuda_paths

    setup_cuda_paths()
    try:
        return resolve_ort_providers()
    except Exception:
        return ["CPUExecutionProvider"]


def _auto_profile() -> str:
    providers = _ort_providers()
    if any(p in providers for p in ("CUDAExecutionProvider", "DmlExecutionProvider")):
        return "laptop"
    return "nano"


def _check_gpu():
    providers = _ort_providers()
    if "CUDAExecutionProvider" in providers:
        print("[GPU] Aktif: ONNX Runtime CUDAExecutionProvider")
    elif "DmlExecutionProvider" in providers:
        print("[GPU] Aktif: ONNX Runtime DmlExecutionProvider (DirectML)")
    else:
        print("[!] GPU tidak terdeteksi; ONNX Runtime berjalan di CPU.")
        print("    Pengujian tetap bisa jalan, tetapi FPS kemungkinan lebih rendah.\n")


def run_pipeline(argv):
    _ensure_src_path()
    _check_gpu()
    from full_pipeline import main

    main(argv)


def run_diagnose():
    _ensure_src_path()
    import diagnose_runtime

    diagnose_runtime.main()


def _base_args(source: str) -> list[str]:
    return [
        "--source", source,
        "--profile", _auto_profile(),
        "--detector-imgsz", "512",
        "--pose-imgsz", "512",
        "--enable-face",  # face tetap aktif; implementasi face_system.py tidak disentuh
    ]


def _ask_people() -> int:
    """Skenario utama 1/2/3 orang + opsi stress test 3+ orang."""
    while True:
        print("\nJumlah manusia:")
        print("  1. 1 orang (skenario utama BAB IV)")
        print("  2. 2 orang (skenario utama BAB IV)")
        print("  3. 3 orang (skenario utama BAB IV)")
        print("  4. 3+ orang / stress test")
        raw = input("Pilih [1/2/3/4]: ").strip()
        if raw in {"1", "2", "3"}:
            return int(raw)
        if raw == "4":
            while True:
                custom = input("Jumlah pasti manusia untuk stress test [4-20]: ").strip()
                if custom.isdigit() and 4 <= int(custom) <= 20:
                    return int(custom)
                print("Masukkan angka 4 sampai 20.")
        print("Pilihan tidak valid.")


ACTIVITY_LABELS = [
    "boxing", "carrying", "clapping", "digging", "jogging",
    "running", "throwing", "walking", "waving",
]


def _ask_activity() -> str:
    print("\nMode aktivitas:")
    print("  1. Tetap / satu label (disarankan untuk uji performa 1 vs 2 vs 3 orang)")
    print("  2. Campuran / random (sistem menyimpan semua label yang TERDETEKSI)")
    mode = input("Pilih [1/2, default=1]: ").strip() or "1"
    if mode == "2":
        print("[INFO] Mode mixed/random: prediksi aktual per subjek tetap disimpan ke detections.csv")
        print("       dan diringkas ke BAB4_ACTIVITY_TIMELINE.csv. Ini BUKAN ground truth akurasi.")
        return "mixed/random"

    print("\nPilih label aktivitas tetap:")
    for i, label in enumerate(ACTIVITY_LABELS, 1):
        print(f"  {i}. {label}")
    raw = input("Nomor/nama label [9=waving]: ").strip().lower()
    if not raw:
        return "waving"
    if raw.isdigit() and 1 <= int(raw) <= len(ACTIVITY_LABELS):
        return ACTIVITY_LABELS[int(raw) - 1]
    if raw in ACTIVITY_LABELS:
        return raw
    print(f"[!] Label '{raw}' tidak dikenal. Dipakai default: waving")
    return "waving"


def run_experiment_menu():
    people = _ask_people()
    activity = _ask_activity()

    print("\n" + "=" * 72)
    print("MODE PENGUJIAN SKRIPSI UAV")
    print(f"Skenario        : {people} orang")
    print(f"Aktivitas plan  : {activity}")
    if activity == "mixed/random":
        print("Pencatatan      : semua label prediksi aktual disimpan per subjek/per detik")
    print("Pengulangan     : 3 kali")
    print("Warm-up         : 5 detik")
    print("Measurement     : 30 detik")
    print("Face/Liveness   : AKTIF (kode face tidak diubah)")
    print("Kontrol segmen  : tekan E sekali -> warm-up -> ukur -> STOP otomatis")
    print("Folder          : output/uav_final/<N>_orang/run_.../rep_XX/")
    if people > 3:
        print("Mode            : STRESS TEST 3+ (tambahan; tidak mengganti skenario utama 1/2/3)")
    print("=" * 72)
    input("Tekan Enter untuk lanjut...")

    args = _base_args("tello")
    args += [
        "--allow-takeoff",
        "--experiment-people", str(people),
        "--experiment-repetitions", "3",
        "--experiment-warmup", "5",
        "--experiment-duration", "30",
        "--experiment-activity", activity,
    ]
    run_pipeline(args)


def run_tello_manual():
    print("\nMode Tello manual.")
    print("E = START/STOP segmen manual. Tidak ada warm-up/auto-stop eksperimen.")
    args = _base_args("tello") + ["--allow-takeoff"]
    run_pipeline(args)


def run_video_file():
    video = _pick_video()
    if video is None:
        return
    args = _base_args("video") + ["--video", video, "--max-frames", "0"]
    run_pipeline(args)


def run_webcam():
    run_pipeline(_base_args("webcam"))


def menu():
    while True:
        print("\n" + "=" * 60)
        print("HAR UAV REALTIME")
        print("1. Tello - PENGUJIAN SKRIPSI 1/2/3 orang + 3+ stress test")
        print("2. Tello - manual / coba sistem")
        print("3. Video file")
        print("4. Webcam")
        print("0. Diagnosa runtime")
        print("=" * 60)
        raw = input("Pilih [1]: ").strip() or "1"

        if raw == "1":
            run_experiment_menu()
            return
        if raw == "2":
            run_tello_manual()
            return
        if raw == "3":
            run_video_file()
            return
        if raw == "4":
            run_webcam()
            return
        if raw == "0":
            run_diagnose()
            return
        print("Pilihan tidak valid.")


def direct_mode(argv: list[str]) -> None:
    direct = argv[0].lower()

    if direct in {"experiment", "uji", "test"}:
        run_experiment_menu()
        return

    if direct == "tello":
        run_tello_manual()
        return

    if direct == "video":
        run_video_file()
        return

    if direct == "webcam":
        run_webcam()
        return

    if direct == "diagnose":
        run_diagnose()
        return

    print("Gunakan:")
    print("  python main.py")
    print("  python main.py experiment")
    print("  python main.py tello")
    print("  python main.py video")
    print("  python main.py webcam")
    print("  python main.py diagnose")
    raise SystemExit(1)


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            direct_mode(sys.argv[1:])
        else:
            menu()
    except KeyboardInterrupt:
        print("\nDihentikan pengguna.")
