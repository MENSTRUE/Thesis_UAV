"""Entrypoint sederhana: menu interaktif, tanpa perlu hafal parameter."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEOS = [ROOT / "videos", ROOT / "data" / "videos"]


def _pick_video():
    candidates = []
    for folder in DEFAULT_VIDEOS:
        if folder.is_dir():
            candidates.extend(sorted(p for p in folder.rglob("*.mp4")))
    if not candidates:
        print("[!] Tidak ada video di:", ", ".join(str(p) for p in DEFAULT_VIDEOS))
        print("    Simpan video ke folder videos/ atau lewati dan pilih sumber lain.")
        return None
    print("Pilih video:")
    for index, video in enumerate(candidates):
        print(f"  {index + 1}. {video.relative_to(ROOT)}")
    while True:
        raw = input("Nomor (Enter = pertama): ").strip()
        if not raw:
            return str(candidates[0])
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return str(candidates[int(raw) - 1])
        print("Nomor tidak valid.")


def _ort_providers():
    sys.path.insert(0, str(ROOT / "src"))
    from runtime_utils import resolve_ort_providers, setup_cuda_paths
    setup_cuda_paths()
    try:
        return resolve_ort_providers()
    except Exception:
        return ["CPUExecutionProvider"]


def _auto_profile() -> str:
    if any(p in _ort_providers() for p in ("CUDAExecutionProvider", "DmlExecutionProvider")):
        return "laptop"
    return "nano"


def _check_gpu():
    """Cetak status GPU di startup; peringatkan bila GPU tidak aktif."""
    providers = _ort_providers()
    if "CUDAExecutionProvider" in providers:
        print("[GPU] Aktif: onnxruntime CUDAExecutionProvider")
    elif "DmlExecutionProvider" in providers:
        print("[GPU] Aktif: onnxruntime DmlExecutionProvider (DirectML)")
    else:
        print("[!] GPU tidak terdeteksi (onnxruntime CPU saja).")
        print("    Jalankan scripts/setup.ps1 (default GPU) atau -DML untuk DirectML.")
        print("    Lanjut dengan CPU...\n")


def run_pipeline(argv):
    _check_gpu()
    from full_pipeline import main

    main(argv)


def run_diagnose():
    sys.path.insert(0, str(ROOT / "src"))
    import diagnose_runtime

    diagnose_runtime.main()


def menu():
    mode = None
    while True:
        print()
        print("1. Tello (drone) - default")
        print("2. Video file")
        print("3. Webcam")
        print("0. Diagnosa runtime")
        raw = input("Pilih [1]: ").strip()
        if not raw:
            mode = "tello"
            break
        if raw in {"1", "2", "3", "0"}:
            mode = {"1": "tello", "2": "video", "3": "webcam", "0": "diagnose"}[raw]
            break
        print("Pilihan tidak valid.")
    if mode == "diagnose":
        run_diagnose()
        return
    args = ["--source", mode, "--profile", _auto_profile(),
            "--detector-imgsz", "512", "--pose-imgsz", "512"]
    if mode == "tello":
        print()
        print("Tombol: SPACE=naik/turun, WASD=gerak, T=takeoff, L=land, Q=foto,")
        print("        E=rekam, C/X=kecepatan, []=trim, F=emergency, TAB=reset trim")
        args.append("--enable-face")
        args.append("--allow-takeoff")
    if mode == "video":
        video = _pick_video()
        if video is None:
            return
        args += ["--video", video]
        args.append("--max-frames")
        args.append("0")
    args.append("--enable-face")
    print(f"\nJalankan: python main.py {' '.join(args)}")
    print("Tekan ESC untuk keluar.")
    run_pipeline(args)


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            direct = sys.argv[1]
            if direct in {"tello", "video", "webcam"}:
                args = ["--source", direct, "--profile", _auto_profile(),
                        "--detector-imgsz", "512", "--pose-imgsz", "512"]
                if direct == "video":
                    video = _pick_video()
                    if video is None:
                        sys.exit(1)
                    args += ["--video", video]
                args.append("--enable-face")
                if direct == "tello":
                    args.append("--allow-takeoff")
                run_pipeline(args)
            elif direct == "diagnose":
                run_diagnose()
            else:
                print("Gunakan: python main.py [tello|video|webcam|diagnose]")
                sys.exit(1)
        else:
            menu()
    except KeyboardInterrupt:
        print("\nDihentikan pengguna.")
