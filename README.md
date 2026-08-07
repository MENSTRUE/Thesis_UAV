# Thesis UAV — Human Activity Recognition and Face Identification

Implementasi penelitian pengenalan aktivitas manusia pada video UAV menggunakan
YOLOv8s, ByteTrack, YOLO26s-Pose, representasi fitur Body110, CNN-BiLSTM ONNX,
InsightFace, dan MiniFASNetV2.

DJI Tello digunakan sebagai kamera dan pengirim video. Inferensi dijalankan
secara **off-board** pada laptop atau NVIDIA Jetson yang berfungsi sebagai
ground station.

## Alur sistem

```text
Video/Tello
  -> OpenCV / PyAV
  -> YOLOv8s person detection
  -> ByteTrack (Track ID)
  -> crop manusia
  -> YOLO26s-Pose (17 keypoint)
  -> Raw51: 17 x (x, y, confidence)
  -> sequence 30 frame
  -> Body110
  -> CNN-BiLSTM ONNX
  -> kelas aktivitas + confidence
  -> InsightFace + MiniFASNetV2 (opsional)
```

![Alur pipeline](docs/images/alur_pipeline.png)

## Struktur repository

```text
drone-human_gesture/
├── main.py                 # entrypoint sederhana (menu interaktif)
├── notebooks/
├── src/
│   ├── full_pipeline.py    # pipeline penuh off-board
│   ├── tello_control.py    # kontrol Tello + HUD + foto/rekaman
│   ├── face_system.py      # face recognition + anti-spoofing
│   ├── runtime_utils.py    # pemilihan provider ONNX Runtime
│   └── diagnose_runtime.py # pemeriksaan runtime
├── face_assets/
├── models/
├── requirements/
├── scripts/                # setup.ps1 / setup.sh
├── docs/
└── tools/
```

## Instalasi

Dibutuhkan Python 3.11. Gunakan skrip setup yang menginstall `uv`, membuat
`.venv`, dan menyiapkan dependensi sesuai mesin.

Windows PowerShell:

```powershell
./scripts/setup.ps1            # default: GPU NVIDIA
./scripts/setup.ps1 -CPU       # ONNX Runtime CPU
./scripts/setup.ps1 -DML       # ONNX Runtime DirectML
```

Linux/macOS:

```bash
./scripts/setup.sh             # default: NVIDIA GPU
./scripts/setup.sh -CPU
./scripts/setup.sh -DML
```

Tanpa skrip, aktifkan `.venv` dan pakai `uv pip install -r requirements/...`
sesuai pilihan GPU/CPU/DML.

## Menjalankan

```bash
python main.py                 # menu interaktif: Tello / video / webcam / diagnosa
python main.py tello            # langsung mode Tello
python main.py video            # pilih video dari folder videos/
python main.py webcam
python main.py diagnose         # cek CUDA/DML/CPU + aset wajah
```

Mode Tello — tombol:

```
SPACE=naik/turun   WASD=gerak    T=takeoff    L=land
Q=foto   E=rekam   C/X=kecepatan   []=trim   F=emergency   TAB=reset trim
```

Tello: takeoff via tombol `T`. Saat dijalankan langsung lewat
`src/full_pipeline.py`, mode Tello butuh flag `--allow-takeoff` untuk
mengizinkan takeoff; tanpa flag, hanya stream + kontrol tanpa terbang.

## Model yang dibutuhkan

Letakkan file di `models/`:

```
models/
├── yolov8s_512_fp32.onnx          # detector
├── yolo26s-pose_512_fp32.onnx     # pose
├── har_window_30_representative.onnx
├── feature_mean.npy
├── feature_std.npy
└── pipeline_metadata.json
```

Lihat [models/README.md](models/README.md) untuk aturan Git LFS dan validasi
bentuk model. Statistik normalisasi wajib berasal dari training model HAR yang
sama.

## Diagnosa runtime

```bash
python main.py diagnose
```

Menampilkan versi OpenCV, ONNX Runtime + provider (CUDA/DML/CPU), PyAV, dan
status aset wajah (model + jumlah identitas). YOLO deteksi/pose dan HAR semuanya
berjalan lewat ONNX Runtime (tanpa PyTorch/Ultralytics).

## Pengujian langsung

```bash
python src/full_pipeline.py --source video --video path/to/video.mp4 --profile laptop
python src/full_pipeline.py --source webcam --camera 0 --profile laptop
python src/full_pipeline.py --source tello --profile nano --enable-face
```

## Rekaman video dan log CSV

- **Codec H.264** (PyAV/libx264, yuv420p + faststart) — kompatibel dengan
  pemutar default Windows/Android/iPhone. OpenCV `mp4v` lama ditinggalkan
  karena sering tidak bisa diputar.
- Mode **Tello**: tekan `E` untuk mulai/berhenti rekaman. Video disimpan di
  `captures/videos/tello_*.mp4` (frame + HUD). Rekaman yang masih aktif
  difinalisasi otomatis saat program ditutup.
- Mode **video/webcam**: video output ditulis ke `--output`.
- Setiap eksekusi membuat folder `output/run_<timestamp>/` berisi:
  - `detections.csv` — 1 baris per deteksi per frame: `frame_index,
    t_run_s, track_id, x1, y1, x2, y2, det_conf, activity, activity_score,
    pose_valid, face_identity, face_similarity, liveness, liveness_score`
  - `frames.csv` — agregat per frame: `frame_index, t_run_s, n_people,
    dominant_activity, fps_ema, ms_detector, ms_bytetrack, ms_pose,
    ms_body110_har, ms_face_liveness, ms_total`
  - `benchmark.json` (video/webcam) atau `session_benchmark.json` (Tello)

## Privasi dan keselamatan

- Database embedding wajah tidak disimpan di repository publik.
- Jangan commit foto wajah, embedding personal, token, atau credential Kaggle.
- Mode Tello tidak melakukan takeoff otomatis tanpa perintah pengguna tombol
  dan flag `--allow-takeoff`.
- Pengujian terbang dilakukan di area aman dengan baterai cukup dan propeller
  guard.

## Status

- Pipeline ONNX penuh: tersedia.
- Multi-person detection dan ByteTrack: tersedia.
- Body110 + CNN-BiLSTM: tersedia.
- Kontrol DJI Tello (keyboard + gamepad, HUD, foto/rekaman): tersedia.
- Face recognition dan liveness: tersedia sebagai modul opsional.
- Model binary: disediakan terpisah melalui Kaggle/Git LFS karena ukuran dan
  lisensinya.