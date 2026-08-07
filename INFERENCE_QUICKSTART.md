# Inference Quick Start

Panduan ini digunakan untuk menjalankan pipeline inference tanpa membuka
notebook training.

## 1. File kode yang perlu disimpan di repository

File minimum untuk inference lokal:

```text
src/
├── body110.py
├── diagnose_runtime.py
├── export_yolo_tensorrt.py
├── full_pipeline.py
├── tello_control.py
├── face_system.py
└── runtime_utils.py

requirements/
├── requirements-full.txt
├── requirements-har.txt
├── requirements-gpu.txt
└── requirements-dml.txt

models/
├── README.md
├── yolov8s_512_fp32.onnx
├── yolo26s-pose_512_fp32.onnx
├── har_window_30_representative.onnx
├── feature_mean.npy
├── feature_std.npy
└── pipeline_metadata.json
```

`pipeline_metadata.json` memuat array `class_names`; `class_mapping.json`
dengan bentuk lain-nya juga dikenali.

File notebook yang sebaiknya ikut disimpan:

```text
notebooks/FULL_ONNX_HAR_UAV_Kaggle.ipynb
notebooks/YOLO_Detector_Pose_Export_ONNX_Kaggle.ipynb
```

## 2. Model untuk notebook Kaggle

Notebook ONNX menggunakan file berikut:

```text
yolov8s_512_fp32.onnx
yolo26s-pose_512_fp32.onnx
har_window_30_representative.onnx
feature_mean.npy
feature_std.npy
pipeline_metadata.json
```

Untuk modul wajah opsional:

```text
face_assets/models/MiniFASNetV2.onnx
```

Database embedding wajah disimpan lokal dengan struktur cent-atroid per
identitas:

```text
face_assets/database/embeddings/<nama>/anchor_centroid.npy
```

Jangan push database embedding atau foto wajah pribadi ke repository publik.

## 3. Instalasi

Gunakan Python 3.11.

Windows PowerShell:

```powershell
./scripts/setup.ps1          # GPU NVIDIA (default)
./scripts/setup.ps1 -CPU
./scripts/setup.ps1 -DML
```

Linux:

```bash
./scripts/setup.sh           # GPU NVIDIA (default)
./scripts/setup.sh -CPU
./scripts/setup.sh -DML
```

Alternatif manual (uv):

```bash
uv venv --python 3.11 .venv
uv pip install -p .venv -r requirements/requirements-full.txt -r requirements/requirements-gpu.txt
```

## 4. Uji environment

```bash
python main.py diagnose
```

## 5. Inference video

Jika memakai `pipeline_metadata.json`:

```bash
python src/full_pipeline.py \
  --source video \
  --video "videos/uji.mp4" \
  --profile quality \
  --mapping models/pipeline_metadata.json
```

Di Windows PowerShell, perintah satu barisnya:

```powershell
python src/full_pipeline.py --source video --video "videos/uji.mp4" --profile quality --mapping models/pipeline_metadata.json
```

Hasil video dan laporan performa disimpan secara default ke folder baru
`output/run_<timestamp>/` (`full_pipeline.mp4` + `benchmark.json`), ditambah
log CSV `detections.csv` dan `frames.csv`. Setiap eksekusi membuat folder run
baru agar hasil lama tidak tertimpa.

Cara paling sederhana — menu interaktif (pilih video dari `videos/`):

```bash
python main.py
```

## 6. Inference webcam

```bash
python src/full_pipeline.py --source webcam --camera 0 --profile laptop --mapping models/pipeline_metadata.json
```

## 7. Inference dengan wajah dan liveness

```bash
python src/full_pipeline.py \
  --source video \
  --video "videos/uji.mp4" \
  --profile laptop \
  --mapping models/pipeline_metadata.json \
  --enable-face \
  --face-assets face_assets
```

Kotak wajah pada pipeline menggunakan cache berdasarkan `Track ID`, sehingga
hasil deteksi wajah dari frame sebelumnya dapat dipertahankan ketika detektor
wajah sesaat gagal mendeteksi.

## 8. Inference DJI Tello tanpa takeoff otomatis

```bash
python main.py tello
```

Opsi `--allow-takeoff` hanya digunakan ketika memang ingin mengizinkan perintah
takeoff dari aplikasi dan pengujian dilakukan di area yang aman. Tanpa flag
tersebut, Tello hanya menerima stream dan kontrol tanpa terbang.

## 9. Benchmark tanpa tampilan dan tanpa menyimpan video

```bash
python src/full_pipeline.py \
  --source video \
  --video "videos/uji.mp4" \
  --profile laptop \
  --mapping models/pipeline_metadata.json \
  --no-display \
  --no-save-video \
  --report output/benchmark.json
```

## 10. Push model besar dengan Git LFS

```bash
git lfs install
git add .gitattributes models/
git commit -m "Add deployment models"
git push
```

Pastikan kuota Git LFS mencukupi. File `.engine` sebaiknya tidak dibagikan
lintas perangkat karena kompatibilitasnya bergantung pada GPU, TensorRT, dan
lingkungan target.
