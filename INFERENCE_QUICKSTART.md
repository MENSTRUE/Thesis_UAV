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
└── offboard_tello_har.py

requirements/
├── requirements-full.txt
└── requirements-har.txt

models/
├── README.md
├── yolov8s.pt
├── yolo26s-pose.pt
├── har_window_30_representative.onnx
├── feature_mean.npy
├── feature_std.npy
└── pipeline_metadata.json
```

`pipeline_metadata.json` harus memuat `class_mapping`. Jika nama file yang
tersedia adalah `class_mapping.json`, file tersebut dapat digunakan langsung.

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

Database embedding wajah disimpan lokal dengan struktur:

```text
face_assets/database/embeddings/<nama>/emb_0001.npy
```

Jangan push database embedding atau foto wajah pribadi ke repository publik.

## 3. Instalasi

Gunakan Python 3.11.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements/requirements-full.txt
```

Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements/requirements-full.txt
```

## 4. Uji environment

```bash
python src/diagnose_runtime.py
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

Hasil video disimpan secara default ke `output/full_pipeline.mp4`, sedangkan
laporan performa disimpan ke `output/benchmark.json`.

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
python src/full_pipeline.py --source tello --profile nano --mapping models/pipeline_metadata.json
```

Opsi `--allow-takeoff` hanya digunakan ketika memang ingin mengizinkan perintah
takeoff dari aplikasi dan pengujian dilakukan di area yang aman.

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
