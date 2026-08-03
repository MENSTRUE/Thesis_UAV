# Model files

Folder ini disiapkan untuk model deployment. Binary model belum terdapat dalam
commit awal karena file tidak tersedia pada workspace dan file besar perlu
disimpan melalui Git LFS.

## Model HAR

Salin dari hasil training `model_seed5/models/`:

```text
har_window_30_representative.onnx
feature_mean.npy
feature_std.npy
pipeline_metadata.json
```

Validasi yang wajib:

- input HAR: `batch x 30 x 110`;
- input mask: `batch x 30`;
- `feature_mean.npy`: shape `(110,)`;
- `feature_std.npy`: shape `(110,)`;
- urutan kelas mengikuti metadata model yang sama.

## Model YOLO

Untuk notebook Kaggle:

```text
yolov8s_512_fp32.onnx
yolo26s-pose_512_fp32.onnx
```

Untuk aplikasi lokal atau Jetson:

```text
yolov8s.pt
yolo26s-pose.pt
```

File `.engine` TensorRT harus diekspor pada perangkat target menggunakan
`src/export_yolo_tensorrt.py`. Engine dari GPU atau versi TensorRT lain belum
tentu kompatibel.

## Git LFS

```bash
git lfs install
git add models/
git commit -m "Add deployment models"
git push
```

Periksa batas kuota Git LFS akun GitHub sebelum mengunggah file besar.

