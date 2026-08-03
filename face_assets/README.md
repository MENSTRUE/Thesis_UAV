# Optional face assets

Struktur lokal yang dibutuhkan ketika memakai `--enable-face`:

```text
face_assets/
├── database/
│   └── embeddings/
│       ├── person_1/emb_0001.npy
│       └── person_2/emb_0001.npy
└── models/
    └── MiniFASNetV2.onnx
```

Folder `database/` sengaja diabaikan oleh `.gitignore` karena embedding wajah
merupakan data biometrik. Simpan database tersebut hanya pada perangkat yang
berwenang dan jangan push ke repository publik.

`MiniFASNetV2.onnx` dapat ditempatkan secara lokal pada
`face_assets/models/`. Pastikan hak distribusi model terpenuhi sebelum
memasukkannya ke Git LFS.

