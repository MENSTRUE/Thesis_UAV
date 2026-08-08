"""Re-ID ByteTrack -> subject_id berbasis embedding wajah.

Layer kedua di atas ByteTrack: ByteTrack memberi id kontinu per gerakan IoU,
layer ini memetakan byte_track_id -> subject_id yang stabil walau orang
keluar-masuk frame.

- Match aktif: cosine(embedding) > reid_cos ATAU IoU > reid_iou (greedy).
- Track tanpa match > max_missed frame -> pindah ke pool retired (embedding
  disimpan).
- Muncul lagi dengan cosine > retired_cos -> re-ID ke subject lama, bukan
  subject baru (model tau ini sebenarnya id yang sama).

Embedding wajah hanya tersedia bila FaceSystem aktif; saat tidak ada face
pada sebuah byte track, subject_id = byte_track_id (fallback ByteTrack).
State temporal HAR/pose TIDAK dikelola di sini, pipeline yang mengatur.
"""
from collections import deque

import numpy as np

try:
    from face_system import FaceSystem, normalize_embedding
except ImportError:
    from src.face_system import FaceSystem, normalize_embedding


def _iou(a, b):
    """IoU dua bbox (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-8)


class TrackReId:
    """Map byte_track_id -> subject_id stabil via embedding wajah.

    Misal: ByteTrack meng-assign id baru saat orang keluar-masuk frame. Pool
    retired menyimpan embedding; begitu embedding yang sama muncul lagi,
    subject_id lama dikembalikan sehingga identitas/display/reporting konsisten.
    """

    def __init__(self, cos_thresh=0.25, iou_thresh=0.35, retired_cos=0.30,
                 max_missed=30, max_retired=200):
        self.cos_thresh = float(cos_thresh)
        self.iou_thresh = float(iou_thresh)
        self.retired_cos = float(retired_cos)
        self.max_missed = int(max_missed)
        self.max_retired = int(max_retired)
        self.next_id = 1
        self.by_byte = {}    # byte_track_id -> {subject_id, emb, bbox, missed}
        self.retired = []    # list [subject_id, emb]
        self.n_known = 0

    def _score(self, emb, t_emb, bbox, t_bbox):
        c = float(np.dot(emb, t_emb))
        i = _iou(bbox, t_bbox)
        if c > self.cos_thresh or i > self.iou_thresh:
            return max(c, i)
        return 0.0

    def update(self, face_map, bbox_map):
        """face_map: {keyword byte_id: {embedding, face_box, identity, ...}}
        bbox_map:  {byte_id: bbox} untuk track tanpa wajah (fallback IoU).

        Return: {byte_id: subject_id}. byte_id tanpa face -> subject_id passthrough.
        """
        result = {}
        assigned = set()

        # 1) Track dengan wajah: match ke subject aktif / re-ID retired / baru.
        for byte_id, face in face_map.items():
            emb = normalize_embedding(np.asarray(face["embedding"], dtype=np.float32))
            bbox = face.get("face_box")
            if bbox is None:
                bbox = bbox_map.get(byte_id)
            if bbox is None:
                continue

            best_id, best_score = None, 0.0
            for bid, t in self.by_byte.items():
                if t["subject_id"] in assigned:
                    continue
                s = self._score(emb, t["emb"], bbox, t["bbox"])
                if s > best_score:
                    best_score, best_id = s, t["subject_id"]
            if best_id is None:
                for sid, r_emb in self.retired:
                    if sid in assigned:
                        continue
                    if float(np.dot(emb, r_emb)) > self.retired_cos:
                        best_id = sid
                        break
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            assigned.add(best_id)
            self.by_byte[byte_id] = {
                "subject_id": best_id, "emb": emb, "bbox": bbox, "missed": 0,
            }
            result[byte_id] = best_id

        # 2) Byte track tanpa wajah: fallback ByteTrack (passthrough id baru).
        for byte_id, bbox in bbox_map.items():
            if byte_id in result:
                continue
            t = self.by_byte.get(byte_id)
            if t is None:
                t = {"subject_id": self.next_id, "emb": None, "bbox": bbox,
                     "missed": 0}
                self.next_id += 1
                self.by_byte[byte_id] = t
            t["bbox"] = bbox
            t["missed"] = 0
            result[byte_id] = t["subject_id"]

        # 3) Track hilang -> missed++, ret/retired setelah max_missed.
        for byte_id in list(self.by_byte):
            if byte_id in result:
                continue
            t = self.by_byte[byte_id]
            t["missed"] += 1
            if t["missed"] > self.max_missed:
                self.retired.append([t["subject_id"], t["emb"]])
                del self.by_byte[byte_id]

        # Batasi jumlah retired (cukup untuk eksperimen; lru bila penuh).
        if len(self.retired) > self.max_retired:
            self.retired = self.retired[-self.max_retired:]
        self.n_known = self.next_id - 1
        return result

    def reset(self):
        self.next_id = 1
        self.by_byte.clear()
        self.retired.clear()
        self.n_known = 0


class ReIdFaceSystem(FaceSystem):
    """FaceSystem + embedding di output `process()` utk TrackReId.

    Adapter di atas FaceSystem (file asli tidak diubah): menambahkan kunci
    `embedding` pada dict hasil `process()` sehingga wajah yang sama bisa di-re-id
    walau track ByteTrack berubah.
    """

    def process(self, frame, track_boxes):
        output = {}
        for face in self.app.get(frame):
            track_id = self.match_track(face.bbox, track_boxes)
            if track_id is None:
                continue
            identity, similarity = self.recognize(face.embedding)
            history = self._liveness_history.setdefault(track_id, deque(maxlen=30))
            live_score = self.anti_spoof.predict_real_score(frame, face.bbox, history)
            output[track_id] = {
                "identity": identity,
                "similarity": similarity,
                "liveness": "real" if live_score >= self.liveness_threshold else "spoof",
                "liveness_score": live_score,
                "face_box": np.asarray(face.bbox, dtype=np.float32),
                "embedding": np.asarray(face.embedding, dtype=np.float32),
            }
        return output


if __name__ == "__main__":
    e1 = np.array([1.0, 0.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0, 0.0])
    b1, b2 = np.array([0, 0, 50, 50]), np.array([100, 0, 150, 50])
    tr = TrackReId()
    ids0 = tr.update(
        {1: {"embedding": e1, "face_box": b1}, 2: {"embedding": e2, "face_box": b2}},
        {},
    )
    assert list(ids0.values()) == [1, 2], f"id awal: {ids0}"
    ids1 = tr.update(
        {1: {"embedding": e1, "face_box": (2, 2, 52, 52)}},
        {},
    )
    assert ids1[1] == 1, "IoU match harus menjaga id"
    for _ in range(tr.max_missed + 2):
        tr.update({}, {})
    assert not tr.by_byte and len(tr.retired) == 2, "semua track harus retired"
    ids3 = tr.update({1: {"embedding": e1, "face_box": (5, 5, 55, 55)}}, {})
    assert ids3 == {1: 1}, f"re-ID ke id lama ({ids3})"
    assert tr.n_known == 2
    print("[OK] track_reid self-check passed")