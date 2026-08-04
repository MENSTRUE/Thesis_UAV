"""Diagnosa takeoff Tello — TANPA rc, TANPA stream.

Connect -> diam mutlak 2 detik -> takeoff -> land. Log semua datagram UDP 8889.

Cara pakai:
    uv run python scripts/tello_probe.py

Hasil TAKEOFF OK  -> masalah di pipeline kita (rc flood), lanjut hardening.
Hasil FAILED      -> masalah drone/firmware, bukan kode.
"""

import logging
import time

from djitellopy import Tello

Tello.LOGGER.setLevel(logging.DEBUG)

tello = Tello(retry_count=1)
ok = False
try:
    tello.connect()
    print("[PROBE] SDK mode OK (command -> response diterima)")

    state = tello.get_own_udp_object()
    print(f"[PROBE] respons queue awal: {len(state['responses'])}")

    print("[PROBE] diam mutlak 2 detik (tanpa rc, tanpa stream)...")
    time.sleep(2.0)

    print("[PROBE] kirim takeoff (timeout 20s)...")
    tello.takeoff()
    print("[PROBE] TAKEOFF OK")
    ok = True
    time.sleep(1.0)

    print("[PROBE] kirim land...")
    tello.land()
    print("[PROBE] LAND OK")
finally:
    time.sleep(0.5)
    state = tello.get_own_udp_object()
    print(f"[PROBE] antrean respons akhir: {len(state['responses'])}, "
          f"isi: {state['responses']}")
    try:
        tello.end()
    except Exception as exc:
        print(f"[PROBE] end() gagal: {exc}")