"""Kontrol manual DJI Tello terintegrasi: driver video PyAV low-latency,
input keyboard + XInput gamepad, HUD telemetry, foto, dan rekaman.

Diadaptasi dari proyek dji-tello (drone.py, input_handler.py,
video_handler.py, config.py). Gamepad hanya aktif di Windows (ctypes),
keyboard GetAsyncKeyState hanya di Windows; selain itu posisi analog
dibaca dari tombol cv2 (WASD/panah) yang dikirim via poll(cv2_key).
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as w
import json
import logging
import math
import os
import platform
import sys
import time
from datetime import datetime
from threading import Lock, Thread
from typing import Tuple

import cv2
from fractions import Fraction


# ---------------------------------------------------------------------------
# Konfigurasi (dari dji-tello/config.py)
# ---------------------------------------------------------------------------

PHOTO_DIR = "captures/photos"
VIDEO_DIR = "captures/videos"
TRIM_STEP = 3
TRIM_MAX = 30
DEADZONE = 0.15
SPEED_MODES = (30, 50, 70, 100)
HOLD_DELAY = 0.8
BATTERY_WARN = 20
BATTERY_CRITICAL = 10
CONFIG_FILE = "config.json"


def load_config() -> Tuple[int, int]:
    try:
        with open(CONFIG_FILE) as f:
            d = json.load(f)
        return int(d.get("trim_lr", 0)), int(d.get("speed_idx", 3))
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return 0, 3


def save_config(trim_lr: int, speed_idx: int) -> None:
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"trim_lr": trim_lr, "speed_idx": speed_idx}, f)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Video decoder low-latency (PyAV) — dari dji-tello/drone.py
# ---------------------------------------------------------------------------

class VideoDecoder:
    def __init__(self, port=11111):
        self._frame = None
        self._lock = Lock()
        self._running = False
        self._thread = None
        self._container = None
        self._address = f"udp://@0.0.0.0:{port}"

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = Thread(target=self._decode, daemon=True)
        self._thread.start()

    def _decode(self):
        try:
            import av
            self._container = av.open(self._address, timeout=(5, None))
            for frame in self._container.decode(video=0):
                if not self._running:
                    break
                with self._lock:
                    self._frame = frame.to_ndarray(format="bgr24")
        except Exception:
            pass
        finally:
            if self._container:
                try:
                    self._container.close()
                except Exception:
                    pass

    @property
    def frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Input keyboard + gamepad XInput — dari dji-tello/input_handler.py
# ---------------------------------------------------------------------------

_WIN = sys.platform == "win32"

if _WIN:
    user32 = ctypes.windll.user32
    _VK = {"W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44,
           "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27}
    _BRACKET_L, _BRACKET_R = 0xDB, 0xDD
else:
    user32 = None
    _VK = {}


def _load_xinput():
    if not _WIN:
        return None
    for name in ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"):
        try:
            return ctypes.windll.LoadLibrary(name)
        except OSError:
            pass
    return None


def held(vk: int) -> bool:
    if not _WIN:
        return False
    return ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000 != 0


class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", w.WORD),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class XINPUT_STATE(ctypes.Structure):
    _fields_ = [("dwPacketNumber", w.DWORD), ("Gamepad", XINPUT_GAMEPAD)]


class XINPUT_VIBRATION(ctypes.Structure):
    _fields_ = [("wLeftMotorSpeed", w.WORD), ("wRightMotorSpeed", w.WORD)]


_BTN = {
    "A": 0x1000, "B": 0x2000, "BACK": 0x0020, "START": 0x0010,
    "DPAD_LEFT": 0x0004, "DPAD_RIGHT": 0x0008, "LB": 0x0100, "RB": 0x0200,
}


class InputState:
    def __init__(self):
        self.lr = 0.0
        self.fb = 0.0
        self.ud = 0.0
        self.yaw = 0.0
        self.takeoff_land = False
        self.emergency_land = False
        self.photo = False
        self.record_toggle = False
        self.trim_left = False
        self.trim_right = False
        self.trim_reset = False
        self.speed_up = False
        self.speed_down = False
        self.switch_mode = False
        self.quit = False


class InputHandler:
    def __init__(self, deadzone=DEADZONE):
        self.mode = "keyboard"
        self.deadzone = deadzone
        self._xinput = _load_xinput()
        self._connected = False
        self._prev_w = 0
        self._p = {}
        self._start_hold = 0.0
        self._vibe_until = 0.0

    def has_gamepad(self) -> bool:
        if not self._xinput:
            return False
        state = XINPUT_STATE()
        if self._xinput.XInputGetState(0, ctypes.byref(state)) == 0:
            self._connected = True
            return True
        self._connected = False
        return False

    def vibrate(self, left=0, right=0, duration=0.0):
        if not self._xinput:
            return
        v = XINPUT_VIBRATION(left, right)
        self._xinput.XInputSetState(0, ctypes.byref(v))
        if duration:
            self._vibe_until = time.time() + duration

    def switch_mode(self):
        self.mode = "gamepad" if self.mode == "keyboard" else "keyboard"

    def _poll_gamepad(self, st):
        if not self._xinput:
            return
        state = XINPUT_STATE()
        if self._xinput.XInputGetState(0, ctypes.byref(state)) != 0:
            self._connected = False
            return
        self._connected = True
        g = state.Gamepad
        d = self.deadzone

        st.lr = (g.sThumbRX / 32768.0) if abs(g.sThumbRX) > 32768 * d else 0.0
        st.fb = -(g.sThumbRY / 32768.0) if abs(g.sThumbRY) > 32768 * d else 0.0
        st.ud = -(g.sThumbLY / 32768.0) if abs(g.sThumbLY) > 32768 * d else 0.0
        st.yaw = -(g.sThumbLX / 32768.0) if abs(g.sThumbLX) > 32768 * d else 0.0

        cur = g.wButtons
        edges = cur & ~self._prev_w
        now = time.time()
        start_pressed = cur & _BTN["START"]

        if start_pressed and not (self._prev_w & _BTN["START"]):
            self._start_hold = now
        elif start_pressed and (self._prev_w & _BTN["START"]):
            if now - self._start_hold >= HOLD_DELAY:
                st.emergency_land = True
        elif not start_pressed and (self._prev_w & _BTN["START"]):
            if now - self._start_hold < HOLD_DELAY:
                st.takeoff_land = True
            self._start_hold = 0.0

        if edges & _BTN["A"]:
            st.photo = True
        if edges & _BTN["B"]:
            st.record_toggle = True
        if edges & _BTN["BACK"]:
            st.trim_reset = True
        if edges & _BTN["LB"]:
            st.speed_down = True
        if edges & _BTN["RB"]:
            st.speed_up = True
        if cur & _BTN["DPAD_LEFT"]:
            st.trim_left = True
        if cur & _BTN["DPAD_RIGHT"]:
            st.trim_right = True

        self._prev_w = cur
        if self._vibe_until and now > self._vibe_until:
            self._xinput.XInputSetState(0, ctypes.byref(XINPUT_VIBRATION(0, 0)))
            self._vibe_until = 0.0

    def poll(self, cv2_key=-1):
        st = InputState()

        if cv2_key == 27:
            st.quit = True
        if cv2_key == ord("r"):
            st.switch_mode = True
        if cv2_key == 9:
            st.trim_reset = True
        sp = cv2_key == 32
        if sp and not self._p.get("tk"):
            st.takeoff_land = True
        self._p["tk"] = sp

        if self.mode == "keyboard":
            st.fb = float(held(_VK["W"])) - float(held(_VK["S"]))
            st.lr = float(held(_VK["D"])) - float(held(_VK["A"]))
            st.ud = float(held(_VK["UP"])) - float(held(_VK["DOWN"]))
            st.yaw = float(held(_VK["RIGHT"])) - float(held(_VK["LEFT"]))

            if cv2_key == ord("q"):
                st.photo = True
            if cv2_key == ord("e"):
                st.record_toggle = True
            if cv2_key == ord("c"):
                st.speed_up = True
            if cv2_key == ord("x"):
                st.speed_down = True

            tl = held(_BRACKET_L) if _WIN else False
            tr = held(_BRACKET_R) if _WIN else False
            if tl and not self._p.get("tl"):
                st.trim_left = True
            if tr and not self._p.get("tr"):
                st.trim_right = True
            self._p["tl"], self._p["tr"] = tl, tr

        self._poll_gamepad(st)
        return st


# ---------------------------------------------------------------------------
# VideoHandler: HUD + foto + rekaman — dari dji-tello/video_handler.py
# ---------------------------------------------------------------------------

class H264Mp4Writer:
    """Writer MP4 H.264 via PyAV (libx264).

    Pengganti cv2.VideoWriter mp4v (MPEG-4 Part 2) yang tidak bisa diputar di
    banyak perangkat (Android/iPhone default player). H.264 + yuv420p adalah
    format paling kompatibel; faststart menempatkan moov di awal file sehingga
    tetap bisa dibuka meski proses dihentikan paksa.
    """

    def __init__(self, path: str, width: int, height: int, fps: float = 20.0):
        import av

        self._container = av.open(str(path), mode="w", options={"movflags": "faststart"})
        self._stream = self._container.add_stream("h264", rate=Fraction(fps))
        self._stream.width, self._stream.height = int(width), int(height)
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "23"}
        self._open = True

    @property
    def is_open(self):
        return self._open

    def write(self, frame):
        if not self._open or frame is None:
            return
        import av

        vframe = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in self._stream.encode(vframe):
            self._container.mux(packet)

    def close(self):
        if not self._open:
            return
        import av

        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._open = False


class VideoHandler:
    def __init__(self):
        os.makedirs(PHOTO_DIR, exist_ok=True)
        os.makedirs(VIDEO_DIR, exist_ok=True)
        self._recording = False
        self._rec_start = 0
        self._writer = None
        self.photo_count = 0

    @property
    def recording(self):
        return self._recording

    def draw_drone_state(self, frame, lr, fb, ud, yaw):
        h, w = frame.shape[:2]
        cx, cy = 100, h - 100
        box = frame[cy - 40:cy + 40, cx - 40:cx + 40].copy()
        cv2.rectangle(box, (0, 0), (80, 80), (0, 0, 0), -1)
        cv2.addWeighted(box, 0.35, frame[cy - 40:cy + 40, cx - 40:cx + 40], 0.65, 0, frame[cy - 40:cy + 40, cx - 40:cx + 40])

        cv2.circle(frame, (cx, cy), 8, (180, 180, 180), 2)
        cv2.line(frame, (cx - 6, cy - 6), (cx + 6, cy + 6), (140, 140, 140), 1)
        cv2.line(frame, (cx + 6, cy - 6), (cx - 6, cy + 6), (140, 140, 140), 1)

        def _draw(val, ox, oy, dx, dy):
            col = (0, 0, 255) if abs(val) > 0.05 else (50, 50, 50)
            mag = int(min(abs(val), 1.0) * 25)
            if mag < 2:
                return
            cv2.arrowedLine(frame, (ox, oy), (ox + dx * mag, oy + dy * mag), col, 2, tipLength=0.3)

        _draw(fb, cx, cy - 12, 0, -1)
        _draw(fb, cx, cy + 12, 0, 1)
        _draw(lr, cx - 12, cy, -1, 0)
        _draw(lr, cx + 12, cy, 1, 0)
        _draw(ud, cx + 25, cy - 8, 0, -1)
        _draw(ud, cx + 25, cy + 8, 0, 1)
        _draw(yaw, cx - 25, cy - 8, -1, 0)
        _draw(yaw, cx - 25, cy + 8, 1, 0)

    def render(self, frame, battery, flying, mode, rec, trim_lr, speed_pct=100,
               grid=False, height=0, flight_time=0, lr=0.0, fb=0.0, ud=0.0, yaw=0.0):
        h, w = frame.shape[:2]
        bar = frame[:52].copy()
        cv2.rectangle(bar, (0, 0), (w, 52), (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.55, frame[:52], 0.45, 0, frame[:52])

        cv2.putText(frame, mode.upper(), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bat_col = ((0, 0, 255) if battery <= BATTERY_CRITICAL
                   else ((0, 255, 255) if battery <= BATTERY_WARN else (255, 255, 255)))
        cv2.putText(frame, f"BAT {battery}%", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bat_col, 1)

        status = "FLY" if flying else "GRD"
        col = (0, 255, 0) if flying else (0, 0, 255)
        cv2.putText(frame, status, (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.putText(frame, f"PH {self.photo_count}", (160, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"TR {trim_lr:+d}", (320, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"SPD {speed_pct}%", (320, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"ALT {height / 100:.1f}m", (440, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"TM {flight_time // 60:02d}:{flight_time % 60:02d}", (560, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if rec:
            elapsed = int(time.time() - self._rec_start)
            cv2.circle(frame, (w - 40, 20), 6, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 75, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(frame, f"{elapsed // 60:02d}:{elapsed % 60:02d}", (w - 78, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if grid:
            for i in (1, 2):
                cv2.line(frame, (w * i // 3, 0), (w * i // 3, h), (180, 180, 180), 1)
                cv2.line(frame, (0, h * i // 3), (w, h * i // 3), (180, 180, 180), 1)

        self.draw_drone_state(frame, lr, fb, ud, yaw)

        if battery <= BATTERY_CRITICAL:
            red = frame.copy()
            cv2.rectangle(red, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(red, 0.2, frame, 0.8, 0, frame)
            text = "BATTERY CRITICAL"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(frame, text, ((w - size[0]) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif battery <= BATTERY_WARN:
            cv2.putText(frame, "LOW BATTERY", (w - 130, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame

    def capture_photo(self, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{PHOTO_DIR}/tello_{ts}.jpg", frame)
        self.photo_count += 1

    def toggle_recording(self, frame_shape):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording(frame_shape)

    def _start_recording(self, shape):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        h, w = shape[:2]
        self._writer = H264Mp4Writer(f"{VIDEO_DIR}/tello_{ts}.mp4", w, h, fps=20.0)
        self._recording = True
        self._rec_start = time.time()

    def _stop_recording(self):
        if self._writer:
            self._writer.close()
            self._writer = None
        self._recording = False

    def write_frame(self, frame):
        if self._writer:
            self._writer.write(frame)


# ---------------------------------------------------------------------------
# DroneControl — Tello + decoder + battery/flight — dari dji-tello/drone.py
# ---------------------------------------------------------------------------

class DroneControl:
    def __init__(self, allow_takeoff=True):
        from djitellopy import Tello
        # djitellopy set LOGGER.setLevel(INFO) saat di-import; redam ke WARNING
        # di sini (setelah import) agar rc/command spam tidak membanjiri console.
        Tello.LOGGER.setLevel(logging.WARNING)
        self.tello = Tello()
        self.allow_takeoff = allow_takeoff
        self._is_flying = False
        self._battery = 0
        self._bt = 0
        self._decoder = None
        # ponytail: throttle rc ke ~10Hz; channel UDP 8889 dipakai sama dengan
        # takeoff/land, banjir rc bikin respons takeoff hilang/timeout.
        self._rc_blocked = False
        self._last_rc = 0.0
        self._rc_min_interval = 0.1  # detik

    @property
    def is_flying(self):
        return self._is_flying

    def connect(self):
        self.tello.connect()
        print(f"Baterai Tello: {self.tello.get_battery()}%")
        self.tello.streamon()
        self._decoder = VideoDecoder(port=self.tello.vs_udp_port)
        self._decoder.start()
        time.sleep(1.0)

    def read(self):
        frame = self._decoder.frame if self._decoder else None
        if frame is None or frame.size == 0:
            return None
        return frame.copy()

    def takeoff(self):
        if not self.allow_takeoff:
            print("Takeoff diblokir; aktifkan dengan --allow-takeoff")
            return
        if self._is_flying:
            return
        # Berhenti kirim rc + bersihkan channel sebelum takeoff supaya
        # respons 'ok' tidak tertelan antrean rc di port 8889.
        self._rc_blocked = True
        self.tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        for attempt in (1, 2):
            try:
                self.tello.takeoff()
                self._is_flying = True
                break
            except Exception as exc:
                if attempt == 1:
                    print("[!] Takeoff gagal, retry sekali...")
                    time.sleep(0.5)
                else:
                    print(f"[!] Takeoff gagal: {exc}")
        self._rc_blocked = False

    def land(self):
        if not self._is_flying:
            return
        self._rc_blocked = True
        self.tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        try:
            self.tello.land()
            self._is_flying = False
        except Exception as exc:
            print(f"[!] Land gagal: {exc}")
        self._rc_blocked = False

    def toggle_flight(self):
        if self._is_flying:
            self.land()
        else:
            self.takeoff()

    def send_rc(self, lr, f_y, ud, yaw):
        # rc ground-flood mengwedge command channel Tello (takeoff tidak
        # direspons sampai power-cycle). Kirim rc HANYA saat terbang.
        if self._rc_blocked or not self._is_flying:
            return
        now = time.time()
        if now - self._last_rc < self._rc_min_interval:
            return
        self._last_rc = now
        self.tello.send_rc_control(int(lr), int(f_y), int(ud), int(yaw))

    def get_height(self):
        return self.tello.get_height()

    def get_flight_time(self):
        return self.tello.get_flight_time()

    def get_battery(self):
        now = time.time()
        if now - self._bt > 2:
            try:
                self._battery = self.tello.get_battery()
            except RuntimeError:
                pass
            self._bt = now
        return self._battery

    def close(self):
        if self._decoder:
            self._decoder.stop()
            self._decoder = None
        # Cleanup non-blocking: streamoff/land fire-and-forget lalu tutup
        # socket langsung. streamoff() blocking bisa hang ~28s di drone yang
        # mulai tidak respons dan meninggalkan sesi zombie (command run
        # berikutnya ditolak diam-diam).
        if self._is_flying:
            try:
                self.tello.send_command_without_return("land")
            except Exception:
                pass
            self._is_flying = False
        try:
            self.tello.send_command_without_return("streamoff")
        except Exception:
            pass
        from djitellopy import tello as _dt
        for sock in (getattr(_dt, "client_socket", None), getattr(_dt, "state_socket", None)):
            try:
                sock.close()
            except Exception:
                pass


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def rate_curve(x):
    return math.copysign(abs(x) ** 3, x)


def rc_from_state(st, speed_pct, trim_lr):
    """InputState -> (lr, pitch, ud, yaw) sesuai pipeline RC dji-tello."""
    spd = speed_pct / 100.0
    lr = clamp(int(rate_curve(st.lr) * 100 * spd) + trim_lr, -100, 100)
    fb = clamp(int(-rate_curve(st.fb) * 100 * spd), -100, 100)
    ud = clamp(int(-rate_curve(st.ud) * 100 * spd), -100, 100)
    yaw = clamp(int(-rate_curve(st.yaw) * 100 * spd), -100, 100)
    return lr, fb, ud, yaw