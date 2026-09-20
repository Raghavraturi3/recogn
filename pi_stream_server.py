"""
Run this ON the Raspberry Pi 4B. It streams the Pi's camera as MJPEG over
HTTP so your PC's live_detect.py can pull it in exactly like a local webcam
- just point VIDEO_SOURCE at this Pi's stream URL instead of 0.

SETUP ON THE PI
---------------
    sudo apt update
    sudo apt install python3-opencv python3-flask -y
    python3 pi_stream_server.py

Find the Pi's IP with `hostname -I`. Then on the PC:
    VIDEO_SOURCE = "http://<pi-ip>:8000/stream.mjpg"

CAMERA COMPATIBILITY NOTE
--------------------------
- USB webcam: cv2.VideoCapture(0) below just works.
- Official Pi Camera Module (CSI ribbon cable): on Raspberry Pi OS Bookworm
  (libcamera stack), plain cv2.VideoCapture(0) usually will NOT see it. Two
  fixes:
    (a) Enable the legacy camera stack: `sudo raspi-config` -> Interface
        Options -> Legacy Camera -> Enable, then reboot. cv2.VideoCapture(0)
        will work as-is after that.
    (b) Or use Picamera2 to grab frames instead - see the commented
        alternate capture loop near the bottom of this file.

TUNING FOR LATENCY
-------------------
- WIDTH/HEIGHT and JPEG_QUALITY are the two biggest latency knobs. Smaller
  and more compressed = less network time = less delay.
- Use Ethernet if at all possible; WiFi (especially 2.4GHz) adds jitter.
"""

import time
import threading

import cv2
from flask import Flask, Response

# ---- CONFIG -----------------------------------------------------------

CAMERA_INDEX = 0        # 0 = default camera
WIDTH = 640
HEIGHT = 480
JPEG_QUALITY = 80        # 0-100; lower = faster/smaller, more artifacts
PORT = 8000

# ---- CAPTURE ------------------------------------------------------------

app = Flask(__name__)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open camera on the Pi. If you're using the official "
        "Camera Module on Bookworm, see the CAMERA COMPATIBILITY NOTE above."
    )

frame_lock = threading.Lock()
latest_frame = None
running = True


def capture_loop():
    """Same idea as your PC-side ThreadedCamera: keep only the newest frame
    so the stream never falls behind."""
    global latest_frame
    while running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame
        else:
            time.sleep(0.01)


threading.Thread(target=capture_loop, daemon=True).start()

# ---- ALTERNATE CAPTURE LOOP FOR PICAMERA2 (CSI Camera Module, Bookworm) --
# If cv2.VideoCapture(0) doesn't find your Camera Module, comment out the
# cap = cv2.VideoCapture(...) block and capture_loop() above, and use this
# instead (pip install picamera2 first, or it's preinstalled on Raspberry
# Pi OS):
#
# from picamera2 import Picamera2
# picam2 = Picamera2()
# picam2.configure(picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT)}))
# picam2.start()
#
# def capture_loop():
#     global latest_frame
#     while running:
#         frame = picam2.capture_array()          # returns RGB
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         with frame_lock:
#             latest_frame = frame

# ---- STREAMING ------------------------------------------------------------

def generate():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue
        ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


@app.route("/stream.mjpg")
def stream():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    return '<html><body><img src="/stream.mjpg"></body></html>'


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    finally:
        running = False
        cap.release()