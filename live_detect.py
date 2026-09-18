import sys
import os

# ------------------------------------------------------------------
# Windows + pip-installed CUDA/cuDNN fix: onnxruntime does not reliably
# auto-discover the DLLs shipped inside nvidia-cublas-cu12 / nvidia-cudnn-cu12
# etc. unless their folders are explicitly added to the DLL search path
# BEFORE onnxruntime is imported. This finds every "nvidia/<pkg>/bin" folder
# inside the current environment's site-packages and registers it.
# ------------------------------------------------------------------
if sys.platform.startswith("win"):
    import sysconfig
    site_packages = sysconfig.get_paths()["purelib"]
    nvidia_root = os.path.join(site_packages, "nvidia")
    if os.path.isdir(nvidia_root):
        added = []
        for pkg_name in os.listdir(nvidia_root):
            bin_dir = os.path.join(nvidia_root, pkg_name, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                added.append(bin_dir)
        print(f"Added {len(added)} CUDA/cuDNN DLL directories to search path.")
        if not added:
            print("WARNING: found nvidia/ folder but no bin/ subfolders inside it — "
                  "check that nvidia-cublas-cu12 / nvidia-cudnn-cu12 actually installed.")
    else:
        print("WARNING: no 'nvidia' folder found in site-packages — "
              "nvidia-cublas-cu12/nvidia-cudnn-cu12 may not be installed in this environment.")

import cv2
import time
import threading
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis

KNOWN_FACES_DIR = "known_faces"
VIDEO_SOURCE = 0

SIMILARITY_THRESHOLD = 0.40

# 320x320 is a big cut in detector cost vs 640x640. Only raise this if you
# need far-face range and can afford the extra ms.
DET_SIZE = (320, 320)

MODEL_NAME = "buffalo_s"

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

PROCESS_EVERY_N_FRAMES = 2

# Where TensorRT caches its compiled engines. First run after any change to
# DET_SIZE, MODEL_NAME, or this cache path will take 30-90s to build the
# engine — that is NOT a hang, let it finish. Every run after that reuses
# the cached engine and starts fast.
TRT_CACHE_DIR = "trt_cache"
os.makedirs(TRT_CACHE_DIR, exist_ok=True)


class ThreadedCamera:
    def __init__(self, source, width=None, height=None):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def check_gpu():
    available = onnxruntime.get_available_providers()
    print("ONNX Runtime available providers:", available)
    if "CUDAExecutionProvider" not in available:
        print("!! CUDAExecutionProvider NOT available — running on CPU. !!")
    if "TensorrtExecutionProvider" not in available:
        print("!! TensorrtExecutionProvider not available — falling back to CUDA only. !!")
    return available


def build_app():
    available = check_gpu()

    providers = []
    if "TensorrtExecutionProvider" in available:
        providers.append(("TensorrtExecutionProvider", {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": TRT_CACHE_DIR,
            "trt_fp16_enable": True,
        }))
    providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    # Only load the detection + recognition models — skip landmark_3d_68,
    # landmark_2d_106, and genderage since nothing in this script uses them.
    app = FaceAnalysis(name=MODEL_NAME, providers=providers,
                        allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    return app


def build_known_encodings(app):
    known_encodings = []
    known_names = []

    if not os.path.isdir(KNOWN_FACES_DIR):
        raise FileNotFoundError("known_faces folder not found. Run capture_reference.py first.")

    for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                print(f"Warning: no face found in {img_path}, skipping.")
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            known_encodings.append(face.normed_embedding)
            known_names.append(person_name)

    if not known_encodings:
        raise ValueError("No usable faces found in known_faces/. Add clearer photos.")

    return np.array(known_encodings), known_names


def match_faces(faces, known_encodings, known_names):
    results = []
    for face in faces:
        emb = face.normed_embedding
        sims = known_encodings @ emb
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        x1, y1, x2, y2 = face.bbox.astype(int)
        if best_sim > SIMILARITY_THRESHOLD:
            name, color = known_names[best_idx], (0, 255, 0)
        else:
            name, color = "Unknown", (0, 0, 255)
        results.append(((x1, y1, x2, y2), f"{name} ({best_sim:.2f})", color))
    return results


def draw_results(frame, results):
    for (x1, y1, x2, y2), text, color in results:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    app = build_app()
    known_encodings, known_names = build_known_encodings(app)
    print(f"Loaded {len(known_encodings)} reference photos across "
          f"{len(set(known_names))} people: {sorted(set(known_names))}")

    cam = ThreadedCamera(VIDEO_SOURCE, CAPTURE_WIDTH, CAPTURE_HEIGHT)

    frame_count = 0
    last_results = []
    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0
    detect_ms = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                t0 = time.time()
                faces = app.get(frame)
                detect_ms = (time.time() - t0) * 1000
                last_results = match_faces(faces, known_encodings, known_names)

            draw_results(frame, last_results)

            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer = time.time()

            cv2.putText(frame, f"FPS: {fps_display}  |  detect: {detect_ms:.0f}ms",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Live Person Recognition (GPU)", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()