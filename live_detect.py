import sys
import os
import time
import threading

import cv2
import numpy as np
import onnxruntime
import pygame

from insightface.app import FaceAnalysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
SOUND_DIR = os.path.join(BASE_DIR, "sound")

VIDEO_SOURCE = 0

SIMILARITY_THRESHOLD = 0.40

# (256, 256), Lower computation, may miss smaller faces
# (320, 320), Your previous setting; good baseline
# (480, 480), Higher resolution, potentially better small-face detection
# (640, 640), Common higher-resolution setting
# (800, 800), Experimental; increased computation
# (1024, 1024), Experimental; potentially substantial extra cost

DET_SIZE = (256, 256)

MODEL_NAME = "buffalo_s"

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

PROCESS_EVERY_N_FRAMES = 2


AUDIO_COOLDOWN = 3.0

# How many consecutive detection cycles without a
# person before considering them to have left.
PERSON_LOST_CYCLES = 5


AUDIO_FILES = {
    "Raghav": "te.mp3",

    # Examples:
    # "Person2": "person2.mp3",
    # "Person3": "person3.wav",
}

# Optional audio for unknown faces.
PLAY_UNKNOWN_AUDIO = False
UNKNOWN_AUDIO_FILE = "unknown.wav"


try:
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
    print("Audio system initialized.")

except pygame.error as e:
    AUDIO_AVAILABLE = False
    print(f"Audio initialization failed: {e}")


def play_person_audio(name):
    """
    Play the audio assigned to a recognized person.
    """

    if not AUDIO_AVAILABLE:
        return

    if name == "Unknown":
        if not PLAY_UNKNOWN_AUDIO:
            return

        audio_file = UNKNOWN_AUDIO_FILE

    else:
        audio_file = AUDIO_FILES.get(name)

        if audio_file is None:
            return

    audio_path = os.path.join(SOUND_DIR, audio_file)

    if not os.path.isfile(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    try:
        if pygame.mixer.music.get_busy():
            return

        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        print(f"Playing audio for: {name}")

    except pygame.error as e:
        print(f"Audio playback error: {e}")



class ThreadedCamera:

    def __init__(self, source, width=None, height=None):

        backend = (
            cv2.CAP_DSHOW
            if sys.platform.startswith("win")
            else 0
        )

        self.cap = cv2.VideoCapture(source, backend)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video source: {source}"
            )

        if width:
            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                width
            )

        if height:
            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                height
            )

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.ret, self.frame = self.cap.read()

        if not self.ret or self.frame is None:
            self.cap.release()
            raise RuntimeError(
                "Camera opened but could not read a frame."
            )

        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self._update,
            daemon=True
        )

        self.thread.start()

    def _update(self):

        while self.running:

            ret, frame = self.cap.read()

            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):

        with self.lock:

            frame = (
                self.frame.copy()
                if self.frame is not None
                else None
            )

            return self.ret, frame

    def release(self):

        self.running = False

        self.thread.join(timeout=1)

        self.cap.release()


# ============================================================
# GPU SETUP
# ============================================================

def check_gpu():

    available = onnxruntime.get_available_providers()

    print(
        "ONNX Runtime available providers:",
        available
    )

    if "CUDAExecutionProvider" not in available:

        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. "
            "Check your ONNX Runtime GPU installation."
        )

    return available


def build_app():

    check_gpu()

    # Use CUDA directly.
    # TensorRT is deliberately excluded because its DLL
    # initialization was failing in your current setup.

    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=providers,
        allowed_modules=[
            "detection",
            "recognition",
        ],
    )

    app.prepare(
        ctx_id=0,
        det_size=DET_SIZE,
    )

    print("\nModel execution providers:")

    for model_name, model in app.models.items():

        print(
            model_name,
            model.session.get_providers()
        )

    return app


# ============================================================
# LOAD REFERENCE FACES
# ============================================================

def build_known_encodings(app):

    known_encodings = []
    known_names = []

    if not os.path.isdir(KNOWN_FACES_DIR):

        raise FileNotFoundError(
            "known_faces folder not found."
        )

    for person_name in sorted(
        os.listdir(KNOWN_FACES_DIR)
    ):

        person_dir = os.path.join(
            KNOWN_FACES_DIR,
            person_name
        )

        if not os.path.isdir(person_dir):
            continue

        image_files = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        for img_file in image_files:

            img_path = os.path.join(
                person_dir,
                img_file
            )

            img = cv2.imread(img_path)

            if img is None:
                continue

            faces = app.get(img)

            if not faces:

                print(
                    f"Warning: no face found in "
                    f"{img_path}, skipping."
                )

                continue

            # Use the largest face in each reference image.
            face = max(
                faces,
                key=lambda f: (
                    (f.bbox[2] - f.bbox[0])
                    * (f.bbox[3] - f.bbox[1])
                )
            )

            known_encodings.append(
                face.normed_embedding
            )

            known_names.append(person_name)

    if not known_encodings:

        raise ValueError(
            "No usable faces found in known_faces/. "
            "Add clearer photos."
        )

    print(
        f"Loaded {len(known_encodings)} reference photos "
        f"across {len(set(known_names))} people: "
        f"{sorted(set(known_names))}"
    )

    return (
        np.array(known_encodings),
        known_names
    )


# ============================================================
# FACE MATCHING
# ============================================================

def match_faces(
    faces,
    known_encodings,
    known_names
):

    results = []

    for face in faces:

        emb = face.normed_embedding

        similarities = known_encodings @ emb

        best_idx = np.argmax(similarities)

        best_sim = similarities[best_idx]

        x1, y1, x2, y2 = face.bbox.astype(int)

        if best_sim > SIMILARITY_THRESHOLD:

            name = known_names[best_idx]
            color = (0, 255, 0)

        else:

            name = "Unknown"
            color = (0, 0, 255)

        label = f"{name} ({best_sim:.2f})"

        results.append(
            (
                (x1, y1, x2, y2),
                name,
                label,
                color
            )
        )

    return results


# ============================================================
# DRAW RESULTS
# ============================================================

def draw_results(frame, results):

    for (
        (x1, y1, x2, y2),
        name,
        label,
        color
    ) in results:

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


# ============================================================
# AUDIO TRIGGER MANAGEMENT
# ============================================================

last_audio_name = None
last_audio_time = 0.0

current_person = None
person_lost_cycles = 0


def update_person_audio(results):

    global last_audio_name
    global last_audio_time
    global current_person
    global person_lost_cycles

    now = time.monotonic()

    if not results:

        person_lost_cycles += 1

        if person_lost_cycles >= PERSON_LOST_CYCLES:

            current_person = None

        return

    person_lost_cycles = 0

    # Select the largest detected face.
    largest_face = max(
        results,
        key=lambda result: (
            (result[0][2] - result[0][0])
            * (result[0][3] - result[0][1])
        )
    )

    name = largest_face[1]

    # If the person hasn't changed, don't restart audio.
    if name == current_person:
        return

    current_person = name

    # Cooldown applies to the same identity.
    if (
        name == last_audio_name
        and now - last_audio_time < AUDIO_COOLDOWN
    ):
        return

    play_person_audio(name)

    last_audio_name = name
    last_audio_time = now


# ============================================================
# MAIN
# ============================================================

def main():

    global AUDIO_AVAILABLE

    app = build_app()

    known_encodings, known_names = (
        build_known_encodings(app)
    )

    cam = None

    frame_count = 0

    last_results = []

    fps_timer = time.monotonic()
    fps_counter = 0
    fps_display = 0

    detect_ms = 0.0

    try:

        cam = ThreadedCamera(
            VIDEO_SOURCE,
            CAPTURE_WIDTH,
            CAPTURE_HEIGHT
        )

        print("\nCamera started.")
        print("Press Q to quit.")

        while True:

            ret, frame = cam.read()

            if not ret or frame is None:
                continue

            frame_count += 1

            # Run detection every N frames.
            if (
                frame_count
                % PROCESS_EVERY_N_FRAMES
                == 0
            ):

                t0 = time.perf_counter()

                faces = app.get(frame)

                detect_ms = (
                    time.perf_counter() - t0
                ) * 1000

                last_results = match_faces(
                    faces,
                    known_encodings,
                    known_names
                )

                update_person_audio(
                    last_results
                )

            draw_results(
                frame,
                last_results
            )

            fps_counter += 1

            now = time.monotonic()

            if now - fps_timer >= 1.0:

                fps_display = fps_counter

                fps_counter = 0
                fps_timer = now

            cv2.putText(
                frame,
                (
                    f"FPS: {fps_display} | "
                    f"Detect: {detect_ms:.2f} ms"
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                "Live Person Recognition (CUDA)",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        if cam is not None:
            cam.release()

        cv2.destroyAllWindows()

        if AUDIO_AVAILABLE:
            pygame.mixer.quit()


if __name__ == "__main__":
    main()