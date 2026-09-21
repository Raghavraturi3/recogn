"""
main.py — Single-file face recognition server using the browser as the
camera source (supports OBS Virtual Camera or any webcam).

Architecture:
    Browser (OBS/webcam)
        -> captures frames
        -> sends JPEGs over WebSocket to server
    Server
        -> runs InsightFace recognition on each Nth frame
        -> matches against known_faces/
        -> draws boxes + labels
        -> sends annotated JPEG + JSON metadata back
    Browser
        -> displays annotated frames live
        -> optionally plays audio clips from /sound/

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000

Then open: http://localhost:8000

Folder layout expected:
    your_project/
        main.py               <- this file
        live_detect.py        <- your existing helpers
        known_faces/
            Raghav/  *.jpg
            Rahul/   *.jpg
        sound/
            t.mp3
            unknown.wav
"""

import asyncio
import os
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import live_detect as ld


# ================================================================ CONFIG ===

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUND_DIR = os.path.join(BASE_DIR, "sound")

# JPEG quality of the server -> browser annotated frame.
OUTPUT_JPEG_QUALITY = 75

# Downscale incoming browser frames to this width before model inference.
PROCESS_WIDTH = 640

# Optional: enable browser-side audio playback.
# If True, the HTML page plays /sound/<file> when a known face is detected.
# You can also leave server-side pygame audio running (it plays on the
# server machine's speakers).
ENABLE_BROWSER_AUDIO = True

# Map name -> audio file inside sound/ (used only if ENABLE_BROWSER_AUDIO).
BROWSER_AUDIO_FILES = {
    "Raghav": "t.mp3",
    "Rahul": "t.mp3",
}

# Cooldown between plays for the same person (ms).
BROWSER_AUDIO_COOLDOWN_MS = 3000


# ============================================================== APP SETUP ===

app = FastAPI(title="Face Recognition (Browser Camera)")


# Global model state — loaded once at startup.
face_app = None
known_encodings = None
known_names = None


def _load_models():
    """Blocking model load. Runs in a worker thread at startup."""
    global face_app, known_encodings, known_names

    print("=" * 60)
    print("Loading face recognition model...")
    print("=" * 60)

    face_app = ld.build_app()
    known_encodings, known_names = ld.build_known_encodings(face_app)

    print("=" * 60)
    print("Model ready. Open http://localhost:8000 in a browser.")
    print("=" * 60)


@app.on_event("startup")
async def _on_startup():
    await asyncio.to_thread(_load_models)


@app.on_event("shutdown")
async def _on_shutdown():
    if ld.AUDIO_AVAILABLE:
        try:
            ld.pygame.mixer.quit()
        except Exception as e:
            print("Audio shutdown warning:", e)


# Serve /sound/* so the browser can fetch audio clips.
if os.path.isdir(SOUND_DIR):
    app.mount("/sound", StaticFiles(directory=SOUND_DIR), name="sound")
    print(f"Serving audio from: {SOUND_DIR}")
else:
    print(f"WARNING: sound folder not found at {SOUND_DIR}")


# ============================================================ WEBSOCKET ====

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("Browser connected.")

    frame_count = 0
    last_results = []
    detect_ms = 0.0

    fps_timer = time.monotonic()
    fps_count = 0
    fps_display = 0

    try:
        while True:
            # --- 1. Receive a raw JPEG frame from the browser ---
            data = await ws.receive_bytes()

            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # --- 2. Downscale to bound inference cost ---
            h, w = frame.shape[:2]
            if w > PROCESS_WIDTH:
                scale = PROCESS_WIDTH / w
                new_h = int(h * scale)
                frame = cv2.resize(
                    frame,
                    (PROCESS_WIDTH, new_h),
                    interpolation=cv2.INTER_AREA,
                )

            frame_count += 1
            fps_count += 1

            # --- 3. Run recognition every Nth frame ---
            if frame_count % ld.PROCESS_EVERY_N_FRAMES == 0:
                t0 = time.perf_counter()
                faces = face_app.get(frame)
                detect_ms = (time.perf_counter() - t0) * 1000

                last_results = ld.match_faces(
                    faces, known_encodings, known_names
                )

                # Server-side audio (plays on server PC's speakers).
                # If you only want browser audio, comment the next line.
                ld.update_person_audio(last_results)

            # --- 4. Draw boxes + labels on the current frame ---
            ld.draw_results(frame, last_results)

            # --- 5. FPS display ---
            now = time.monotonic()
            if now - fps_timer >= 1.0:
                fps_display = fps_count
                fps_count = 0
                fps_timer = now

            cv2.putText(
                frame,
                f"FPS: {fps_display} | Detect: {detect_ms:.2f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # --- 6. Encode annotated frame ---
            ok, buf = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY],
            )
            if not ok:
                continue

            # --- 7. Send metadata (JSON) then the image (binary) ---
            names = sorted({r[1] for r in last_results})
            await ws.send_json({
                "names": names,
                "fps": fps_display,
                "detect_ms": round(detect_ms, 2),
            })
            await ws.send_bytes(buf.tobytes())

    except WebSocketDisconnect:
        print("Browser disconnected.")
    except Exception as e:
        print("WS error:", e)


# ================================================================= ROUTES ===

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_HTML)


@app.get("/health")
def health():
    return {
        "status": "running",
        "recognition_ready": face_app is not None,
    }


# ================================================================== HTML ====

# Note: double curly braces {{ }} are NOT needed here because this is a
# raw string returned as-is (no .format() applied). The JS below uses
# template literals with backticks which are fine inside a Python r-string.

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Recognition Dashboard</title>
<style>
  * { box-sizing: border-box; }

  body {
    margin: 0;
    padding: 24px;
    background: #10141c;
    color: #f1f5f9;
    font-family: Arial, sans-serif;
  }

  .container { max-width: 1100px; margin: auto; }

  h1 { margin: 0 0 8px 0; }
  h3 { margin: 8px 0; font-size: 15px; color: #cbd5e1; }

  .status {
    margin: 12px 0;
    color: #86efac;
    min-height: 20px;
  }
  .status.error { color: #fca5a5; }

  .muted { color: #94a3b8; }

  .panel {
    background: #1e293b;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 16px;
    border: 1px solid #334155;
  }

  button, select {
    padding: 10px 14px;
    margin: 6px 6px 0 0;
    border-radius: 8px;
    border: 1px solid #475569;
    font-size: 14px;
  }
  button {
    background: #2563eb;
    color: white;
    cursor: pointer;
  }
  button:hover { background: #1d4ed8; }
  button:disabled { opacity: .6; cursor: not-allowed; }

  select {
    background: #0f172a;
    color: white;
    min-width: 260px;
  }

  .row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
  }
  .row > div {
    flex: 1 1 320px;
    min-width: 320px;
  }

  video, img.output {
    width: 100%;
    display: block;
    border-radius: 12px;
    border: 1px solid #334155;
    background: #000;
    aspect-ratio: 4 / 3;
    object-fit: cover;
  }

  .metrics {
    font-family: monospace;
    color: #cbd5e1;
    margin: 4px 0;
  }

  .hidden { display: none; }
</style>
</head>
<body>
<div class="container">

  <h1>Face Recognition Dashboard</h1>
  <div class="status" id="status">
    Click "Request Camera Permission" to begin.
  </div>

  <div class="panel">
    <button id="permBtn">Request Camera Permission</button>
    <br>
    <select id="cameraSelect" disabled>
      <option value="">Request permission first</option>
    </select>
    <br>
    <button id="startBtn" disabled>Start</button>
    <button id="stopBtn" disabled>Stop</button>
    <p class="muted">
      Tip: start OBS Virtual Camera first, then pick it from the list.
      It will auto-select if its label contains "OBS".
    </p>
  </div>

  <div class="row">
    <div>
      <h3>Local preview (your camera)</h3>
      <video id="preview" autoplay playsinline muted></video>
    </div>
    <div>
      <h3>Recognized (server output)</h3>
      <img id="output" class="output" alt="annotated feed">
    </div>
  </div>

  <div class="panel" style="margin-top:16px">
    <h3>Recognition Status</h3>
    <p id="people" class="metrics">No faces detected</p>
    <p id="metrics" class="metrics">FPS: - | Detect: - ms</p>
  </div>

</div>

<script>
// ------------------------------------------------------------------ DOM ---
const statusEl  = document.getElementById("status");
const permBtn   = document.getElementById("permBtn");
const startBtn  = document.getElementById("startBtn");
const stopBtn   = document.getElementById("stopBtn");
const camSelect = document.getElementById("cameraSelect");
const preview   = document.getElementById("preview");
const output    = document.getElementById("output");
const peopleEl  = document.getElementById("people");
const metricsEl = document.getElementById("metrics");

// ------------------------------------------------------------- Globals ---
let stream    = null;
let ws        = null;
let sendLoop  = null;
let recvUrl   = null;
let recvBusy  = false;

const SEND_INTERVAL_MS = 66;    // ~15 fps upload
const SEND_WIDTH       = 640;   // downscale before sending
const SEND_QUALITY     = 0.7;

// -------------------------------------------------- Permission + devices ---
permBtn.onclick = async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus(
      "Camera API unavailable. Use localhost or HTTPS.",
      true
    );
    return;
  }

  permBtn.disabled = true;
  setStatus("Requesting camera permission...");

  let tmp = null;
  try {
    tmp = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });
    tmp.getTracks().forEach(t => t.stop());
    tmp = null;

    await loadDevices();
    setStatus("Permission granted. Pick a camera and press Start.");
  } catch (e) {
    if (tmp) tmp.getTracks().forEach(t => t.stop());
    setStatus(e.name + ": " + e.message, true);
  } finally {
    permBtn.disabled = false;
  }
};

async function loadDevices() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter(d => d.kind === "videoinput");

  camSelect.innerHTML = "";

  if (!cams.length) {
    camSelect.add(new Option("No cameras found", ""));
    camSelect.disabled = true;
    startBtn.disabled = true;
    return;
  }

  cams.forEach((c, i) => {
    const label = c.label || ("Camera " + (i + 1));
    camSelect.add(new Option(label, c.deviceId));
  });

  // Auto-select OBS Virtual Camera if present.
  const obs = cams.find(c =>
    (c.label || "").toLowerCase().includes("obs")
  );
  if (obs) camSelect.value = obs.deviceId;

  camSelect.disabled = false;
  startBtn.disabled = false;
}

// -------------------------------------------------------- Start / stop ----
startBtn.onclick = async () => {
  const deviceId = camSelect.value;
  if (!deviceId) {
    setStatus("Pick a camera first.", true);
    return;
  }

  startBtn.disabled = true;
  setStatus("Starting camera...");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: { exact: deviceId },
        width:  { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    preview.srcObject = stream;
    await preview.play();

    openSocket();
    stopBtn.disabled = false;
    setStatus("Running. Recognition active.");

  } catch (e) {
    setStatus(e.name + ": " + e.message, true);
    startBtn.disabled = false;
  }
};

stopBtn.onclick = () => stopAll();

function stopAll() {
  if (sendLoop) { clearInterval(sendLoop); sendLoop = null; }
  if (ws) { try { ws.close(); } catch (_) {} ws = null; }
  if (recvUrl) { URL.revokeObjectURL(recvUrl); recvUrl = null; }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  preview.srcObject = null;
  output.src = "";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Stopped.");
  peopleEl.textContent = "No faces detected";
  metricsEl.textContent = "FPS: - | Detect: - ms";
}

// ------------------------------------------------------ WebSocket loop ----
function openSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(proto + "://" + location.host + "/ws");
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    sendLoop = setInterval(() => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      if (recvBusy) return;              // backpressure
      if (!preview.videoWidth) return;   // video not ready yet

      const scale = SEND_WIDTH / preview.videoWidth;
      canvas.width  = SEND_WIDTH;
      canvas.height = Math.round(preview.videoHeight * scale);

      ctx.drawImage(preview, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        (blob) => {
          if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return;
          recvBusy = true;
          blob.arrayBuffer().then(buf => ws.send(buf));
        },
        "image/jpeg",
        SEND_QUALITY
      );
    }, SEND_INTERVAL_MS);
  };

  ws.onmessage = (ev) => {
    // Text message = metadata
    if (typeof ev.data === "string") {
      try {
        const info = JSON.parse(ev.data);

        if (info.names && info.names.length) {
          peopleEl.textContent = "Detected: " + info.names.join(", ");
        } else {
          peopleEl.textContent = "No faces detected";
        }

        metricsEl.textContent =
          "FPS: " + info.fps +
          " | Detect: " + info.detect_ms + " ms";

        playBrowserAudio(info.names || []);
      } catch (e) {
        // ignore malformed JSON
      }
      return;
    }

    // Binary message = annotated frame
    recvBusy = false;
    const blob = new Blob([ev.data], { type: "image/jpeg" });
    if (recvUrl) URL.revokeObjectURL(recvUrl);
    recvUrl = URL.createObjectURL(blob);
    output.src = recvUrl;
  };

  ws.onclose = () => {
    recvBusy = false;
  };

  ws.onerror = (e) => {
    console.error("WS error", e);
  };
}

// ----------------------------------------------------- Browser audio ------
const ENABLE_BROWSER_AUDIO = __ENABLE_BROWSER_AUDIO__;
const AUDIO_FILES = __AUDIO_FILES_JSON__;
const AUDIO_COOLDOWN_MS = __AUDIO_COOLDOWN_MS__;

let currentPerson = null;
let lastPlayed = { name: null, t: 0 };
let currentAudio = null;

function playBrowserAudio(names) {
  if (!ENABLE_BROWSER_AUDIO) return;

  const known = names.find(n => AUDIO_FILES[n]);
  if (!known) {
    // Reset so the same person triggers again after leaving + returning.
    currentPerson = null;
    return;
  }

  if (known === currentPerson) return;
  currentPerson = known;

  const now = Date.now();
  if (lastPlayed.name === known &&
      now - lastPlayed.t < AUDIO_COOLDOWN_MS) {
    return;
  }

  const file = AUDIO_FILES[known];
  if (!file) return;

  try {
    if (currentAudio) {
      currentAudio.pause();
      currentAudio = null;
    }
    currentAudio = new Audio("/sound/" + encodeURIComponent(file));
    currentAudio.play().catch(() => {
      // Autoplay may be blocked until the user interacts with the page.
      // The Start button click usually satisfies this.
    });
    lastPlayed = { name: known, t: now };
  } catch (e) {
    // ignore
  }
}

// ------------------------------------------------------------ Utilities ---
function setStatus(msg, isError) {
  statusEl.textContent = msg;
  statusEl.className = "status" + (isError ? " error" : "");
}

window.addEventListener("beforeunload", stopAll);
</script>
</body>
</html>
"""


# ======================================================= TEMPLATE INJECT ===

# Inject Python config values into the HTML's JS block. We do it once at
# import time so every request gets the same rendered string.
INDEX_HTML = (
    INDEX_HTML
    .replace("__ENABLE_BROWSER_AUDIO__", "true" if ENABLE_BROWSER_AUDIO else "false")
    .replace("__AUDIO_FILES_JSON__", _json := __import__("json").dumps(BROWSER_AUDIO_FILES))
    .replace("__AUDIO_COOLDOWN_MS__", str(BROWSER_AUDIO_COOLDOWN_MS))
)


# ================================================================== MAIN ====

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )