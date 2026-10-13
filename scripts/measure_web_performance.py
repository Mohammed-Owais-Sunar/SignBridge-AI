"""Phase 8 performance measurement: the video path, end to end (headless).

Run (project virtual environment, from the project folder):

    python scripts/measure_web_performance.py

WHY THIS EXISTS
    The first Phase 8 version re-ran the ENTIRE Streamlit script once
    per video frame (st.rerun + st.image) and the server log measured
    exactly 6 script runs per second - the preview was capped at ~6 FPS.
    The fix moved video out of Streamlit's render loop. This script
    measures that fixed path WITHOUT a browser, driven exactly the way
    the page drives it:

        CameraWorker thread -> src/mjpeg_server.py -> MJPEG client
        (Phase 5/6 loop)       (multipart feed)      (what <img> reads)

    It reports the processing rate, the rate actually DELIVERED to the
    client, the measured prediction latency and the process CPU, then
    verdicts the result against the documented 6 FPS baseline.
    The page side (no st.rerun loop, st.fragment statistics, MJPEG
    <img>) is covered by scripts/test_web_ui.py.

NOTES
    * The keepalive toucher emulates the page fragment, which touches
      the worker several times per second (src/camera_worker.py).
    * The first frame takes ~3 s (MediaPipe model warm-up), so the
      client waits up to FIRST_FRAME_TIMEOUT_SECONDS for the first
      bytes and only then switches to a short idle timeout.
    * Numbers depend on the machine and on what the camera delivers.
"""

import socket
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_realtime import load_classifier          # noqa: E402
from src import mjpeg_server                                  # noqa: E402
from src.camera_worker import CameraWorker                    # noqa: E402
from src.config import MODEL_PATH as HAND_MODEL_PATH          # noqa: E402
from src.hand_detector import HandDetector                    # noqa: E402

MEASURE_SECONDS = 15.0        # steady-state window (after warm-up)
WARMUP_SECONDS = 1.5          # camera open + MediaPipe first inference
FIRST_FRAME_TIMEOUT = 20.0    # first frame waits for the model warm-up
IDLE_TIMEOUT = 3.0            # no frame for this long -> stream is over
KEEPALIVE_INTERVAL = 0.5      # page fragment rate (app.py: 0.4 s)
MIN_FEED_FPS = 15.0           # verdict: well above the old 6 FPS cap
JPEG_MARKER = b"Content-Type: image/jpeg"


def feed_client(port, results):
    """Count the JPEG parts the feed delivers until the stream goes idle.

    Plain socket instead of http.client: with `Connection: close` the
    HTTPResponse hands the connection to the response object, which sets
    HTTPConnection.sock to None - the timeouts below would have nowhere
    to go. recv() also returns as soon as bytes arrive, so the very first
    frame is observed exactly when the feed writes it.
    """
    connection = None
    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=10)
        connection.settimeout(FIRST_FRAME_TIMEOUT)
        connection.sendall(
            b"GET /feed HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Connection: close\r\n\r\n")

        # --- status line + headers (sent immediately by the server) ------
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = connection.recv(4096)
            if not chunk:
                results["error"] = "connection closed before the headers"
                return
            buffer += chunk
        header, _, body = buffer.partition(b"\r\n\r\n")
        results["status"] = int(header.split(b"\r\n")[0].split()[1])

        # --- multipart body: count JPEG parts until IDLE_TIMEOUT ---------
        connection.settimeout(IDLE_TIMEOUT)
        received, tail = 0, b""
        first_at = last_at = None
        started = time.monotonic()
        try:
            while True:
                chunk = connection.recv(65536)
                if not chunk:            # server closed the connection
                    break
                data = tail + chunk
                hits = data.count(JPEG_MARKER)
                if hits:
                    now = time.monotonic()
                    first_at = first_at or now
                    last_at = now
                    received += hits
                tail = data[-(len(JPEG_MARKER) - 1):]   # no double counting
        except TimeoutError:
            pass                         # idle for IDLE_TIMEOUT: done
        except OSError as error:
            results["error"] = str(error)
        results.update(frames=received, first=first_at, last=last_at,
                       elapsed=time.monotonic() - started)
    except OSError as error:
        results["error"] = str(error)
    finally:
        if connection is not None:
            connection.close()


def main() -> int:
    print("=" * 62, flush=True)
    print("SIGNBRIDGE AI - PHASE 8 VIDEO PATH MEASUREMENT (headless)", flush=True)
    print("=" * 62, flush=True)

    model, class_names = load_classifier()
    detector = HandDetector(model_path=HAND_MODEL_PATH, max_hands=2,
                            detection_confidence=0.5,
                            tracking_confidence=0.5)
    worker = CameraWorker()
    port = mjpeg_server.ensure_running()
    print(f"feed url: http://localhost:{port}/feed", flush=True)

    worker.start(model, class_names, detector)

    stop_touching = threading.Event()

    def toucher():                       # emulates the page fragment
        while not stop_touching.wait(KEEPALIVE_INTERVAL):
            worker.touch()

    threading.Thread(target=toucher, daemon=True).start()
    time.sleep(WARMUP_SECONDS)
    if not worker.is_alive():
        print(f"WORKER FAILED TO START: {worker.error}", flush=True)
        stop_touching.set()
        return 1

    results = {}
    client = threading.Thread(target=feed_client, args=(port, results),
                              daemon=True)
    client.start()

    cpu0, wall0 = time.process_time(), time.monotonic()
    time.sleep(MEASURE_SECONDS)
    worker.stop()
    client.join(timeout=FIRST_FRAME_TIMEOUT)
    stop_touching.set()
    cpu1, wall1 = time.process_time(), time.monotonic()

    wall, cpu = wall1 - wall0, cpu1 - cpu0
    snap = worker.snapshot
    processed = snap.get("frames", 0)
    delivered = results.get("frames", 0)
    span = (results.get("last") or 0) - (results.get("first") or 0)
    feed_fps = (delivered - 1) / span if delivered > 1 and span > 0 else 0.0
    processing_fps = processed / wall if wall else 0.0

    print("\n" + "=" * 62, flush=True)
    print("MEASUREMENT", flush=True)
    print("=" * 62, flush=True)
    print(f"window              : {wall:.1f} s", flush=True)
    print(f"worker processed    : {processed} frames"
          f"  ({processing_fps:.1f} FPS)", flush=True)
    print(f"MJPEG delivered     : {delivered} parts, {feed_fps:.1f} FPS"
          f" to the client (over {span:.1f} s)", flush=True)
    print(f"feed status         : {results.get('status')}", flush=True)
    if snap.get("latency") is not None:
        print(f"prediction latency  : {snap['latency']:.1f} ms"
              " (EMA, features + model)", flush=True)
    print(f"process CPU         : {cpu:.1f} s / {wall:.1f} s wall"
          f" = {cpu / wall * 100:.0f}% of one core", flush=True)
    if results.get("error"):
        print(f"client error        : {results['error']}", flush=True)
    if snap.get("error"):
        print(f"worker error        : {snap['error']}", flush=True)

    print("\n" + "-" * 62, flush=True)
    if feed_fps >= MIN_FEED_FPS:
        print(f"VERDICT: PASS - {feed_fps:.1f} FPS delivered vs the old"
              f" ~6 FPS cap (video stays outside Streamlit's render loop).",
              flush=True)
        return 0
    print(f"VERDICT: FAIL - {feed_fps:.1f} FPS delivered, expected at least"
          f" {MIN_FEED_FPS:.0f} FPS on this machine.", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
