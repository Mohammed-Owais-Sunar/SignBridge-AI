"""Phase 8: continuous recognition worker (background thread).

WHY: the Streamlit page must NOT re-run once per video frame (the first
Phase 8 version did, which capped the preview at ~6 FPS). This worker
owns the camera and runs the UNCHANGED Phase 5/6 loop at full speed in
its own thread:

    capture -> mirror once -> MediaPipe Hand Landmarker -> parse_hands
    -> draw_landmarks -> build_sample() -> predict_gesture() [Random
    Forest] -> VoteWindow -> confidence threshold -> SentenceBuilder
    .observe() -> overlay -> JPEG -> src/mjpeg_server.push()

Every step above is IMPORTED from the existing Phase 5/6 modules -
scripts/predict_realtime.py (predict_gesture, VoteWindow, put_text),
src/feature_extractor.py (build_sample), src/hand_detector.py,
src/sentence_builder.py and src/config.py (threshold 0.50, vote window
5, re-arm/cooldown rules). Nothing is re-implemented, skipped or tuned
here; the model and dataset are not touched.

Threading contract:
- the worker NEVER calls Streamlit functions (it has no
  ScriptRunContext); it prints logs and publishes a snapshot dict
- SentenceBuilder.observe() runs ONLY in this thread; manual UI actions
  (SPACE / DELETE / Clear buttons) are serialized through the same
  lock, so Phase 6 edge-trigger/cooldown semantics stay exact
- the page reads builder.sentence and the snapshot (atomic reads) at a
  low rate (st.fragment, ~2-4 Hz); the browser receives video from the
  MJPEG feed, completely independent of Streamlit's render loop
- keepalive: the page "touches" the worker on every fragment run; if
  no page is attached for KEEPALIVE_SECONDS AND no browser is pulling
  the MJPEG preview, the camera is released automatically (browser tab
  closed without pressing Stop). The feed-client check matters because
  browsers throttle JS timers in backgrounded tabs: the tab stops
  touching, but its preview connection stays open - the camera must not
  die while a viewer is still attached.
- ownership: before opening the device, the worker takes the
  cross-process lock from src/camera_lock.py, so a second SignBridge
  process fails fast and names the PID that already owns the camera
  instead of failing with a generic OpenCV error.
"""

import gc
import os
import threading
import time

import cv2

from src import camera_lock, mjpeg_server
from src.config import (CAMERA_BACKEND, CAMERA_INDEX,
                        MIN_PREDICTION_CONFIDENCE,
                        PREDICTION_VOTE_WINDOW)
from src.feature_extractor import build_sample
from src.hand_detector import parse_hands
from src.sentence_builder import NO_HAND_LABEL, SentenceBuilder
from scripts.predict_realtime import (COLOR_CYAN, COLOR_GREEN, COLOR_RED,
                                      COLOR_WHITE, COLOR_YELLOW,
                                      VoteWindow, predict_gesture, put_text)

KEEPALIVE_SECONDS = 10.0    # release the camera after this long without a page
JPEG_QUALITY = 82           # preview quality (640x480 JPEG ~= 40 KB)
WARMUP_FRAMES = 5           # frames dropped after opening (black first frames)

# Camera-open failure messages. The distinction matters: on Windows
# VideoCapture can REPORT a successful open while another process owns the
# device - only a failed read/grab exposes the real state. Each message
# tells the user what to actually do instead of just "camera not working".
OPEN_FAILED_MESSAGE = (
    "Could not open the webcam (index {index}). It is probably in use by "
    "another application (a second SignBridge window, Teams/Zoom/browser "
    "call, the Windows Camera app, or scripts/predict_realtime.py), or "
    "Windows camera privacy is off: Settings > Privacy & security > Camera.")
NO_FRAMES_MESSAGE = (
    "Webcam {index} opened but delivered no frames - another application "
    "is holding the device (Windows allows the open but starves the "
    "reads). Close other camera apps, then press Start again.")
READ_FAILED_MESSAGE = (
    "Failed to read a frame from webcam {index} - the camera was "
    "disconnected or another application took over the device. Close other "
    "camera apps, then press Start again.")


class CameraWorker:
    """Runs the Phase 5/6 pipeline continuously and publishes results."""

    def __init__(self, camera_factory=None, keepalive_seconds=None):
        # SentenceBuilder is owned by this worker (single feed = single
        # sentence); the lock serializes observe() and manual UI actions.
        self.lock = threading.RLock()
        self.builder = SentenceBuilder()
        self._vote = VoteWindow(PREDICTION_VOTE_WINDOW)
        self.last_action = None
        self.error = None
        self.running = False

        # Test seam: lifecycle tests inject a fake camera factory here; the
        # production default opens the configured OpenCV device.
        self._camera_factory = camera_factory or cv2.VideoCapture
        self._keepalive = float(KEEPALIVE_SECONDS if keepalive_seconds is None
                                else keepalive_seconds)

        # Page-facing state: replaced wholesale (atomic dict assignment)
        # at frame rate, READ by the page only several times per second.
        self.snapshot = {"running": False, "prediction": NO_HAND_LABEL,
                         "confidence": None, "raw_text": "", "note": "",
                         "hands": 0, "fps": None, "latency": None,
                         "frames": 0, "error": None}

        self._model = None
        self._class_names = None
        self._detector = None
        self._thread = None
        self._stop = threading.Event()
        self._activity = time.monotonic()
        self._control = threading.Lock()   # guards start/stop only
        self._stats = self._fresh_stats()

    # ------------------------------------------------------------- lifecycle
    @staticmethod
    def _fresh_stats():
        return {"frames": 0, "fps_samples": [],
                "latency_sum": 0.0, "latency_count": 0, "latency_max": 0.0,
                "detect_sum": 0.0, "detect_count": 0,
                "encode_sum": 0.0, "encode_count": 0,
                "frame_ms_sum": 0.0, "frame_ms_max": 0.0,
                # per-stage timings (repair instrumentation, monotonic clock)
                "capture_sum": 0.0, "capture_count": 0,
                "flip_sum": 0.0,
                "color_sum": 0.0, "infer_sum": 0.0, "infer_count": 0,
                "draw_sum": 0.0,
                "features_sum": 0.0, "features_count": 0,
                "model_sum": 0.0, "model_count": 0,
                "smooth_sum": 0.0,
                "push_sum": 0.0, "push_count": 0,
                "capture_errors": 0, "sent": 0}

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def is_stopping(self):
        """True while a stopped worker thread is still winding down.

        The UI uses this to show a 'Stopping...' state instead of offering
        a Start button that would silently do nothing while the old thread
        still owns the camera.
        """
        return self.is_alive() and not self.running

    def touch(self):
        """Called by the page (every run / fragment run): 'a page exists'."""
        self._activity = time.monotonic()

    def start(self, model, class_names, detector):
        """Spawn the feed thread (idempotent).

        Returns '' when the worker is running (or just started), and a
        short human-readable message when the start cannot happen yet -
        the previous session's thread is still releasing the camera. The
        page shows that message instead of silently ignoring the click.
        Camera failures themselves are reported asynchronously via
        error/snapshot.
        """
        with self._control:
            if self.is_alive():
                if self._stop.is_set() or not self.running:
                    return ("The previous camera session is still stopping - "
                            "the camera is being released. Try again in a "
                            "moment.")
                return ""   # already running: nothing to do
            self._model, self._class_names, self._detector = \
                model, class_names, detector
            self._stop.clear()
            self.error = None
            self.last_action = None
            self._stats = self._fresh_stats()
            self._vote.clear()
            self._activity = time.monotonic()
            self.running = True
            self._thread = threading.Thread(target=self._run,
                                            name="signbridge-feed",
                                            daemon=True)
            self._thread.start()
        return ""

    def stop(self, timeout=0.5):
        """Signal the thread to stop WITHOUT blocking the Streamlit thread.

        The join only absorbs a normal frame-time shutdown (~1 frame +
        release, well under the default timeout). If the camera backend is
        slow to release, we return anyway: the thread finishes on its own,
        releases the camera in its finally block, and the page shows the
        'stopping' state until is_alive() turns False. A blocked UI thread
        used to freeze the page for up to 3 s on every Stop click.
        """
        with self._control:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        self.running = False
        return thread is not None and thread.is_alive()

    # ---------------------------------------------------------- manual input
    def manual(self, operation):
        """Apply a deliberate UI action (SPACE / DELETE / CLEAR).

        Serialized with observe() through the same lock. Like the Phase 6
        keyboard action, this bypasses the edge trigger and the cooldown:
        those bound GESTURE repeats, not deliberate clicks.
        """
        with self.lock:
            if operation == "SPACE":
                self.builder.insert_space()
                self.last_action = "SPACE"
            elif operation == "DELETE":
                self.last_action = self.builder.delete_word()
            else:
                self.builder.clear()
                self.last_action = "CLEAR"
            sentence = self.builder.sentence
        print(f"[feed] manual: {self.last_action} -> \"{sentence}\"",
              flush=True)

    # ------------------------------------------------------------ internals
    def _set_error(self, message):
        self.error = message
        snapshot = dict(self.snapshot)
        snapshot.update(running=False, error=message)
        self.snapshot = snapshot
        print(f"[feed] ERROR: {message}", flush=True)

    @staticmethod
    def _overlay(frame, prediction, confidence, sentence, fps, latency):
        """On-frame status, drawn with Phase 5's own put_text/colors."""
        height = frame.shape[0]
        if prediction == NO_HAND_LABEL:
            color = COLOR_RED
        elif prediction in ("Uncertain", "Sample rejected"):
            color = COLOR_YELLOW
        else:
            color = COLOR_GREEN
        put_text(frame, f"Gesture: {prediction}", (10, 38), color, 1.0, 2)
        if confidence is not None:
            put_text(frame, f"Confidence: {confidence * 100:.1f}%",
                     (10, 72), COLOR_WHITE, 0.75, 2)
        visible = sentence if len(sentence) <= 46 else "..." + sentence[-43:]
        put_text(frame, f"Message: {visible if visible else '(empty)'}",
                 (10, height - 44), COLOR_CYAN, 0.7, 2)
        line = f"{fps:.1f} FPS"
        if latency is not None:
            line += f"   pred {latency:.1f} ms"
        put_text(frame, line, (10, height - 14), COLOR_WHITE, 0.55, 1)

    @staticmethod
    def _ema(previous, value):
        return value if previous is None else 0.9 * previous + 0.1 * value

    def _log_second(self, elapsed_in_second, count, stats, frame_ms,
                    stage_ema, hands, sentence):
        fps_now = count / max(elapsed_in_second, 1e-6)
        stats["fps_samples"].append(fps_now)
        samples = stats["fps_samples"]
        pred_avg = (stats["latency_sum"] / stats["latency_count"]
                    if stats["latency_count"] else 0.0)
        # Per-stage EMA of the last second (all measured, none simulated).
        # cap=camera.read flip=mirror rgb=BGR->RGB mp=MediaPipe inference
        # draw=parse+draw feat=build_sample rf=Random Forest smooth=vote +
        # SentenceBuilder jpg=JPEG encode push=MJPEG publish.
        print(f"[feed] t={len(samples)}s fps={fps_now:.1f} "
              f"avg={sum(samples) / len(samples):.1f} "
              f"min={min(samples):.1f} max={max(samples):.1f} | "
              f"frame={frame_ms:.1f} ms "
              f"(cap {stage_ema.get('capture') or 0:.1f} / "
              f"flip {stage_ema.get('flip') or 0:.1f} / "
              f"rgb {stage_ema.get('color') or 0:.1f} / "
              f"mp {stage_ema.get('infer') or 0:.1f} / "
              f"draw {stage_ema.get('draw') or 0:.1f} / "
              f"feat {stage_ema.get('features') or 0:.1f} / "
              f"rf {stage_ema.get('model') or 0:.1f} / "
              f"smooth {stage_ema.get('smooth') or 0:.1f} / "
              f"jpg {stage_ema.get('encode') or 0:.1f} / "
              f"push {stage_ema.get('push') or 0:.1f}) | "
              f"pred_avg={pred_avg:.1f} ms | "
              f"hands={hands} | \"{sentence}\"", flush=True)

    @staticmethod
    def _stage_line(label, total, count):
        if not count:
            return None
        return f"{label:<23}: {total / count:6.2f} ms  ({count} frames)"

    def _print_summary(self, duration, stats, camera_fps, sentence):
        samples = stats["fps_samples"]
        print("\n" + "=" * 60, flush=True)
        print("PHASE 8 FEED SUMMARY (background worker)", flush=True)
        print("=" * 60, flush=True)
        print(f"Duration            : {duration:.1f} s", flush=True)
        print(f"Frames processed    : {stats['frames']}", flush=True)
        if samples:
            print(f"FPS (per-second)    : avg "
                  f"{sum(samples) / len(samples):.1f}  min {min(samples):.1f}"
                  f"  max {max(samples):.1f}  ({len(samples)} samples)",
                  flush=True)
        if stats["frames"]:
            print(f"Frame time          : avg "
                  f"{stats['frame_ms_sum'] / stats['frames']:.1f} ms  "
                  f"max {stats['frame_ms_max']:.1f} ms "
                  f"(capture + MediaPipe + features + model + JPEG)",
                  flush=True)
        if stats["latency_count"]:
            print(f"Prediction latency  : avg "
                  f"{stats['latency_sum'] / stats['latency_count']:.2f} ms  "
                  f"max {stats['latency_max']:.2f} ms "
                  f"({stats['latency_count']} predictions, "
                  f"features + model only)", flush=True)
        if stats["detect_count"]:
            print(f"MediaPipe detect    : avg "
                  f"{stats['detect_sum'] / stats['detect_count']:.1f} ms "
                  f"({stats['detect_count']} frames)", flush=True)
        if stats["encode_count"]:
            print(f"JPEG encode         : avg "
                  f"{stats['encode_sum'] / stats['encode_count']:.1f} ms "
                  f"({stats['encode_count']} frames)", flush=True)
        print(f"Capture errors      : {stats['capture_errors']}",
              flush=True)
        print(f"MJPEG frames sent   : {stats['sent']}", flush=True)
        expected = camera_fps * duration
        dropped = int(max(0.0, expected - stats["frames"]))
        print(f"Camera rate         : {camera_fps:.0f} FPS expected -> "
              f"~{dropped} frames handled by the camera driver queue "
              f"(read at processing pace, same behaviour as Phase 5)",
              flush=True)
        # Per-stage breakdown (monotonic clock around each stage). The
        # pipeline has NO resize stage: MediaPipe and the encoder both work
        # on the native 640x480 frame, so no scaling cost exists.
        stages = [
            self._stage_line("capture (camera.read)",
                             stats["capture_sum"], stats["capture_count"]),
            self._stage_line("mirror (cv2.flip)",
                             stats["flip_sum"], stats["frames"]),
            self._stage_line("color BGR->RGB",
                             stats["color_sum"], stats["detect_count"]),
            self._stage_line("MediaPipe inference",
                             stats["infer_sum"], stats["infer_count"]),
            self._stage_line("parse + draw landmarks",
                             stats["draw_sum"], stats["frames"]),
            self._stage_line("build_sample (normalize)",
                             stats["features_sum"], stats["features_count"]),
            self._stage_line("Random Forest inference",
                             stats["model_sum"], stats["model_count"]),
            self._stage_line("vote + observe (smooth)",
                             stats["smooth_sum"], stats["frames"]),
            self._stage_line("JPEG encode",
                             stats["encode_sum"], stats["encode_count"]),
            self._stage_line("MJPEG push",
                             stats["push_sum"], stats["push_count"]),
            self._stage_line("end-to-end frame",
                             stats["frame_ms_sum"], stats["frames"]),
        ]
        print("Stage timings (avg ms/frame, monotonic clock):", flush=True)
        for line in stages:
            if line:
                print(f"  {line}", flush=True)
        print(f"Sentence            : \"{sentence}\"", flush=True)
        print("No speech/audio was produced (TTS belongs to Phase 7).",
              flush=True)
        print("=" * 60, flush=True)

    # ------------------------------------------------------------ main loop
    def _open_camera(self):
        """Open the configured webcam (backend pinnable via src/config)."""
        if CAMERA_BACKEND:
            backend = getattr(cv2, f"CAP_{CAMERA_BACKEND.upper()}", None)
            if backend is not None:
                return self._camera_factory(CAMERA_INDEX, backend)
        return self._camera_factory(CAMERA_INDEX)

    def _run(self):
        camera = None
        owned = None        # cross-process camera lock (src/camera_lock)
        camera_fps = 30.0
        stats = self._stats
        started = time.monotonic()
        sentence = ""
        try:
            try:
                # 1) Cross-process ownership FIRST: a second SignBridge
                #    instance must fail fast with the holder's PID instead
                #    of a generic OpenCV error deep inside the loop.
                try:
                    owned = camera_lock.acquire(CAMERA_INDEX)
                except camera_lock.CameraBusyError as busy:
                    if busy.holder_pid == os.getpid():
                        detail = ("another SignBridge session in THIS "
                                  "process (close the extra browser tab "
                                  "or window)")
                    elif busy.holder_pid:
                        detail = (f"another SignBridge process "
                                  f"(PID {busy.holder_pid}) - stop it, "
                                  f"e.g. Get-Process -Id {busy.holder_pid}")
                    else:
                        detail = "another SignBridge process"
                    self._set_error(
                        f"Webcam {CAMERA_INDEX} is already owned by "
                        f"{detail}. See README \"Camera in use\" for how "
                        f"to find and close the holder.")
                    return

                # 2) Open the device. On Windows the open can REPORT success
                #    while another process owns the camera - only the
                #    warm-up grabs below really prove we may read frames.
                camera = self._open_camera()
                if camera is None or not camera.isOpened():
                    if camera is not None:
                        camera.release()
                        camera = None
                    self._set_error(
                        OPEN_FAILED_MESSAGE.format(index=CAMERA_INDEX))
                    return

                reported = camera.get(cv2.CAP_PROP_FPS)
                if reported and 1 <= reported <= 120:
                    camera_fps = float(reported)
                grabbed = 0
                for _ in range(WARMUP_FRAMES):
                    # drop warm-up frames (Phase 5 behaviour); a device
                    # owned elsewhere fails EVERY grab although it claims
                    # to be opened
                    grabbed += int(bool(camera.grab()))
                if grabbed == 0:
                    camera.release()
                    camera = None
                    self._set_error(
                        NO_FRAMES_MESSAGE.format(index=CAMERA_INDEX))
                    return
                # Freeze the already-loaded static graph (MediaPipe model,
                # Random Forest, module caches) into the permanent GC
                # generation: generational collections then stop scanning it
                # every few seconds, which removes the periodic ~100 ms
                # frame-time spikes it caused in the live browser session.
                gc.freeze()
                print(f"[feed] camera opened (index {CAMERA_INDEX}), "
                      f"expecting ~{camera_fps:.0f} FPS", flush=True)

                vote = self._vote
                model, class_names = self._model, self._class_names
                detector = self._detector

                previous = None
                fps_ema = None
                latency_ema = None
                stage_ema = {}
                second_mark = time.monotonic()
                second_count = 0

                while not self._stop.is_set():
                    # Keepalive: release the camera when NO page is attached
                    # AND no browser is still streaming the preview. The
                    # feed-client check matters because browsers throttle
                    # JS timers in backgrounded tabs (page stops touching)
                    # while the MJPEG connection stays open - the camera
                    # must not die in that state.
                    if (time.monotonic() - self._activity > self._keepalive
                            and mjpeg_server.connected_clients() == 0):
                        print("[feed] no browser page and no preview "
                              f"client for {self._keepalive:.0f} s - "
                              "releasing the camera", flush=True)
                        break

                    t0 = time.perf_counter()
                    ok, frame = camera.read()
                    capture_ms = (time.perf_counter() - t0) * 1000.0
                    stats["capture_sum"] += capture_ms
                    stats["capture_count"] += 1
                    if not ok:
                        stats["capture_errors"] += 1
                        self._set_error(
                            READ_FAILED_MESSAGE.format(index=CAMERA_INDEX))
                        break

                    # Mirror exactly once, BEFORE detection (Phase 2.1 rule -
                    # identical to training and to scripts/predict_realtime).
                    t_flip = time.perf_counter()
                    frame = cv2.flip(frame, 1)
                    flip_ms = (time.perf_counter() - t_flip) * 1000.0
                    stats["flip_sum"] += flip_ms

                    t_detect = time.perf_counter()
                    results = detector.detect(frame)
                    detect_ms = (time.perf_counter() - t_detect) * 1000.0
                    stats["detect_sum"] += detect_ms
                    stats["detect_count"] += 1
                    # Sub-split of detect(): BGR->RGB vs MediaPipe inference
                    # (reported by HandDetector.last_ms; a test stub without
                    # it counts the whole detect time as inference).
                    stage_split = getattr(detector, "last_ms", None) or {}
                    color_ms = float(stage_split.get("color", 0.0) or 0.0)
                    infer_ms = float(stage_split.get("infer", detect_ms)
                                     or detect_ms)
                    stats["color_sum"] += color_ms
                    stats["infer_sum"] += infer_ms
                    stats["infer_count"] += 1

                    t_draw = time.perf_counter()
                    hands = parse_hands(results)
                    detector.draw_landmarks(frame, hands)
                    draw_ms = (time.perf_counter() - t_draw) * 1000.0
                    stats["draw_sum"] += draw_ms

                    prediction = NO_HAND_LABEL
                    confidence = None
                    raw_text = ""
                    note = ""
                    latency = None
                    features_ms = None
                    model_ms = None

                    if not hands:
                        # Phase 5 rule: never predict from an all-zero
                        # vector; clear stale votes across an absence.
                        vote.clear()
                    else:
                        t_pred = time.perf_counter()
                        try:
                            t_feat = time.perf_counter()
                            features = build_sample(hands)   # Phase 3 pipeline
                            features_ms = (time.perf_counter() - t_feat) * 1000.0
                            t_model = time.perf_counter()
                            raw_label, raw_confidence, proba = \
                                predict_gesture(model, features, class_names)
                            model_ms = (time.perf_counter() - t_model) * 1000.0
                        except ValueError as error:
                            prediction = "Sample rejected"
                            note = str(error)
                        else:
                            stats["features_sum"] += features_ms
                            stats["features_count"] += 1
                            stats["model_sum"] += model_ms
                            stats["model_count"] += 1
                            latency = (time.perf_counter() - t_pred) * 1000.0
                            vote.push(raw_label)
                            stable = vote.result()
                            raw_text = f"{raw_label} ({raw_confidence * 100:.1f}%)"
                            if raw_confidence < MIN_PREDICTION_CONFIDENCE:
                                # Documented threshold (src/config.py)
                                prediction = "Uncertain"
                                confidence = raw_confidence
                                note = (f"below threshold "
                                        f"{MIN_PREDICTION_CONFIDENCE:.2f}")
                            else:
                                prediction = stable or raw_label
                                confidence = proba.get(prediction,
                                                       raw_confidence)

                    # Phase 6: exactly one observe() per frame, edge-
                    # triggered (held gestures fire once; re-arm rules in
                    # src/sentence_builder.py, unchanged).
                    t_smooth = time.perf_counter()
                    with self.lock:
                        action = self.builder.observe(prediction)
                        sentence = self.builder.sentence
                    smooth_ms = (time.perf_counter() - t_smooth) * 1000.0
                    stats["smooth_sum"] += smooth_ms
                    if action:
                        self.last_action = action
                        print(f"[feed] action: {action} -> \"{sentence}\"",
                              flush=True)

                    # --- measured timing (real, never simulated) -----------
                    now = time.monotonic()
                    frame_ms = (now - t0) * 1000.0
                    if previous is not None:
                        instantaneous = 1.0 / max(now - previous, 1e-6)
                        fps_ema = (instantaneous if fps_ema is None
                                   else 0.9 * fps_ema + 0.1 * instantaneous)
                    previous = now
                    stats["frames"] += 1
                    stats["frame_ms_sum"] += frame_ms
                    stats["frame_ms_max"] = max(stats["frame_ms_max"],
                                                 frame_ms)
                    if latency is not None:
                        stats["latency_count"] += 1
                        stats["latency_sum"] += latency
                        stats["latency_max"] = max(stats["latency_max"],
                                                    latency)
                        latency_ema = (latency if latency_ema is None
                                       else 0.9 * latency_ema + 0.1 * latency)
                    second_count += 1

                    # Per-stage EMA powering the once-per-second log line.
                    stage_ema["capture"] = self._ema(stage_ema.get("capture"),
                                                     capture_ms)
                    stage_ema["flip"] = self._ema(stage_ema.get("flip"),
                                                  flip_ms)
                    stage_ema["color"] = self._ema(stage_ema.get("color"),
                                                   color_ms)
                    stage_ema["infer"] = self._ema(stage_ema.get("infer"),
                                                   infer_ms)
                    stage_ema["draw"] = self._ema(stage_ema.get("draw"),
                                                  draw_ms)
                    stage_ema["smooth"] = self._ema(stage_ema.get("smooth"),
                                                    smooth_ms)
                    if features_ms is not None:
                        stage_ema["features"] = self._ema(
                            stage_ema.get("features"), features_ms)
                    if model_ms is not None:
                        stage_ema["model"] = self._ema(
                            stage_ema.get("model"), model_ms)

                    # --- preview: annotate, JPEG, publish ------------------
                    self._overlay(frame, prediction, confidence, sentence,
                                  fps_ema or 0.0, latency_ema)
                    t_encode = time.perf_counter()
                    encoded, buffer = cv2.imencode(
                        ".jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    encode_ms = (time.perf_counter() - t_encode) * 1000.0
                    stats["encode_sum"] += encode_ms
                    stats["encode_count"] += 1
                    stage_ema["encode"] = self._ema(stage_ema.get("encode"),
                                                    encode_ms)
                    push_ms = 0.0
                    if encoded:
                        # Push the encoder buffer itself (bytes-like) -
                        # avoids a ~45 KB copy per frame; the feed server
                        # writes it straight to the socket.
                        t_push = time.perf_counter()
                        mjpeg_server.push(buffer)
                        push_ms = (time.perf_counter() - t_push) * 1000.0
                        stats["push_sum"] += push_ms
                        stats["push_count"] += 1
                        stats["sent"] += 1
                    stage_ema["push"] = self._ema(stage_ema.get("push"),
                                                  push_ms)

                    # --- page-facing snapshot (read ~2-4 times/second) -----
                    self.snapshot = {
                        "running": True, "prediction": prediction,
                        "confidence": confidence, "raw_text": raw_text,
                        "note": note, "hands": len(hands),
                        "fps": fps_ema, "latency": latency_ema,
                        "frames": stats["frames"], "error": self.error,
                    }

                    if now - second_mark >= 1.0:
                        self._log_second(now - second_mark, second_count,
                                         stats, frame_ms, stage_ema,
                                         len(hands), sentence)
                        second_count = 0
                        second_mark = now

            except Exception as error:  # noqa: BLE001 - never die silently
                self._set_error(f"Feed stopped: {error}")

        finally:
            if camera is not None:
                camera.release()
            if owned is not None:
                owned.release()
                print("[feed] camera lock released", flush=True)
            duration = time.monotonic() - started
            self.running = False
            snapshot = dict(self.snapshot)
            snapshot.update(running=False, fps=None, latency=None)
            self.snapshot = snapshot
            if stats["frames"]:
                self._print_summary(duration, stats, camera_fps, sentence)
            else:
                print("[feed] worker stopped before any frame was processed",
                      flush=True)
