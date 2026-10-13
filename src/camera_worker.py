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
  no page is attached for KEEPALIVE_SECONDS the camera is released
  automatically (browser tab closed without pressing Stop)
"""

import gc
import threading
import time

import cv2

from src import mjpeg_server
from src.config import (MIN_PREDICTION_CONFIDENCE,
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


class CameraWorker:
    """Runs the Phase 5/6 pipeline continuously and publishes results."""

    def __init__(self):
        # SentenceBuilder is owned by this worker (single feed = single
        # sentence); the lock serializes observe() and manual UI actions.
        self.lock = threading.RLock()
        self.builder = SentenceBuilder()
        self._vote = VoteWindow(PREDICTION_VOTE_WINDOW)
        self.last_action = None
        self.error = None
        self.running = False

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
                "capture_errors": 0, "sent": 0}

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def touch(self):
        """Called by the page (every run / fragment run): 'a page exists'."""
        self._activity = time.monotonic()

    def start(self, model, class_names, detector):
        """Spawn the feed thread (idempotent). Returns None always; any
        camera failure is reported asynchronously via error/snapshot."""
        with self._control:
            if self.is_alive():
                return None
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
        return None

    def stop(self, timeout=3.0):
        """Signal the thread, wait for it (it releases the camera)."""
        with self._control:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        self.running = False

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

    def _log_second(self, elapsed_in_second, count, stats, frame_ms,
                    latency_ema, detect_ema, encode_ema, hands, sentence):
        fps_now = count / max(elapsed_in_second, 1e-6)
        stats["fps_samples"].append(fps_now)
        samples = stats["fps_samples"]
        pred_avg = (stats["latency_sum"] / stats["latency_count"]
                    if stats["latency_count"] else 0.0)
        print(f"[feed] t={len(samples)}s fps={fps_now:.1f} "
              f"avg={sum(samples) / len(samples):.1f} "
              f"min={min(samples):.1f} max={max(samples):.1f} | "
              f"frame={frame_ms:.1f} ms "
              f"(detect {detect_ema or 0:.1f} / pred {pred_avg:.1f} / "
              f"enc {encode_ema or 0:.1f}) | "
              f"hands={hands} | \"{sentence}\"", flush=True)

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
        print(f"Sentence            : \"{sentence}\"", flush=True)
        print("No speech/audio was produced (TTS belongs to Phase 7).",
              flush=True)
        print("=" * 60, flush=True)

    # ------------------------------------------------------------ main loop
    def _run(self):
        camera = None
        camera_fps = 30.0
        stats = self._stats
        started = time.monotonic()
        sentence = ""
        try:
            try:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    camera.release()
                    camera = None
                    self._set_error(
                        "Could not open the webcam. Make sure a camera is "
                        "connected and not used by another application "
                        "(close the Phase 5 preview window if it is open).")
                    return

                reported = camera.get(cv2.CAP_PROP_FPS)
                if reported and 1 <= reported <= 120:
                    camera_fps = float(reported)
                for _ in range(WARMUP_FRAMES):
                    camera.grab()     # drop warm-up frames (Phase 5 behaviour)
                # Freeze the already-loaded static graph (MediaPipe model,
                # Random Forest, module caches) into the permanent GC
                # generation: generational collections then stop scanning it
                # every few seconds, which removes the periodic ~100 ms
                # frame-time spikes it caused in the live browser session.
                gc.freeze()
                print(f"[feed] camera opened, expecting ~{camera_fps:.0f} FPS",
                      flush=True)

                vote = self._vote
                model, class_names = self._model, self._class_names
                detector = self._detector

                previous = None
                fps_ema = None
                latency_ema = None
                detect_ema = None
                encode_ema = None
                second_mark = time.monotonic()
                second_count = 0

                while not self._stop.is_set():
                    if time.monotonic() - self._activity > KEEPALIVE_SECONDS:
                        print("[feed] no browser page attached for "
                              f"{KEEPALIVE_SECONDS:.0f} s - releasing the "
                              "camera", flush=True)
                        break

                    t0 = time.perf_counter()
                    ok, frame = camera.read()
                    if not ok:
                        stats["capture_errors"] += 1
                        self._set_error("Failed to read a frame from the "
                                        "webcam.")
                        break

                    # Mirror exactly once, BEFORE detection (Phase 2.1 rule -
                    # identical to training and to scripts/predict_realtime).
                    frame = cv2.flip(frame, 1)
                    t_detect = time.perf_counter()
                    results = detector.detect(frame)
                    detect_ms = (time.perf_counter() - t_detect) * 1000.0
                    stats["detect_sum"] += detect_ms
                    stats["detect_count"] += 1
                    detect_ema = (detect_ms if detect_ema is None
                                  else 0.9 * detect_ema + 0.1 * detect_ms)
                    hands = parse_hands(results)
                    detector.draw_landmarks(frame, hands)

                    prediction = NO_HAND_LABEL
                    confidence = None
                    raw_text = ""
                    note = ""
                    latency = None

                    if not hands:
                        # Phase 5 rule: never predict from an all-zero
                        # vector; clear stale votes across an absence.
                        vote.clear()
                    else:
                        t_pred = time.perf_counter()
                        try:
                            features = build_sample(hands)   # Phase 3 pipeline
                            raw_label, raw_confidence, proba = \
                                predict_gesture(model, features, class_names)
                        except ValueError as error:
                            prediction = "Sample rejected"
                            note = str(error)
                        else:
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
                    with self.lock:
                        action = self.builder.observe(prediction)
                        sentence = self.builder.sentence
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
                    encode_ema = (encode_ms if encode_ema is None
                                  else 0.9 * encode_ema + 0.1 * encode_ms)
                    if encoded:
                        # Push the encoder buffer itself (bytes-like) -
                        # avoids a ~45 KB copy per frame; the feed server
                        # writes it straight to the socket.
                        mjpeg_server.push(buffer)
                        stats["sent"] += 1

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
                                         stats, frame_ms, latency_ema,
                                         detect_ema, encode_ema,
                                         len(hands), sentence)
                        second_count = 0
                        second_mark = now

            except Exception as error:  # noqa: BLE001 - never die silently
                self._set_error(f"Feed stopped: {error}")

        finally:
            if camera is not None:
                camera.release()
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
