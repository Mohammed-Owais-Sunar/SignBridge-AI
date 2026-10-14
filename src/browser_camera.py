"""Browser webcam recognition processor for Streamlit Community Cloud.

WHY THIS EXISTS
    Streamlit Community Cloud runs on a remote Linux server with no webcam,
    so the server-side cv2.VideoCapture(0) pipeline (src/camera_worker.py)
    cannot work there. This processor receives frames from the VISITOR'S
    browser through streamlit-webrtc: the browser captures its local
    webcam, sends every frame to the app server, and displays the
    annotated frame the processor sends back (WebRtcMode.SENDRECV). The
    server never opens a camera device in this mode - no OpenCV probe, no
    camera lock, nothing to leak, and no Windows-style camera errors.

PIPELINE (every step imported from the existing modules - nothing is
re-implemented, skipped or tuned; model/dataset/threshold/vote window
are exactly the local pipeline's):

    browser frame (av.VideoFrame, bgr24)
      -> mirror exactly once, BEFORE detection (the training/Phase-2.1
         contract, identical to scripts/predict_realtime)
      -> optional downscale to <= 640 px wide for inference only
         (landmark coordinates are normalized, so they still map onto the
         full-size preview; build_sample normalizes per frame)
      -> HandDetector.detect() -> parse_hands() -> build_sample()
      -> predict_gesture() (Random Forest, validate-once fast path)
      -> VoteWindow (PREDICTION_VOTE_WINDOW) -> confidence threshold
         (MIN_PREDICTION_CONFIDENCE)
      -> SentenceBuilder.observe() (edge trigger + cooldown, unchanged)
      -> annotate -> from_ndarray() back to the browser

THREADING CONTRACT
    recv() runs in streamlit-webrtc's worker thread; page fragments read
    self.snapshot (wholesale dict replacement - atomic in CPython) and
    call manual(), which is serialized with observe() through self.lock,
    so Phase 6 edge-trigger/cooldown semantics stay exact. There is no
    background thread of this class and no camera handle - session end,
    tab close or disconnect needs no cleanup beyond what the streamer
    already does with the peer connection.

START/STOP CONTRACT
    The page fragment syncs self.running with the streamer state via
    set_running(ctx.state.playing) on every tick. A stopped->running
    transition resets the vote window, error and frame counter, so a
    restarted session never inherits stale state; a running->stopped
    transition only clears the live-rate readouts.
"""

import threading
import time

import cv2

from src.config import MIN_PREDICTION_CONFIDENCE, PREDICTION_VOTE_WINDOW
from src.feature_extractor import build_sample
from src.hand_detector import parse_hands
from src.sentence_builder import NO_HAND_LABEL, SentenceBuilder
from scripts.predict_realtime import (COLOR_CYAN, COLOR_GREEN, COLOR_RED,
                                      COLOR_WHITE, COLOR_YELLOW,
                                      VoteWindow, predict_gesture, put_text)

# Frames wider than this are downscaled for INFERENCE ONLY; landmarks are
# normalized so they still map onto the full-size annotated preview.
MAX_INFERENCE_WIDTH = 640
# Bounded instrumentation: one line per this many frames, never per frame.
LOG_EVERY_FRAMES = 30


class BrowserCameraProcessor:
    """Thread-safe state holder and streamlit-webrtc frame callback."""

    def __init__(self):
        self.lock = threading.RLock()
        self.builder = SentenceBuilder()
        self._vote = VoteWindow(PREDICTION_VOTE_WINDOW)
        self._model = None
        self._class_names = None
        self._detector = None
        self.running = False
        self.error = None
        self.last_action = None
        self.snapshot = {
            "running": False, "prediction": NO_HAND_LABEL,
            "confidence": None, "raw_text": "", "note": "",
            "hands": 0, "fps": None, "latency": None,
            "frames": 0, "error": None,
        }
        self._last_frame_at = None
        self._fps_ema = None
        self._frames = 0

    # ----------------------------------------------------------- lifecycle
    def configure(self, model, class_names, detector):
        """Attach the cached model/detector (process resources; assigning
        these references is cheap and does not rebuild the inference
        pipeline). Called by the page on every tick - it must stay cheap
        and must NOT reset recognition state (see set_running)."""
        self._model = model
        self._class_names = class_names
        self._detector = detector

    def touch(self):
        """Kept for compatibility with the page fragments (no-op: the
        browser pipeline has no camera hold to keep alive)."""

    def is_alive(self):
        return self.running

    def is_stopping(self):
        """False: the streamer owns the connection lifecycle, so there is
        no background thread winding down (kept for worker_state())."""
        return False

    def set_running(self, value):
        """Sync with the streamer state (called by the page every tick).

        A stopped->running transition resets the vote window, error and
        frame counters so a restarted session never inherits the previous
        one's smoothing state.
        """
        value = bool(value)
        if value and not self.running:
            self._vote.clear()
            self.error = None
            self._frames = 0
            self._fps_ema = None
            self._last_frame_at = None
        self.running = value
        snapshot = dict(self.snapshot)
        snapshot["running"] = value
        if not value:
            snapshot["fps"] = None
        self.snapshot = snapshot

    # --------------------------------------------------------- manual input
    def manual(self, operation):
        """Apply a deliberate UI action (SPACE / DELETE / CLEAR).

        Serialized with observe() through the same lock, exactly like the
        CameraWorker: bypasses the edge trigger and the cooldown (those
        bound GESTURE repeats, not deliberate clicks).
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
        print(f"[browser-cam] manual: {self.last_action} -> \"{sentence}\"",
              flush=True)

    # ------------------------------------------------------------ internals
    def _set_error(self, message):
        self.error = message
        snapshot = dict(self.snapshot)
        snapshot.update(running=self.running, error=message)
        self.snapshot = snapshot
        print(f"[browser-cam] ERROR: {message}", flush=True)

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

    def recv(self, video_frame):
        """Process one browser-provided frame; return it annotated.

        streamlit-webrtc calls this (SENDRECV mode) for every frame the
        visitor's camera delivers. The frame is mirrored once, run through
        the unchanged Phase 5/6 pipeline and sent back for display. Any
        per-frame failure is captured into the snapshot (the live session
        keeps running); the unannotated frame is returned so the preview
        never freezes on a recognition error.
        """
        frame = video_frame.to_ndarray(format="bgr24")
        try:
            return self._process(video_frame, frame)
        except Exception as error:  # noqa: BLE001 - never kill the stream
            self._set_error(f"Recognition error: {error}")
            return video_frame.from_ndarray(frame, format="bgr24")

    def _process(self, video_frame, frame):
        # Mirror exactly once, BEFORE detection (Phase 2.1 rule - identical
        # to training and to scripts/predict_realtime).
        frame = cv2.flip(frame, 1)

        if self._model is None or self._detector is None:
            raise RuntimeError("Recognition model is not initialized.")

        # Downscale wide frames for inference only (see MAX_INFERENCE_WIDTH).
        inference_frame = frame
        if frame.shape[1] > MAX_INFERENCE_WIDTH:
            scale = MAX_INFERENCE_WIDTH / frame.shape[1]
            inference_frame = cv2.resize(
                frame,
                (MAX_INFERENCE_WIDTH,
                 max(1, int(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        detect_start = time.perf_counter()
        results = self._detector.detect(inference_frame)
        detect_ms = (time.perf_counter() - detect_start) * 1000.0
        if self._frames % LOG_EVERY_FRAMES == 0:
            print(f"[browser-cam] frame {self._frames}: "
                  f"detect {detect_ms:.1f} ms", flush=True)

        hands = parse_hands(results)
        self._detector.draw_landmarks(frame, hands)

        prediction = NO_HAND_LABEL
        confidence = None
        raw_text = ""
        note = ""
        latency = None

        if not hands:
            # Phase 5 rule: never predict from an all-zero vector; clear
            # stale votes across an absence.
            self._vote.clear()
        else:
            # build_sample + predict_gesture, timed together like the
            # worker's prediction-latency measurement.
            t_pred = time.perf_counter()
            try:
                features = build_sample(hands)        # Phase 3 pipeline
                raw_label, raw_confidence, probabilities = predict_gesture(
                    self._model, features, self._class_names)
            except ValueError as error:
                prediction = "Sample rejected"
                note = str(error)
            else:
                latency = (time.perf_counter() - t_pred) * 1000.0
                self._vote.push(raw_label)
                stable = self._vote.result()
                raw_text = f"{raw_label} ({raw_confidence * 100:.1f}%)"
                if raw_confidence < MIN_PREDICTION_CONFIDENCE:
                    # Documented threshold (src/config.py), same as local.
                    prediction = "Uncertain"
                    confidence = raw_confidence
                    note = (f"below threshold "
                            f"{MIN_PREDICTION_CONFIDENCE:.2f}")
                else:
                    prediction = stable or raw_label
                    confidence = probabilities.get(prediction,
                                                    raw_confidence)

        # Phase 6: exactly one observe() per frame, edge-triggered (held
        # gestures fire once; re-arm rules in src/sentence_builder.py,
        # unchanged).
        with self.lock:
            action = self.builder.observe(prediction)
            sentence = self.builder.sentence
            if action:
                self.last_action = action
                print(f"[browser-cam] action: {action} -> \"{sentence}\"",
                      flush=True)

        # Arrival-rate FPS EMA (frames per second the browser delivers).
        now = time.perf_counter()
        if self._last_frame_at is not None:
            instant_fps = 1.0 / max(now - self._last_frame_at, 1e-6)
            self._fps_ema = (instant_fps if self._fps_ema is None
                             else 0.9 * self._fps_ema + 0.1 * instant_fps)
        self._last_frame_at = now
        self._frames += 1

        self._overlay(frame, prediction, confidence, sentence,
                      self._fps_ema or 0.0, latency)

        # A successful frame clears any previous error BEFORE the
        # snapshot is published, so the page never shows a stale one.
        self.error = None
        self.snapshot = {
            "running": True, "prediction": prediction,
            "confidence": confidence, "raw_text": raw_text,
            "note": note, "hands": len(hands),
            "fps": self._fps_ema, "latency": latency,
            "frames": self._frames, "error": None,
        }

        return video_frame.from_ndarray(frame, format="bgr24")
