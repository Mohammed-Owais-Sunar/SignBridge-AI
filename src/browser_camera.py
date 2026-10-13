"""Browser webcam recognition processor for Streamlit Community Cloud.

Frames arrive from the visitor's browser via streamlit-webrtc; the server
never tries to open a local webcam with cv2.VideoCapture(). The existing
MediaPipe, feature extraction, Random Forest, vote-window and SentenceBuilder
pipeline is reused unchanged.
"""
import threading
import time

import cv2

from src.config import MIN_PREDICTION_CONFIDENCE, PREDICTION_VOTE_WINDOW
from src.feature_extractor import build_sample
from src.hand_detector import parse_hands
from src.sentence_builder import NO_HAND_LABEL, SentenceBuilder
from scripts.predict_realtime import (
    COLOR_CYAN, COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_YELLOW,
    VoteWindow, predict_gesture, put_text,
)


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

    def configure(self, model, class_names, detector):
        # The cached model/detector are process resources; assigning these
        # references is cheap and does not rebuild the inference pipeline.
        self._model, self._class_names, self._detector = model, class_names, detector

    def touch(self):
        # Kept for compatibility with the existing page fragments.
        pass

    def is_alive(self):
        return self.running

    def set_running(self, value):
        self.running = bool(value)
        snap = dict(self.snapshot)
        snap["running"] = self.running
        if not self.running:
            snap["fps"] = None
        self.snapshot = snap

    def manual(self, operation):
        with self.lock:
            if operation == "SPACE":
                self.builder.insert_space()
                self.last_action = "SPACE"
            elif operation == "DELETE":
                self.last_action = self.builder.delete_word()
            else:
                self.builder.clear()
                self.last_action = "CLEAR"

    @staticmethod
    def _overlay(frame, prediction, confidence, sentence, fps, latency):
        h = frame.shape[0]
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
                 (10, h - 44), COLOR_CYAN, 0.7, 2)
        line = f"{fps:.1f} FPS"
        if latency is not None:
            line += f"   pred {latency:.1f} ms"
        put_text(frame, line, (10, h - 14), COLOR_WHITE, 0.55, 1)

    def recv(self, video_frame):
        """Process one browser-provided frame and return annotated video."""
        frame = video_frame.to_ndarray(format="bgr24")
        # Match the project's training/inference contract: mirror once before
        # MediaPipe detection, feature extraction and preview annotation.
        frame = cv2.flip(frame, 1)
        t0 = time.perf_counter()
        prediction, confidence, raw_text, note = NO_HAND_LABEL, None, "", ""
        latency = None
        hands = []

        try:
            if self._model is None or self._detector is None:
                raise RuntimeError("Recognition model is not initialized.")
            # Run MediaPipe on a smaller inference copy to reduce per-frame
            # detection cost. Landmark coordinates are normalized, so they can
            # still be drawn on the full-resolution preview frame.
            inference_frame = frame
            if frame.shape[1] > 640:
                scale = 640.0 / frame.shape[1]
                inference_frame = cv2.resize(
                    frame,
                    (640, max(1, int(frame.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            results = self._detector.detect(inference_frame)
            hands = parse_hands(results)
            self._detector.draw_landmarks(frame, hands)

            if not hands:
                self._vote.clear()
            else:
                pred_start = time.perf_counter()
                try:
                    features = build_sample(hands)
                    raw_label, raw_confidence, probabilities = predict_gesture(
                        self._model, features, self._class_names)
                except ValueError as exc:
                    prediction, note = "Sample rejected", str(exc)
                else:
                    latency = (time.perf_counter() - pred_start) * 1000.0
                    self._vote.push(raw_label)
                    stable = self._vote.result()
                    raw_text = f"{raw_label} ({raw_confidence * 100:.1f}%)"
                    if raw_confidence < MIN_PREDICTION_CONFIDENCE:
                        prediction, confidence = "Uncertain", raw_confidence
                        note = f"below threshold {MIN_PREDICTION_CONFIDENCE:.2f}"
                    else:
                        prediction = stable or raw_label
                        confidence = probabilities.get(prediction, raw_confidence)

            with self.lock:
                action = self.builder.observe(prediction)
                sentence = self.builder.sentence
                if action:
                    self.last_action = action

            now = time.perf_counter()
            if self._last_frame_at is not None:
                instant_fps = 1.0 / max(now - self._last_frame_at, 1e-6)
                self._fps_ema = (instant_fps if self._fps_ema is None else
                                 0.9 * self._fps_ema + 0.1 * instant_fps)
            self._last_frame_at = now
            self._frames += 1
            self._overlay(frame, prediction, confidence, sentence,
                          self._fps_ema or 0.0, latency)
            self.snapshot = {
                "running": True, "prediction": prediction,
                "confidence": confidence, "raw_text": raw_text,
                "note": note, "hands": len(hands),
                "fps": self._fps_ema, "latency": latency,
                "frames": self._frames, "error": None,
            }
            self.error = None
        except Exception as exc:
            self.error = f"Recognition error: {exc}"
            snap = dict(self.snapshot)
            snap.update(running=True, error=self.error, note=self.error)
            self.snapshot = snap

        return video_frame.from_ndarray(frame, format="bgr24")
