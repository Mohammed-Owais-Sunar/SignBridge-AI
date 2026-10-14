"""Browser camera path checks (src/browser_camera.py, src/camera_mode.py).

Run (project virtual environment, from the project folder):

    python scripts/test_browser_camera.py

WHAT IS CHECKED (all offline - no webcam, no browser, no network):
    A. Files, imports and dependencies (app.py wiring, requirements).
    B. Camera-mode selection: explicit server/browser/auto, unknown value,
       and that explicit browser mode NEVER calls the OpenCV probe.
    C. Processor lifecycle: defaults, start/stop transitions, vote-window
       reset on restart, manual sentence actions.
    D. Frame processing with stub detectors + a real Random Forest:
       no-hand frames, 126-feature input contract, confidence threshold,
       valid-class path, wide-frame downscale for inference only,
       unconfigured and crashing-pipeline error handling.
"""

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import browser_camera, camera_mode  # noqa: E402
from src.config import MIN_PREDICTION_CONFIDENCE, SIGN_CLASSES  # noqa: E402
from src.feature_extractor import MAX_FEATURE_LENGTH, build_sample  # noqa: E402
from src.sentence_builder import NO_HAND_LABEL  # noqa: E402
from scripts.test_feature_extraction import (make_hand,  # noqa: E402
                                             synthetic_landmarks)

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    mark = "PASS" if condition else "FAIL"
    suffix = f"  [{detail}]" if detail else ""
    print(f"  [{mark}] {name}{suffix}", flush=True)


def blank_frame(width=640, height=480):
    """One black bgr24 av.VideoFrame, like a browser would deliver."""
    return av.VideoFrame.from_ndarray(
        np.zeros((height, width, 3), dtype=np.uint8), format="bgr24")


# ---------------------------------------------------------------------------
# Stub detectors (MediaPipe-shaped results; the real parse_hands runs)
# ---------------------------------------------------------------------------
class NoHandDetector:
    last_ms = {"color": 0.05, "infer": 0.15}

    def detect(self, _frame):
        return SimpleNamespace(hand_landmarks=[], handedness=[])

    def draw_landmarks(self, _frame, _hands):
        return None


class WithHandDetector:
    """Records the width of the frame it was given (downscale check)."""

    last_ms = {"color": 0.05, "infer": 0.25}

    def __init__(self):
        self.seen_widths = []

    def detect(self, frame):
        self.seen_widths.append(frame.shape[1])
        landmarks = [SimpleNamespace(x=point[0], y=point[1], z=point[2])
                     for point in synthetic_landmarks()]
        return SimpleNamespace(
            hand_landmarks=[landmarks],
            handedness=[[SimpleNamespace(category_name="Right",
                                          score=0.99)]])

    def draw_landmarks(self, _frame, _hands):
        return None


class CrashingDetector:
    last_ms = {"color": 0.05, "infer": 0.25}

    def detect(self, _frame):
        raise RuntimeError("pipeline exploded")

    def draw_landmarks(self, _frame, _hands):
        return None


class FixedProbaModel:
    """predict_proba returns a fixed (1, 12) distribution.

    No scikit-learn internals on purpose: predict_gesture() then uses its
    plain-model fallback, exactly like it does for a foreign model.
    """

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=np.float64)

    def predict_proba(self, _X):
        return self._probs.reshape(1, -1)


def tiny_forest(seed=0):
    """A REAL small Random Forest over the 12 project classes (fast path)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(12 * 20, MAX_FEATURE_LENGTH))
    y = np.repeat(SIGN_CLASSES, 20)
    model = RandomForestClassifier(n_estimators=5, random_state=seed)
    model.fit(X, y)
    return model


def uniform_model():
    return FixedProbaModel([1.0 / len(SIGN_CLASSES)] * len(SIGN_CLASSES))


def confident_model(label):
    probs = [0.0] * len(SIGN_CLASSES)
    probs[SIGN_CLASSES.index(label)] = 1.0
    return FixedProbaModel(probs)


# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SIGNBRIDGE AI - BROWSER CAMERA PATH CHECKS")
    print("=" * 70)

    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(
        encoding="utf-8").lower()

    # --- A: files + dependencies ------------------------------------------
    print("\nA. Files and dependencies")
    check("A1. browser_camera.py and camera_mode.py exist",
          (PROJECT_ROOT / "src" / "browser_camera.py").exists()
          and (PROJECT_ROOT / "src" / "camera_mode.py").exists())
    for name in ("browser_camera.py", "camera_mode.py"):
        path = PROJECT_ROOT / "src" / name
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            ok, detail = True, ""
        except SyntaxError as error:
            ok, detail = False, str(error)
        check(f"A2. src/{name} compiles", ok, detail)
    check("A3. requirements.txt includes streamlit-webrtc",
          "streamlit-webrtc" in requirements)
    check("A4. app.py imports the browser path and guards the dependency",
          "from src import browser_camera, camera_mode" in source
          and "except ImportError" in source
          and "render_browser_camera" in source)
    check("A5. app.py keeps the two pipelines mutually exclusive",
          "if CAMERA_MODE == camera_mode.MODE_BROWSER:" in source
          and "return browser_camera.BrowserCameraProcessor()" in source
          and "return camera_worker.CameraWorker()" in source)

    # --- B: camera-mode selection -----------------------------------------
    print("\nB. Camera-mode selection")
    real_probe = camera_mode.probe_server_camera()
    check("B1. probe against the real index returns a boolean",
          isinstance(real_probe, bool), str(real_probe))
    check("B2. probe against a nonexistent index (99) fails cleanly",
          camera_mode.probe_server_camera(index=99) is False)
    check("B3. explicit server mode", camera_mode.resolve_camera_mode(
        {"SIGNBRIDGE_CAMERA_MODE": "server"}) == camera_mode.MODE_SERVER)
    check("B4. explicit browser mode", camera_mode.resolve_camera_mode(
        {"SIGNBRIDGE_CAMERA_MODE": "browser"}) == camera_mode.MODE_BROWSER)

    # Explicit modes must NEVER touch the OpenCV camera.
    calls = []
    original_probe = camera_mode.probe_server_camera
    camera_mode.probe_server_camera = \
        lambda *args, **kwargs: calls.append(1) or original_probe(*args, **kwargs)
    try:
        camera_mode.resolve_camera_mode({"SIGNBRIDGE_CAMERA_MODE": "browser"})
        camera_mode.resolve_camera_mode({"SIGNBRIDGE_CAMERA_MODE": "server"})
    finally:
        camera_mode.probe_server_camera = original_probe
    check("B5. explicit modes skip the OpenCV probe entirely", calls == [],
          f"{len(calls)} probe call(s)")

    expected_auto = (camera_mode.MODE_SERVER if real_probe
                     else camera_mode.MODE_BROWSER)
    check("B6. auto mode matches the probe outcome",
          camera_mode.resolve_camera_mode(
              {"SIGNBRIDGE_CAMERA_MODE": "auto"}) == expected_auto,
          f"probe={real_probe} -> {expected_auto}")
    check("B7. unknown mode value falls back to auto",
          camera_mode.resolve_camera_mode(
              {"SIGNBRIDGE_CAMERA_MODE": "bogus"}) == expected_auto)

    # --- C: processor lifecycle --------------------------------------------
    print("\nC. Processor lifecycle")
    processor = browser_camera.BrowserCameraProcessor()
    check("C1. fresh processor is idle with the default snapshot",
          processor.snapshot["running"] is False
          and processor.snapshot["prediction"] == NO_HAND_LABEL
          and processor.snapshot["frames"] == 0
          and processor.is_alive() is False and processor.is_stopping() is False)
    processor.set_running(True)
    check("C2. set_running(True) marks the session live",
          processor.is_alive() is True
          and processor.snapshot["running"] is True)
    processor.set_running(False)
    check("C3. set_running(False) stops live readouts",
          processor.snapshot["running"] is False
          and processor.snapshot["fps"] is None)

    # Restart resets the vote window: after 4 same-label votes, a stop and
    # start must clear them (observed through the window's own state -
    # result() alone cannot distinguish a full from a partial window).
    detector = WithHandDetector()
    processor.configure(tiny_forest(), list(SIGN_CLASSES), detector)
    processor.set_running(True)
    for _ in range(4):
        processor.recv(blank_frame())
    votes_before = len(processor._vote._frames)
    processor.set_running(False)
    processor.set_running(True)
    processor.recv(blank_frame())
    check("C4. restart resets the vote window",
          votes_before == 4 and len(processor._vote._frames) == 1,
          f"before={votes_before} after={len(processor._vote._frames)}")

    processor.builder.clear()   # C4's frames already produced words
    processor.manual("SPACE")
    check("C5. manual SPACE on an empty sentence is a documented no-op "
          "(no leading space)",
          processor.builder.sentence == ""
          and processor.last_action == "SPACE")
    processor.builder.add_word("HELLO")
    processor.manual("SPACE")
    check("C5b. manual SPACE appends exactly one separator",
          processor.builder.sentence == "HELLO "
          and processor.last_action == "SPACE")
    processor.manual("DELETE")
    check("C6. manual DELETE clears the word",
          processor.builder.sentence == ""
          and processor.last_action == "DELETE")
    processor.manual("CLEAR")
    check("C7. manual CLEAR resets the message",
          processor.builder.sentence == "" and processor.last_action == "CLEAR")

    # --- D: frame processing (stubs + a real Random Forest) -----------------
    print("\nD. Frame processing")
    check("D1. fabricated hands still build exactly 126 finite features",
          len(build_sample([make_hand("Right")])) == MAX_FEATURE_LENGTH
          and all(np.isfinite(build_sample([make_hand("Right")]))))

    # No-hand frame.
    processor = browser_camera.BrowserCameraProcessor()
    processor.configure(tiny_forest(), list(SIGN_CLASSES), NoHandDetector())
    out = processor.recv(blank_frame())
    check("D2. no-hand frame: valid av frame back, no prediction",
          isinstance(out, av.VideoFrame)
          and processor.snapshot["prediction"] == NO_HAND_LABEL
          and processor.snapshot["hands"] == 0
          and processor.snapshot["frames"] == 1)

    # Hand frame with a confident model: the full pipeline runs.
    processor = browser_camera.BrowserCameraProcessor()
    detector = WithHandDetector()
    processor.configure(confident_model("HELLO"), list(SIGN_CLASSES),
                        detector)
    out = processor.recv(blank_frame())
    snap = processor.snapshot
    check("D3. hand frame runs the pipeline and returns an annotated frame",
          isinstance(out, av.VideoFrame)
          and out.to_ndarray(format="bgr24").shape == (480, 640, 3)
          and snap["hands"] == 1 and snap["frames"] == 1)
    check("D4. confident prediction shows the class, not 'Uncertain'",
          snap["prediction"] == "HELLO"
          and snap["confidence"] == 1.0
          and snap["raw_text"].startswith("HELLO"),
          f"{snap['prediction']} @ {snap['confidence']}")
    check("D5. measured latency is published",
          snap["latency"] is not None)
    processor.recv(blank_frame())   # a second frame defines an interval
    check("D5b. FPS is published once two frames arrived",
          processor.snapshot["fps"] is not None)

    # Below-threshold confidence shows 'Uncertain' with the reason.
    processor = browser_camera.BrowserCameraProcessor()
    processor.configure(uniform_model(), list(SIGN_CLASSES),
                        NoHandDetector())
    processor.recv(blank_frame())  # hands=0 -> no prediction path
    processor.configure(uniform_model(), list(SIGN_CLASSES),
                        WithHandDetector())
    processor.recv(blank_frame())
    snap = processor.snapshot
    check("D6. below-threshold confidence shows 'Uncertain' + the reason",
          snap["prediction"] == "Uncertain"
          and abs(snap["confidence"] - 1.0 / 12.0) < 1e-9
          and f"{MIN_PREDICTION_CONFIDENCE:.2f}" in snap["note"],
          f"{snap['prediction']} @ {snap['confidence']:.4f}")

    # Wide frames are downscaled for INFERENCE ONLY; the preview keeps its
    # native size.
    processor = browser_camera.BrowserCameraProcessor()
    detector = WithHandDetector()
    processor.configure(tiny_forest(), list(SIGN_CLASSES), detector)
    out = processor.recv(blank_frame(width=1280, height=720))
    check("D7. wide frame: inference downscaled, preview stays full-size",
          max(detector.seen_widths) <= browser_camera.MAX_INFERENCE_WIDTH
          and out.to_ndarray(format="bgr24").shape == (720, 1280, 3),
          f"seen widths {detector.seen_widths}")

    # Unconfigured processor: error captured, unannotated frame returned.
    processor = browser_camera.BrowserCameraProcessor()
    frame = blank_frame()
    out = processor.recv(frame)
    check("D8. unconfigured processor reports the error and survives",
          processor.error.startswith("Recognition error")
          and processor.snapshot["error"] == processor.error
          and isinstance(out, av.VideoFrame)
          and np.array_equal(out.to_ndarray(format="bgr24"),
                             frame.to_ndarray(format="bgr24")))

    # Pipeline crash mid-frame: captured into the snapshot, session lives.
    processor = browser_camera.BrowserCameraProcessor()
    processor.configure(tiny_forest(), list(SIGN_CLASSES), CrashingDetector())
    out = processor.recv(blank_frame())
    check("D9. crashing pipeline is captured, not raised",
          processor.error == "Recognition error: pipeline exploded"
          and isinstance(out, av.VideoFrame))
    processor.configure(tiny_forest(), list(SIGN_CLASSES), NoHandDetector())
    processor.recv(blank_frame())
    check("D10. a good frame clears the previous error",
          processor.error is None and processor.snapshot["error"] is None)

    # The two camera sources can never be active together: the browser
    # processor holds no device and takes no lock. AST-based, so docstring
    # mentions of cv2.VideoCapture cannot false-positive.
    import ast
    browser_tree = ast.parse(
        (PROJECT_ROOT / "src" / "browser_camera.py").read_text(
            encoding="utf-8"))
    opens_device = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "VideoCapture"
        for node in ast.walk(browser_tree))
    takes_lock = any(
        (isinstance(node, ast.ImportFrom)
         and (node.module or "").endswith("camera_lock"))
        or (isinstance(node, ast.Import)
            and any(alias.name.endswith("camera_lock")
                    for alias in node.names))
        for node in ast.walk(browser_tree))
    check("D11. browser processor opens no camera and acquires no lock",
          not opens_device and not takes_lock,
          f"VideoCapture call={opens_device} camera_lock import={takes_lock}")

    return _summary()


def _summary():
    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 70, flush=True)
    print(f"browser camera: {len(RESULTS) - len(failed)}/{len(RESULTS)} "
          f"checks passed", flush=True)
    for name in failed:
        print(f"  FAILED: {name}", flush=True)
    print("=" * 70, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
