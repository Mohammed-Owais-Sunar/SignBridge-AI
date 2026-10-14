"""Camera ownership, worker lifecycle and fast-prediction test (repair branch).

Runs WITHOUT a physical webcam: every camera is a mock injected through the
worker's camera_factory seam, and the cross-process lock tests use camera
indices 95/96/99 - never the production index (src/config.CAMERA_INDEX = 0
is patched to a fake index for the worker tests and stays untouched on disk).

WHAT IS VERIFIED
    A. syntax/imports of every module changed on fix/camera-fps-stability;
       production CAMERA_INDEX/CAMERA_BACKEND unchanged
    B. trained model loads unchanged: 12 classes, 126 features, 0.50
       threshold semantics - predict_gesture round-trip on a pipeline vector
    C. fast predict path (predict_proba_single) is BIT-IDENTICAL to
       model.predict_proba on 50 random + 1 pipeline vector, with the plain
       predict_proba fallback exercised for a foreign model object
    D. camera OPEN failure -> clear error naming the index, thread
       terminates, camera released, lock free
    E. opened-but-NO-frames (the Windows "camera already in use" symptom:
       isOpened() True while reads/grabs fail) -> clear error
    F. mid-loop read failure -> clear error, thread exits, release in
       finally
    G. duplicate start prevention: second start is idempotent, one thread
    H. repeated start/stop (3 cycles): every camera released, lock free,
       stop() idempotent
    I. stop() does NOT block the UI even with a backend slow to release:
       returns fast, worker reports is_stopping(), start is refused with a
       message while winding down, thread later winds down on its own
    J. keepalive: auto-release when no page touch AND no MJPEG feed client;
       NOT released while a feed client is attached (backgrounded tab)
    K. cross-process camera lock: another process gets CameraBusyError
       naming the holder PID; same-process double-lock rejected; a stale
       lock file never blocks (OS releases the lock with the process)
    L. end-to-end inside the worker thread: fake hand -> parse_hands ->
       build_sample -> REAL model prediction -> vote -> sentence observe,
       with the per-stage instrumentation counters all populated

"Streamlit startup" and the Start/Stop button controls are verified by
scripts/test_web_ui.py (AppTest) - run BOTH scripts:

Usage (from the project folder):
    python scripts/test_camera_lifecycle.py
    python scripts/test_web_ui.py
"""

import os
import py_compile
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

# Make the project importable when run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2                      # noqa: E402 - project import order
import numpy as np              # noqa: E402

import src.config as config     # noqa: E402
from src import camera_lock, mjpeg_server  # noqa: E402
from src import camera_worker as cw         # noqa: E402
from src.config import (MIN_PREDICTION_CONFIDENCE, SIGN_CLASSES)  # noqa: E402
from src.feature_extractor import MAX_FEATURE_LENGTH, build_sample  # noqa: E402
from scripts.predict_realtime import (load_classifier, predict_gesture,  # noqa: E402
                                      predict_proba_single)
from scripts.test_feature_extraction import make_hand, synthetic_landmarks  # noqa: E402

# Worker lifecycle tests lock a FAKE camera index so a running app (or the
# real webcam) is never involved; 95/96/99 are used by the lock tests.
TEST_INDEX = 97

# ---------------------------------------------------------------------------
# Tiny test harness (same style as the other project test scripts)
# ---------------------------------------------------------------------------
RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    mark = "PASS" if condition else "FAIL"
    suffix = f"  [{detail}]" if detail else ""
    print(f"  [{mark}] {name}{suffix}", flush=True)


def wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _summary():
    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 70, flush=True)
    print(f"camera lifecycle: {len(RESULTS) - len(failed)}/{len(RESULTS)} "
          f"checks passed", flush=True)
    for name in failed:
        print(f"  FAILED: {name}", flush=True)
    print("=" * 70, flush=True)
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Mock camera + mock detector (the physical webcam is never opened)
# ---------------------------------------------------------------------------
class FakeCamera:
    """OpenCV VideoCapture stand-in with configurable failure modes."""

    def __init__(self, index, *args, opened=True, grab_ok=True,
                 read_ok=True, read_delay=0.005, release_delay=0.0):
        self.index = index
        self.opened = opened
        self.grab_ok = grab_ok
        self.read_ok = read_ok
        self.read_delay = read_delay
        self.release_delay = release_delay
        self.released = False
        self.frames_read = 0
        self._frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def isOpened(self):
        return self.opened and not self.released

    def get(self, _prop):
        return 30.0

    def grab(self):
        time.sleep(0.001)
        return self.grab_ok and not self.released

    def read(self):
        if self.read_delay:
            time.sleep(self.read_delay)
        if not self.read_ok:
            return False, None
        self.frames_read += 1
        return True, self._frame.copy()

    def release(self):
        if self.release_delay:
            time.sleep(self.release_delay)
        self.released = True
        self.opened = False


class NoHandDetector:
    """Detector stub: valid MediaPipe-shaped result with zero hands."""

    last_ms = {"color": 0.05, "infer": 0.15}

    def detect(self, _frame):
        return SimpleNamespace(hand_landmarks=[], handedness=[])

    def draw_landmarks(self, _frame, _hands):
        return None


class WithHandDetector:
    """Detector stub: one synthetic 'Right' hand every frame.

    The landmark objects mimic MediaPipe's output (.x/.y/.z) so the REAL
    parse_hands -> build_sample -> predict_gesture path runs inside the
    worker thread against the REAL trained model.
    """

    last_ms = {"color": 0.05, "infer": 0.25}

    def detect(self, _frame):
        landmarks = [SimpleNamespace(x=point[0], y=point[1], z=point[2])
                     for point in synthetic_landmarks()]
        return SimpleNamespace(
            hand_landmarks=[landmarks],
            handedness=[[SimpleNamespace(category_name="Right",
                                         score=0.99)]])

    def draw_landmarks(self, _frame, _hands):
        return None


def make_worker(keepalive=None, **camera_kwargs):
    """CameraWorker with an injected fake camera; returns (worker, cameras)."""
    created = []

    def factory(index, *args):
        camera = FakeCamera(index, *args, **camera_kwargs)
        created.append(camera)
        return camera

    worker = cw.CameraWorker(camera_factory=factory,
                             keepalive_seconds=keepalive)
    return worker, created


def start_worker(worker, model=None, class_names=None, detector=None):
    return worker.start(model, class_names or [],
                        detector or NoHandDetector())


def frames_ready(worker):
    return (worker.running and worker.is_alive()
            and worker.snapshot.get("frames", 0) > 0)


def lock_free(index=TEST_INDEX):
    try:
        lock = camera_lock.acquire(index)
    except camera_lock.CameraBusyError:
        return False
    lock.release()
    return True


def stop_and_wait(worker, timeout=6.0):
    worker.stop()
    return wait_until(lambda: not worker.is_alive(), timeout)


# ---------------------------------------------------------------------------
# A. syntax / imports / production config untouched
# ---------------------------------------------------------------------------
def test_A_syntax():
    print("\nA. syntax and imports", flush=True)
    files = ["app.py", "src/camera_worker.py", "src/camera_lock.py",
             "src/mjpeg_server.py", "src/hand_detector.py", "src/config.py",
             "scripts/predict_realtime.py"]
    ok, detail = True, ""
    for relative in files:
        try:
            py_compile.compile(str(PROJECT_ROOT / relative), doraise=True)
        except py_compile.PyCompileError as error:
            ok, detail = False, f"{relative}: {error}"
            break
    check("A1. all changed files compile", ok, detail)
    # Reaching this line at all proves the imports above succeeded
    # (camera_worker pulls in cv2, camera_lock, mjpeg_server, config,
    # feature_extractor, hand_detector, sentence_builder, predict_realtime).
    check("A2. changed modules import cleanly", True)
    check("A3. production CAMERA_INDEX/CAMERA_BACKEND unchanged",
          config.CAMERA_INDEX == 0 and config.CAMERA_BACKEND == "",
          f"index={config.CAMERA_INDEX} backend={config.CAMERA_BACKEND!r}")


# ---------------------------------------------------------------------------
# B. trained model loads unchanged + predict_gesture round-trip
# ---------------------------------------------------------------------------
def test_B_model():
    print("\nB. trained model load and prediction (no retraining)", flush=True)
    try:
        model, class_names = load_classifier()
    except SystemExit as error:
        check("B1. model loads", False, f"SystemExit: {error}")
        return None, None
    except Exception as error:  # noqa: BLE001 - report, don't crash the run
        check("B1. model loads", False, f"{type(error).__name__}: {error}")
        return None, None

    check("B1. model loads", model is not None)
    check("B2. 12 classes preserved",
          len(class_names) == 12
          and set(class_names) == set(SIGN_CLASSES),
          ", ".join(map(str, class_names)))
    n_features = getattr(model, "n_features_in_", None)
    check("B3. model expects 126 features",
          n_features == MAX_FEATURE_LENGTH == 126, str(n_features))
    check("B4. confidence threshold still 0.50",
          MIN_PREDICTION_CONFIDENCE == 0.50,
          f"{MIN_PREDICTION_CONFIDENCE:.2f}")

    sample = build_sample([make_hand("Right")])
    check("B5. pipeline vector is 126 finite values",
          len(sample) == 126 and bool(np.all(np.isfinite(sample))))
    try:
        label, confidence, proba = predict_gesture(model, sample, class_names)
    except Exception as error:  # noqa: BLE001
        check("B6. predict_gesture runs on a pipeline vector", False,
              f"{type(error).__name__}: {error}")
        return model, class_names
    check("B6. predict_gesture returns a trained class + confidence",
          label in class_names and 0.0 <= confidence <= 1.0,
          f"{label} {confidence:.3f}")
    check("B7. probabilities sum to 1",
          abs(sum(proba.values()) - 1.0) < 1e-5,
          f"sum={sum(proba.values()):.6f}")
    return model, class_names


# ---------------------------------------------------------------------------
# C. fast predict path: bit-identity + fallback
# ---------------------------------------------------------------------------
def test_C_fast_path(model, class_names):
    print("\nC. fast predict path (predict_proba_single)", flush=True)
    if model is None:
        check("C0. model available for fast-path checks", False,
              "model failed to load")
        return

    rng = np.random.default_rng(20261014)
    vectors = rng.uniform(-3.0, 3.0, size=(50, MAX_FEATURE_LENGTH))
    vectors[0] = 0.0                       # degenerate but valid input
    max_diff = 0.0
    identical = True
    for row in vectors:
        fast = predict_proba_single(model, row)
        slow = model.predict_proba([row])
        if fast.shape != slow.shape:
            identical = False
            max_diff = float("inf")
            break
        max_diff = max(max_diff, float(np.max(np.abs(fast - slow))))
        identical = identical and bool(np.array_equal(fast, slow))
    check("C1. fast path bit-identical to predict_proba (50 vectors)",
          identical, f"max abs diff {max_diff}")

    pipeline_vector = build_sample([make_hand("Right")])
    fast = predict_proba_single(model, pipeline_vector)
    slow = model.predict_proba([pipeline_vector])
    check("C2. fast path bit-identical on a pipeline vector",
          bool(np.array_equal(fast, slow)),
          f"shape {fast.shape}")
    check("C3. fast path keeps sklearn's (1, 12) shape",
          fast.shape == (1, 12), str(fast.shape))

    # Fallback: an object WITHOUT sklearn internals must take the plain
    # model.predict_proba() route and return identical numbers.
    class ForeignModel:
        def predict_proba(self, x):
            return model.predict_proba(x)

    fallback = predict_proba_single(ForeignModel(), vectors[1])
    check("C4. foreign model falls back to plain predict_proba",
          bool(np.array_equal(fallback,
                              model.predict_proba([vectors[1]]))))

    # Through predict_gesture end-to-end (label/confidence unchanged).
    label_fast, conf_fast, _ = predict_gesture(model, pipeline_vector,
                                               class_names)
    check("C5. predict_gesture via fast path returns valid output",
          label_fast in class_names and 0.0 <= conf_fast <= 1.0,
          f"{label_fast} {conf_fast:.3f}")


# ---------------------------------------------------------------------------
# D/E/F. camera failure modes -> clear error, thread termination, cleanup
# ---------------------------------------------------------------------------
def _assert_failed_start(name_prefix, worker, created, expected_text):
    check(f"{name_prefix}a. start() accepted the request",
          start_worker(worker) == "")
    check(f"{name_prefix}b. thread terminates after the failure",
          wait_until(lambda: not worker.is_alive(), 5.0))
    check(f"{name_prefix}c. error is specific and actionable",
          bool(worker.error) and expected_text in worker.error
          and str(TEST_INDEX) in worker.error, str(worker.error))
    check(f"{name_prefix}d. camera released", bool(created)
          and created[0].released)
    check(f"{name_prefix}e. camera lock free afterwards", lock_free())


def test_D_open_failure():
    print("\nD. camera open failure (mocked)", flush=True)
    worker, created = make_worker(opened=False)
    _assert_failed_start("D.", worker, created, "Could not open")
    check("D.f. no frames processed", worker.snapshot.get("frames") == 0)


def test_E_opened_but_no_frames():
    print("\nE. opened-but-no-frames (Windows 'in use' symptom)", flush=True)
    worker, created = make_worker(grab_ok=False)
    _assert_failed_start("E.", worker, created, "delivered no frames")


def test_F_read_failure_midloop():
    print("\nF. read failure in the loop (camera taken mid-run)", flush=True)
    worker, created = make_worker(read_ok=False)
    _assert_failed_start("F.", worker, created, "Failed to read a frame")


# ---------------------------------------------------------------------------
# G. duplicate-start prevention
# ---------------------------------------------------------------------------
def test_G_duplicate_start():
    print("\nG. duplicate start prevention", flush=True)
    worker, _created = make_worker()
    check("G1. first start accepted", start_worker(worker) == "")
    check("G2. worker produces frames",
          wait_until(lambda: frames_ready(worker), 5.0))
    thread_before = worker._thread
    check("G3. second start is idempotent ('')",
          start_worker(worker) == "")
    check("G4. no second thread spawned",
          worker._thread is thread_before)
    feed_threads = [t for t in threading.enumerate()
                    if t.name == "signbridge-feed"]
    check("G5. exactly one feed thread exists", len(feed_threads) == 1,
          str(len(feed_threads)))
    check("G6. stop terminates the thread", stop_and_wait(worker))
    check("G7. lock free after stop", lock_free())


# ---------------------------------------------------------------------------
# H. repeated start/stop cycles
# ---------------------------------------------------------------------------
def test_H_repeated_cycles():
    print("\nH. repeated start/stop cycles", flush=True)
    worker, created = make_worker()
    problems = []
    for cycle in range(1, 4):
        if start_worker(worker) != "":
            problems.append(f"cycle {cycle}: start refused")
            break
        if not wait_until(lambda: frames_ready(worker), 5.0):
            problems.append(f"cycle {cycle}: no frames")
            break
        worker.stop()
        if not wait_until(lambda: not worker.is_alive(), 6.0):
            problems.append(f"cycle {cycle}: thread stuck")
            break
    check("H1. 3 start/stop cycles complete", not problems,
          "; ".join(problems))
    check("H2. one camera opened per cycle, ALL released",
          len(created) == 3 and all(cam.released for cam in created),
          f"{len(created)} cameras, "
          f"{sum(cam.released for cam in created)} released")
    check("H3. lock free after cycles", lock_free())
    check("H4. stop() on an already-stopped worker is safe",
          worker.stop() is False)


# ---------------------------------------------------------------------------
# I. non-blocking stop + stopping state + start refusal while winding down
# ---------------------------------------------------------------------------
def test_I_non_blocking_stop():
    print("\nI. non-blocking stop (slow backend release)", flush=True)
    worker, created = make_worker(release_delay=2.0)
    check("I1. start + frames", start_worker(worker) == ""
          and wait_until(lambda: frames_ready(worker), 5.0))
    started = time.perf_counter()
    worker.stop()                       # join timeout 0.5 s
    elapsed = time.perf_counter() - started
    check("I2. stop() returns fast (UI never blocks)", elapsed < 1.5,
          f"{elapsed:.2f} s")
    check("I3. worker reports the stopping state", worker.is_stopping())
    if worker.is_stopping():
        message = start_worker(worker)
        check("I4. start refused with a message while stopping",
              bool(message) and "stopping" in message, message)
    else:
        check("I4. start refused with a message while stopping", False,
              "worker was not in stopping state")
    check("I5. thread winds down on its own",
          wait_until(lambda: not worker.is_alive(), 8.0))
    check("I6. slow-release camera eventually released",
          bool(created) and created[0].released)
    check("I7. lock free after wind-down", lock_free())


# ---------------------------------------------------------------------------
# J. keepalive: release without page/client, keep alive WITH feed client
# ---------------------------------------------------------------------------
def test_J_keepalive():
    print("\nJ. keepalive (auto camera release)", flush=True)
    worker, created = make_worker(keepalive=0.4)
    check("J1. start + frames", start_worker(worker) == ""
          and wait_until(lambda: frames_ready(worker), 5.0))
    check("J2. released with no page touch and no feed client",
          wait_until(lambda: not worker.is_alive(), 4.0))
    check("J3. keepalive stop is not an error", worker.error is None,
          str(worker.error))
    check("J4. camera released and lock free",
          bool(created) and created[0].released and lock_free())

    # A backgrounded tab stops touching the page, but its MJPEG <img>
    # connection stays open - the camera must NOT die in that state.
    worker2, created2 = make_worker(keepalive=0.4)
    original = mjpeg_server.connected_clients
    mjpeg_server.connected_clients = lambda: 1
    try:
        check("J5. start + frames with feed client attached",
              start_worker(worker2) == ""
              and wait_until(lambda: frames_ready(worker2), 5.0))
        time.sleep(1.2)     # 3x the keepalive window, page still silent
        check("J6. feed client keeps the camera alive",
              worker2.is_alive() and worker2.running)
    finally:
        mjpeg_server.connected_clients = original
    check("J7. released once the feed client is gone",
          wait_until(lambda: not worker2.is_alive(), 4.0))
    check("J8. no error on feed-client keepalive stop",
          worker2.error is None, str(worker2.error))
    check("J9. lock free", lock_free())


# ---------------------------------------------------------------------------
# K. cross-process camera lock
# ---------------------------------------------------------------------------
def test_K_cross_process_lock():
    print("\nK. cross-process camera lock", flush=True)
    # same-process double lock (two workers in one Streamlit process)
    first = camera_lock.acquire(96)
    try:
        camera_lock.acquire(96)
        check("K1. same-process double lock rejected", False,
              "no exception raised")
    except camera_lock.CameraBusyError as error:
        check("K1. same-process double lock rejected",
              error.holder_pid == os.getpid(), str(error))
    finally:
        first.release()
    again = camera_lock.acquire(96)
    again.release()
    check("K2. lock immediately reusable after release", True)

    # a stale lock file (dead PID inside) never blocks: the OS lock, not
    # the file content, is what grants ownership
    stale = camera_lock.lock_path(95)
    stale.write_text("424242", encoding="ascii")
    try:
        lock = camera_lock.acquire(95)
        lock.release()
        check("K3. stale lock file does not block a fresh run", True)
    except camera_lock.CameraBusyError as error:
        check("K3. stale lock file does not block a fresh run", False,
              str(error))
    finally:
        stale.unlink(missing_ok=True)

    # another PROCESS holds the lock -> CameraBusyError names its PID.
    # The venv python.exe is a redirector: Popen's pid can differ from the
    # real interpreter's os.getpid(), so the child reports its own PID to a
    # file and BOTH K4 and K5 compare against that - never against
    # process.pid. Cleanup kills the actual holder by its reported PID, so
    # no orphan can outlive this test even if terminate() only hits the
    # launcher.
    import tempfile
    pid_file = Path(tempfile.gettempdir()) / "signbridge_test_holder.pid"
    pid_file.unlink(missing_ok=True)
    holder_code = (
        "import os, sys, time\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from src import camera_lock\n"
        f"open(r'{pid_file}', 'w').write(str(os.getpid()))\n"
        "camera_lock.acquire(99)\n"
        "time.sleep(60)\n"
    )
    executable = getattr(sys, "_base_executable", None) or sys.executable
    process = subprocess.Popen([executable, "-c", holder_code])
    free_after = False
    child_pid = None
    try:
        def _child_pid():
            try:
                text = pid_file.read_text().strip()
                return int(text) if text.isdigit() else None
            except OSError:
                return None

        wait_until(lambda: _child_pid() is not None, 6.0)
        child_pid = _child_pid()
        check("K4a. holder process started and reported its PID",
              child_pid is not None, str(child_pid))
        seen = wait_until(
            lambda: camera_lock.holder_pid(99) == child_pid, 6.0)
        check("K4b. holder PID is published in the lock file", seen,
              f"holder={camera_lock.holder_pid(99)} child={child_pid}")
        try:
            leaked = camera_lock.acquire(99)
        except camera_lock.CameraBusyError as error:
            check("K5. second process rejected with CameraBusyError",
                  error.holder_pid == child_pid, str(error))
        else:
            leaked.release()
            check("K5. second process rejected with CameraBusyError",
                  False, "acquire unexpectedly succeeded")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        holder = camera_lock.holder_pid(99)
        if holder and holder != os.getpid():
            try:
                os.kill(holder, 9)       # TerminateProcess on the real PID
            except OSError:
                pass
        free_after = wait_until(lambda: lock_free(99), 5.0)
        pid_file.unlink(missing_ok=True)
    check("K6. lock free once the holding process exits", free_after)


# ---------------------------------------------------------------------------
# L. end-to-end: worker thread + fake hand + REAL trained model
# ---------------------------------------------------------------------------
def test_L_end_to_end(model, class_names):
    print("\nL. end-to-end worker with a hand (real model, mocked camera)",
          flush=True)
    if model is None:
        check("L0. model available for the end-to-end run", False,
              "model failed to load")
        return
    worker, created = make_worker()
    check("L1. start accepted",
          start_worker(worker, model, class_names, WithHandDetector()) == "")
    check("L2. worker processes frames",
          wait_until(lambda: worker.snapshot.get("frames", 0) > 5, 6.0),
          f"frames={worker.snapshot.get('frames')}")
    snapshot = worker.snapshot
    check("L3. hand detected (parse_hands on stub MediaPipe result)",
          snapshot.get("hands") == 1, str(snapshot.get("hands")))
    check("L4. raw prediction from the real model (build_sample -> "
          "predict_gesture)", bool(snapshot.get("raw_text")),
          str(snapshot.get("raw_text")))
    check("L5. no worker error in the full pipeline", worker.error is None,
          str(worker.error))
    stats = worker._stats
    check("L6. per-stage instrumentation recorded",
          stats["capture_count"] > 0 and stats["features_count"] > 0
          and stats["model_count"] > 0 and stats["push_count"] > 0,
          f"capture={stats['capture_count']} features="
          f"{stats['features_count']} model={stats['model_count']} "
          f"push={stats['push_count']}")
    check("L7. stop terminates the thread", stop_and_wait(worker))
    check("L8. camera released and lock free",
          bool(created) and created[0].released and lock_free())


# ---------------------------------------------------------------------------
def main():
    print("=" * 70, flush=True)
    print("Camera ownership / worker lifecycle test (webcam mocked)", flush=True)
    print("=" * 70, flush=True)
    # The worker tests must never touch the real camera index or a real
    # device: patch the index the worker locks/opens to a fake one.
    cw.CAMERA_INDEX = TEST_INDEX

    test_A_syntax()
    model, class_names = test_B_model()
    test_C_fast_path(model, class_names)
    test_D_open_failure()
    test_E_opened_but_no_frames()
    test_F_read_failure_midloop()
    test_G_duplicate_start()
    test_H_repeated_cycles()
    test_I_non_blocking_stop()
    test_J_keepalive()
    test_K_cross_process_lock()
    test_L_end_to_end(model, class_names)
    return _summary()


if __name__ == "__main__":
    sys.exit(main())
