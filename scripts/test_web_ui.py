"""Phase 8 checks: web UI (app.py) - files, scope, real numbers, ARCHITECTURE.

Run (project virtual environment, from the project folder):

    python scripts/test_web_ui.py

WHAT IS CHECKED
    A. Files and dependencies exist (app.py, .streamlit/config.toml,
       streamlit in requirements.txt, the new feed/worker modules).
    B. Scope discipline: the page and the worker REUSE the Phase 5/6
       pipeline instead of re-implementing it, contain NO text-to-speech
       (Phase 7 is separate) and NO marketing vocabulary.
    C. Real numbers only: the figures the page displays are pinned to the
       Phase 4 evaluation artifacts (evaluation/metrics.json).
    D. The rendered page (headless, via streamlit.testing.v1.AppTest):
       runs without exceptions, required sections, all 12 gestures, the
       real metrics, the expected controls.
    E. PERFORMANCE ARCHITECTURE (the 6 FPS fix): video must NOT go
       through Streamlit's render loop (no st.rerun loop, no st.image),
       the page statistics run in a low-rate st.fragment, the preview is
       an MJPEG feed from a background worker thread that imports the
       unchanged Phase 5/6 pipeline, and the feed handler blocks on a
       condition variable instead of busy-spinning while no frame
       exists yet.
"""

import ast
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

APP_PATH = PROJECT_ROOT / "app.py"
WORKER_PATH = PROJECT_ROOT / "src" / "camera_worker.py"
FEED_PATH = PROJECT_ROOT / "src" / "mjpeg_server.py"
BROWSER_PATH = PROJECT_ROOT / "src" / "browser_camera.py"
MODE_PATH = PROJECT_ROOT / "src" / "camera_mode.py"
CONFIG_PATH = PROJECT_ROOT / ".streamlit" / "config.toml"
METRICS_PATH = PROJECT_ROOT / "evaluation" / "metrics.json"

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    suffix = f"  ({detail})" if detail else ""
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}{suffix}")


def collect_text(app_test) -> str:
    """All user-visible text of a rendered page, whatever the element.

    Expander labels count too: the redesign keeps Model Details / How It
    Works / Project Scope collapsed, and their titles are real page text.
    """
    chunks = []
    for collection_name in ("title", "header", "subheader", "markdown",
                            "caption", "text", "error", "warning", "info",
                            "button", "code", "expander"):
        collection = getattr(app_test, collection_name, None)
        if collection is None:
            continue
        try:
            items = list(collection)
        except TypeError:
            continue
        for item in items:
            value = getattr(item, "value", None)
            if value is None:
                value = getattr(item, "label", "")
            chunks.append(str(value))
    return "\n".join(chunks)


def _compile_ok(path: Path):
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        return True, ""
    except SyntaxError as error:
        return False, str(error)


def main() -> int:
    print("=" * 60)
    print("SIGNBRIDGE AI - PHASE 8 WEB UI CHECKS")
    print("=" * 60)

    source = APP_PATH.read_text(encoding="utf-8") if APP_PATH.exists() else ""
    lowered = source.lower()
    worker_source = (WORKER_PATH.read_text(encoding="utf-8")
                     if WORKER_PATH.exists() else "")
    feed_source = FEED_PATH.read_text(encoding="utf-8") \
        if FEED_PATH.exists() else ""
    browser_source = (BROWSER_PATH.read_text(encoding="utf-8")
                      if BROWSER_PATH.exists() else "")
    mode_source = (MODE_PATH.read_text(encoding="utf-8")
                   if MODE_PATH.exists() else "")
    both = (source + "\n" + worker_source).lower()

    # --- A: files + dependencies ------------------------------------------
    print("\nA. Files and dependencies")
    check("A1a. app.py exists in the project root", APP_PATH.exists())
    check("A1b. app.py is a Streamlit script (set_page_config + cache)",
          "st.set_page_config" in source and "@st.cache_resource" in source)
    check("A1c. worker + feed modules exist",
          WORKER_PATH.exists() and FEED_PATH.exists())
    check("A2. .streamlit/config.toml exists, dark base theme",
          CONFIG_PATH.exists()
          and 'base = "dark"' in CONFIG_PATH.read_text(encoding="utf-8"))
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(
        encoding="utf-8").lower()
    check("A3. requirements.txt includes streamlit",
          "streamlit" in requirements)
    check("A4. requirements.txt includes streamlit-webrtc (browser webcam)",
          "streamlit-webrtc" in requirements)
    check("A5. browser camera + mode modules exist",
          BROWSER_PATH.exists() and MODE_PATH.exists())

    # --- B: scope discipline ----------------------------------------------
    print("\nB. Scope discipline (reused pipeline, no TTS, no marketing)")
    check("B1. app.py is a real Streamlit UI", "import streamlit" in source)
    check("B2. NO text-to-speech anywhere (Phase 7 not started)",
          not any(word in both for word in
                  ("pyttsx3", "gtts", "playsound", "espeak")))
    marketing = ("revolution", "next-generation", "next generation",
                 "state-of-the-art", "cutting-edge", "cutting edge",
                 "world-class", "world class", "seamless", "powered by",
                 "game-chang", "99.9%", "10k+", "loved by", "unlock the")
    hits = sorted({word for word in marketing if word in lowered})
    check("B3. NO marketing vocabulary (design rules)", not hits,
          ", ".join(hits))
    reimplementation = ("def build_sample", "def predict_gesture",
                        "class VoteWindow", "RandomForestClassifier",
                        "class HandDetector", "class SentenceBuilder")
    missing_here = [text for text in reimplementation
                    if text in source or text in worker_source]
    check("B4. pipeline NOT re-implemented (imports, not copies)",
          not missing_here, "found: " + ", ".join(missing_here))
    reused = ("from scripts.predict_realtime import", "load_classifier",
              "predict_gesture", "VoteWindow", "from src.sentence_builder",
              "SentenceBuilder", "build_sample", "HandDetector",
              "parse_hands", "observe(")
    missing = [text for text in reused
               if text not in source and text not in worker_source]
    check("B5. reuses Phase 5/6 modules unchanged", not missing,
          "missing: " + ", ".join(missing) if missing else "")
    browser_reused = ("predict_gesture", "build_sample", "VoteWindow",
                      "parse_hands", "observe(", "SentenceBuilder")
    browser_missing = [text for text in browser_reused
                       if text not in browser_source]
    check("B5b. browser camera path reuses the same pipeline",
          not browser_missing,
          "missing: " + ", ".join(browser_missing) if browser_missing else "")
    browser_reimplemented = ("def build_sample", "def predict_gesture",
                             "class VoteWindow", "class HandDetector",
                             "class SentenceBuilder")
    browser_hits = [text for text in browser_reimplemented
                    if text in source or text in browser_source]
    # AST-based device check: the browser_camera.py docstring legitimately
    # NAMES cv2.VideoCapture when explaining why the server cannot use it.
    browser_tree = ast.parse(browser_source)
    opens_device = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "VideoCapture"
        for node in ast.walk(browser_tree))
    check("B5c. browser path never re-implements the pipeline and never "
          "opens a camera device itself",
          not browser_hits and not opens_device,
          "found: " + ", ".join(browser_hits) + f" device_call={opens_device}")
    check("B5d. app.py wires the browser path (processor + mode switch)",
          "from src import browser_camera, camera_mode" in source
          and "render_browser_camera" in source)
    check("B6. displayed numbers come from evaluation/metrics.json",
          "metrics.json" in source and METRICS_PATH.exists())
    ok, detail = _compile_ok(APP_PATH)
    check("B7a. app.py compiles (no syntax errors)", ok, detail)
    ok, detail = _compile_ok(WORKER_PATH)
    check("B7b. camera_worker.py compiles", ok, detail)
    ok, detail = _compile_ok(FEED_PATH)
    check("B7c. mjpeg_server.py compiles", ok, detail)
    ok, detail = _compile_ok(BROWSER_PATH)
    check("B7d. browser_camera.py compiles", ok, detail)
    ok, detail = _compile_ok(MODE_PATH)
    check("B7e. camera_mode.py compiles", ok, detail)

    # --- C: real numbers only ---------------------------------------------
    print("\nC. Real numbers (pinned to Phase 4 evaluation artifacts)")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    split = metrics["split"]
    params = metrics["model"]["parameters"]
    accuracy_text = f"{metrics['metrics']['accuracy'] * 100:.2f}%"
    check("C1a. dataset: 1,800 samples, 126 features, 12 classes",
          metrics["dataset"]["samples"] == 1800
          and metrics["dataset"]["features"] == 126
          and len(metrics["dataset"]["classes"]) == 12)
    check("C1b. balanced classes: 150 samples per class",
          set(metrics["dataset"]["class_counts"].values()) == {150})
    check("C2a. model: Random Forest, 200 trees",
          metrics["model"]["type"] == "RandomForestClassifier"
          and params["n_estimators"] == 200)
    check("C2b. held-out test accuracy renders as 98.61%",
          accuracy_text == "98.61%", accuracy_text)
    check("C2c. split: 1,440 train / 360 test (20% held out)",
          split["train_samples"] == 1440 and split["test_samples"] == 360
          and split["test_size"] == 0.2)

    # --- D: the rendered page ---------------------------------------------
    print("\nD. Rendered page (headless AppTest)")
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as error:  # noqa: BLE001 - report as failed checks
        check("D0. streamlit AppTest available", False, str(error))
        return _summary()

    # Force the SERVER camera mode for the headless run: AppTest renders
    # no browser, so the webrtc_streamer component cannot be exercised
    # here, and auto mode would probe the real webcam. app.py reads the
    # variable once per script run; this locks the local-Windows path
    # (Start button, MJPEG preview) deterministically.
    os.environ["SIGNBRIDGE_CAMERA_MODE"] = "server"
    app_test = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app_test.run()
    check("D1. page renders without exception", not app_test.exception,
          str(app_test.exception))
    check("D2. no error banners on first load", len(app_test.error) == 0,
          "; ".join(str(e.value) for e in app_test.error))
    page = collect_text(app_test)

    check("D3. header: title + 'Real-Time Gesture Recognition'",
          "SignBridge AI" in page and "Real-Time Gesture Recognition" in page)
    sections = ("Live Recognition", "Current Gesture", "Confidence",
                "Sentence Builder", "Project Overview",
                "Supported Gestures", "Model Details", "How It Works",
                "Project Scope")
    missing_sections = [name for name in sections if name not in page]
    check("D4. all required sections present", not missing_sections,
          "missing: " + ", ".join(missing_sections) if missing_sections else "")

    from src.config import SIGN_CLASSES
    missing_labels = [label for label in SIGN_CLASSES if label not in page]
    check("D5. all 12 gestures listed", not missing_labels,
          "missing: " + ", ".join(missing_labels) if missing_labels else "")

    real_numbers = (accuracy_text, f"{metrics['dataset']['samples']:,}",
                    str(metrics["dataset"]["features"]),
                    f"{params['n_estimators']} trees")
    missing_numbers = [value for value in real_numbers if value not in page]
    check("D6. real metrics rendered (98.61% / 1,800 / 126 / 200 trees)",
          not missing_numbers,
          "missing: " + ", ".join(missing_numbers) if missing_numbers else "")

    labels = [button.label for button in app_test.button]
    expected_buttons = {"Start camera", "SPACE", "DELETE", "Clear"}
    check("D7. controls: Start camera, SPACE, DELETE, Clear",
          expected_buttons <= set(labels), f"found: {labels}")
    disabled = {button.label: button.disabled for button in app_test.button}
    check("D8. edit buttons disabled while the message is empty",
          disabled.get("SPACE") and disabled.get("DELETE")
          and disabled.get("Clear")
          and not disabled.get("Start camera", True),
          str(disabled))
    check("D9. empty message shows the 'Start signing...' placeholder",
          "Start signing to build your message" in page
          and "(empty)" not in page,
          "empty state text")

    # --- D (browser mode): no crash when the component cannot run ---------
    # AppTest mocks the Streamlit Runtime, which streamlit-webrtc's
    # component requires; the page must show a clear error banner instead
    # of an uncaught exception. Runs in a SUBPROCESS: st.cache_resource is
    # process-wide, and the server-mode AppTest above already cached its
    # CameraWorker - a second in-process run would reuse that instead of
    # building the browser processor. Actual WebRTC negotiation needs a
    # real browser; the frame path is covered by
    # scripts/test_browser_camera.py.
    import subprocess
    env = dict(os.environ, SIGNBRIDGE_CAMERA_MODE="browser")
    completed = subprocess.run(
        [sys.executable, "-c",
         "from streamlit.testing.v1 import AppTest\n"
         "t = AppTest.from_file('app.py', default_timeout=60)\n"
         "t.run()\n"
         "print('MARKER_EXCEPTION', bool(t.exception))\n"
         "print('MARKER_ERRORS', repr([str(e.value) for e in t.error]))\n"],
        capture_output=True, text=True, timeout=180,
        cwd=str(PROJECT_ROOT), env=env)
    browser_out = completed.stdout + completed.stderr
    check("D10. browser mode renders without an uncaught exception",
          completed.returncode == 0
          and "MARKER_EXCEPTION False" in completed.stdout,
          browser_out[-200:])
    browser_errors = ""
    for line in completed.stdout.splitlines():
        if line.startswith("MARKER_ERRORS "):
            browser_errors = line
    check("D11. browser component failure shows a clear error banner",
          "browser camera" in browser_errors, browser_errors[:160])

    # --- E: performance architecture (the 6 FPS fix) -----------------------
    print("\nE. Performance architecture (video outside the render loop)")
    # AST check: actual st.<name>(...) CALLS only - the module docstrings
    # legitimately mention st.rerun/st.image when describing the old, slow
    # architecture that this fix replaced.
    st_calls = set()
    for node in ast.walk(ast.parse(source)):
        func = getattr(node, "func", None)
        if (isinstance(node, ast.Call) and isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "st"):
            st_calls.add(func.attr)
    check("E1. no st.rerun call (frames never re-run the script)",
          "rerun" not in st_calls, f"st calls: {sorted(st_calls)}")
    check("E2. no st.image call (Streamlit never encodes frames)",
          "image" not in st_calls, f"st calls: {sorted(st_calls)}")
    check("E3. statistics live in an st.fragment with run_every",
          "@st.fragment" in source and "run_every" in source)
    match = re.search(r"STATS_INTERVAL_SECONDS\s*=\s*([0-9.]+)", source)
    interval = float(match.group(1)) if match else -1.0
    check("E4. panel refresh several times per second (0.25-1.0 s)",
          0.25 <= interval <= 1.0, f"{interval} s")
    check("E5. preview is an MJPEG <img> fed by src/mjpeg_server.py",
          "mjpeg_server" in source and 'src="' in source
          and "multipart/x-mixed-replace" in feed_source)
    feed_imports = set()
    for node in ast.walk(ast.parse(feed_source)):
        if isinstance(node, ast.Import):
            feed_imports.update(alias.name.split(".")[0]
                                for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            feed_imports.add(node.module.split(".")[0])
    check("E6. feed module is stdlib only (no streamlit/cv2/numpy)",
          not (feed_imports & {"streamlit", "cv2", "numpy"})
          and "http" in feed_imports)
    check("E7. worker thread runs the pipeline and pushes frames",
          "threading.Thread" in worker_source
          and "mjpeg_server.push" in worker_source
          and "build_sample" in worker_source
          and "predict_gesture" in worker_source
          and "VoteWindow" in worker_source
          and ".observe(" in worker_source)
    check("E8. worker never imports Streamlit (no ScriptRunContext need)",
          "import streamlit" not in worker_source
          and "from streamlit" not in worker_source)
    check("E9. page keepalive touches the worker (auto camera release)",
          "touch()" in source and "_activity" in worker_source)
    # E10 (re-scoped): the 6 FPS architecture no longer needs WebRTC, but
    # the Cloud deployment needs the browser webcam path. The invariant is
    # now that WebRTC stays CONFINED to the browser path in app.py - the
    # OpenCV worker (and its MJPEG feed) must remain WebRTC/Streamlit-free
    # so local Windows behaviour is untouched.
    check("E10. WebRTC confined to the browser path (worker stays clean)",
          "streamlit_webrtc" not in worker_source
          and "aiortc" not in worker_source
          and "streamlit_webrtc" in source)
    check("E12. camera mode selection via SIGNBRIDGE_CAMERA_MODE, "
          "auto probes safely",
          "SIGNBRIDGE_CAMERA_MODE" in mode_source
          and "MODE_SERVER" in mode_source
          and "MODE_BROWSER" in mode_source
          and "probe_server_camera" in mode_source)
    # E11: _LatestFrame.wait_new() must BLOCK while there is nothing new
    # to send. A client that connects BEFORE the worker's first frame
    # (last_seq -1 vs _seq 0) used to make it return immediately, so the
    # handler looped in a tight spin and burned a core until the 30 s
    # startup timeout. Behaviour check on a private holder: with nothing
    # pushed, wait_new must take about the whole timeout, not 0 ms.
    from src.mjpeg_server import _LatestFrame
    empty_holder = _LatestFrame()
    wait_started = time.perf_counter()
    nothing = empty_holder.wait_new(-1, timeout=0.25)
    waited = time.perf_counter() - wait_started
    check("E11. feed blocks while no frame exists yet (no busy-spin)",
          nothing is None and waited >= 0.15,
          f"wait_new returned after {waited * 1000:.0f} ms")

    return _summary()


def _summary() -> int:
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("\n" + "=" * 46)
    print(f"Checks passed: {passed} / {total}")
    print("=" * 46)
    if passed != total:
        print("FAILED checks:")
        for name, ok, detail in RESULTS:
            if not ok:
                suffix = f"  ({detail})" if detail else ""
                print(f"  - {name}{suffix}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
