# Debugging Report — Camera FPS & Stability Repair

**Branch:** `fix/camera-fps-stability`
**Scope:** root-cause fix for the reported ~5–6 FPS preview and the
"camera already in use" failures, plus an audit of reproducible bugs in the
same area. Evidence below is measured on the development machine
(Windows 11, Ryzen 7 7445HS, webcam 640×480@30, Python 3.13.7,
scikit-learn 1.9.1, mediapipe 0.10.35, OpenCV 5.0.0, streamlit 1.65.0).

**Invariants preserved:** trained Random Forest untouched (no retraining),
12 classes (HELLO … DELETE), 126-dim feature vector, 0.50 confidence
threshold, 5-frame vote window, existing dataset/CSV, and the OpenCV
(single `CameraWorker` → MJPEG) architecture.

---

## 1. Root causes (reproduced, then measured)

### 1.1 Low FPS with a hand in view (~5–14 FPS)

**Root cause — redundant sklearn input validation, once per tree.**
`RandomForestClassifier.predict_proba()` delegates to every tree's
`predict_proba()` with `check_input=True`. Each of the 200 trees
re-validates the **same single 126-value row**:

| path | time / query (measured) |
|---|---|
| `model.predict_proba([row])` (before) | **23.5–27.3 ms** |
| one tree with `check_input=True` | 0.206 ms |
| one tree with `check_input=False` | 0.014 ms |
| → redundant validation per query | **~19 ms (≈90 % of the call)** |

With a hand in view this cost hit the loop on **every frame**
(live: prediction stage avg 28.15 ms → worker dropped to 10.7–14.3 FPS;
reported worst case ~5–6 FPS on a loaded machine).

**Fix:** `predict_proba_single()` in `scripts/predict_realtime.py`
validates the row **once** (`_validate_X_predict`), then accumulates the
tree probabilities with `check_input=False`, mirroring scikit-learn's own
`_accumulate_prediction`. Verified **bit-identical**: max abs difference
`0.0` over 50 random vectors + a real pipeline vector (see §4). A plain
`model.predict_proba()` fallback covers any model without those internals.

**Secondary contributors (also measured):**

- MediaPipe detection ~18 ms/frame and the camera's own 30 FPS pacing are
  now the dominant stages — i.e. the pipeline runs at the camera's native
  rate instead of the prediction path.
- The once-per-second log line previously merged all stages; it was
  replaced with per-stage monotonic timings so regressions are attributable.

### 1.2 "Camera already in use" (generic or silent failures)

Reproduced with a second OpenCV process holding the webcam: the app logged
`[feed] ERROR: Failed to read a frame from the webcam.` and the UI showed a
generic "Camera error".

**Root causes:**

1. **No cross-process ownership.** On Windows (MSMF) a second open can
   report `isOpened() == True` while another process owns the device; the
   first `read()` then fails with no hint *who* holds the camera.
2. **False README claim:** "The OpenCV preview … keeps working
   independently" — it cannot; the webcam is exclusive per process.
3. **Blocking stop:** `stop()` joined the thread for up to 3 s, freezing
   the page on Stop.
4. **Keepalive blind spot:** release was driven only by page "touches";
   browsers throttle JS timers in backgrounded tabs, so a hidden tab could
   hold the camera indefinitely.

**Fixes:**

- `src/camera_lock.py` (new): OS-level advisory byte-range lock
  (`msvcrt`/`flock`) on a file in the temp dir, holder PID written at
  offset 0, lock at offset 64 (they must not overlap: on Windows a locked
  range cannot be read by another process — and PID readers must read *to
  EOF*, because Windows denies any buffered `read(n)` whose requested range
  spans the locked byte). The OS releases the lock automatically when the
  holder exits/crashes, so a stale file never blocks a later run.
  Taken by the worker **and** by every OpenCV script
  (`test_camera.py`, `scripts/test_hand_detection.py`,
  `scripts/predict_realtime.py`).
- **Preflight frame check:** after open, the worker performs warm-up
  grabs and fails with `Webcam N opened but delivered no frames …` if the
  device is starved (the Windows symptom of case 1).
- **Specific, actionable errors** for all three failure modes, each naming
  the webcam index (and the holder PID for SignBridge conflicts).
- **Non-blocking stop:** join timeout 0.5 s; the UI shows a disabled
  *Stopping camera…* button and refuses Start with a message until the
  worker has released the camera (`is_stopping()`).
- **Keepalive = no page touch AND no MJPEG feed client**
  (`mjpeg_server.connected_clients()`): a backgrounded tab's open preview
  connection keeps the camera alive; closing it releases within 10 s.
- **README corrected** and a **"Camera in use"** troubleshooting section
  added (symptom → message → remedy table, stop/restart instructions).

### 1.3 Audit findings fixed along the way

- **Fallback shape bug (introduced during the fast-path work, caught by the
  new tests):** the fallback returned `predict_proba([x])[0]` (shape
  `(12,)`) while the fast path returns `(1, 12)` — `predict_gesture` would
  have received a scalar on the fallback path. Now both paths return
  scikit-learn's shape `(1, n_classes)`.
- **Dependency mismatch:** `requirements.txt` pinned `opencv-python`, but
  mediapipe hard-requires `opencv-contrib-python`; installing both silently
  overwrites the same `cv2` files. The file now pins **only**
  `opencv-contrib-python`.
- **Feed socket robustness:** MJPEG writes now use a 10 s send timeout so
  a dead client cannot wedge the feed server.

---

## 2. Changed files

| file | change |
|---|---|
| `src/camera_lock.py` | **new** — cross-process camera ownership lock (+ PID reporting) |
| `src/camera_worker.py` | lock acquire/release in `_run`, preflight frame check, `CAMERA_BACKEND` support, non-blocking stop + `is_stopping()`, feed-client keepalive, per-stage monotonic instrumentation (capture/flip/color/infer/draw/features/model/smooth/encode/push/end-to-end), specific error messages |
| `src/mjpeg_server.py` | `connected_clients()` + client enter/leave counting, 10 s write timeout |
| `src/hand_detector.py` | `last_ms` split of BGR→RGB conversion vs MediaPipe inference |
| `src/config.py` | `CAMERA_INDEX` / `CAMERA_BACKEND` settings + fast-path note |
| `scripts/predict_realtime.py` | `predict_proba_single()` validate-once fast path (+ shape-consistent fallback), used by `predict_gesture()`; camera lock in `main()` |
| `app.py` | "stopping" worker state in `worker_state()`/`status_html()`, disabled *Stopping camera…* button, Start-message wiring — no new `st.*` call kinds (E1/E2/E10 constraints hold) |
| `requirements.txt` | `opencv-python` → `opencv-contrib-python` (mediapipe requirement, with comment) |
| `README.md` | false concurrency claim removed, keepalive description updated, measured numbers refreshed, new "Camera in use" section |
| `test_camera.py`, `scripts/test_hand_detection.py` | advisory camera lock (named-PID error instead of a hung preview) |
| `scripts/test_camera_lifecycle.py` | **new** — 73-check offline test suite (§3) |
| `DEBUGGING_REPORT.md` | this document |

**Not touched:** `models/`, dataset/CSV, thresholds (0.50), vote window (5),
class list, evaluation metrics, user backup files
(`app.py.backup`, `src/browser_camera.py.backup`, `src/camera_worker.py.bak`
remain untracked and unchanged).

---

## 3. Tests

New suite — `python scripts/test_camera_lifecycle.py` → **73/73 passed**
(offline: cameras are mocked via the worker's `camera_factory` seam; lock
tests use fake indices 95/96/97/99, never the real webcam):

- **A** syntax/imports of every changed module; production
  `CAMERA_INDEX`/`CAMERA_BACKEND` unchanged
- **B** model loads: 12 classes, 126 features, threshold 0.50,
  `predict_gesture` round-trip (probabilities sum to 1)
- **C** fast path **bit-identical** (max abs diff 0.0, 50 random + 1
  pipeline vector), `(1, 12)` shape, foreign-model fallback equivalence
- **D** camera open failure → specific error, thread terminates, camera
  released, lock free
- **E** opened-but-no-frames (the Windows "in use" symptom) → specific error
- **F** mid-loop read failure → specific error, release in `finally`
- **G** duplicate-start prevention (one thread, idempotent start)
- **H** 3× start/stop cycles → one camera per cycle, all released, lock free
- **I** non-blocking stop (<1.5 s even with a 2 s backend release),
  `is_stopping()` state, Start refused while winding down, thread winds
  down on its own
- **J** keepalive releases with no page **and** no feed client; a feed
  client keeps the camera alive (backgrounded-tab case); releases when the
  client disappears
- **K** cross-process lock: second process gets `CameraBusyError` naming
  the holder's real PID (the venv redirector pid vs `os.getpid()` pitfall
  handled), same-process double-lock rejected, stale lock file harmless
- **L** end-to-end in the worker thread: fake hand → `parse_hands` →
  `build_sample` → **real model** → vote → sentence, instrumentation
  counters populated

Existing suites (all re-run after the final edits):

| suite | result |
|---|---|
| `python scripts/test_feature_extraction.py` | **42/42** |
| `python scripts/test_sentence_builder.py` | **40/40** |
| `python scripts/test_web_ui.py` (Streamlit `AppTest`) | **39/39**, exit 0 |
| `python -m compileall app.py test_camera.py src scripts` | clean |

The mocked lifecycle suite does **not** claim physical-webcam coverage;
real hardware was exercised separately by the measurements in §4.

---

## 4. Measured results (before → after)

### Prediction path (the 23–27 ms fix)

| metric | before | after |
|---|---|---|
| `predict_proba` per query | 23.5–27.3 ms | **5.1–5.3 ms** (fast path) |
| outputs vs `model.predict_proba` | — | **bit-identical** (max abs diff 0.0, 50/50 samples) |

### Headless run (`scripts/measure_web_performance.py`, hand in view)

| metric | before | after |
|---|---|---|
| delivered to feed client | 29.7 FPS (no hand) | **30.1 FPS (hand in view)** |
| worker FPS (1 s log avg) | 21–30 (drops with hand) | **29.9 avg** |
| prediction stage (`rf`) | 23–27 ms | **2.4 ms avg** |
| per-stage averages | merged log | capture 12.2 / flip 0.2 / rgb 0.3 / MediaPipe 17.7 / draw 1.1 / features 0.04 / model 2.4 / smooth 0.05 / JPEG 0.7 / push 0.02 ms |

### Live browser (CDP, page under test)

| metric | before | after |
|---|---|---|
| worker FPS with hands in view | **10.7–14.3 FPS** | **29.7–29.8 FPS** |
| stats strip as the user sees it | ~10–14 FPS, pred ~28 ms | **31–45 FPS, pred 2.7–5.4 ms** |
| rendered preview (CDP screencast) | 33.6 composited updates/s | **38.8 composited updates/s** |
| MJPEG transport | healthy (1 request, 0 remounts) | unchanged: 1 request, 0 `<img>` remounts |

*(The in-page canvas sampler used once for rendered FPS now reports a
cross-origin `SecurityError`: the feed lives on `localhost:8502` while the
page is on `localhost:8501`. This is a measurement artifact of the test
harness — rendered FPS is therefore compared with the CDP-screencast
number on both sides.)*

### Lifecycle (live browser)

- Start → live preview: **~4 s** on a cold model/detector (loads cached afterwards)
- Stop: **≤ 1 s** (join timeout 0.5 s; UI never blocks)
- Restart: **~1 s**; rapid stop/start ×3: no errors, camera released each cycle
- Closing the tab: camera auto-released after the 10 s keepalive

### Camera conflicts (live, both reproduced before the fix)

| scenario | before | after |
|---|---|---|
| other app holds the webcam (plain OpenCV/Teams/…) | generic *"Failed to read a frame from the webcam."* | **"Webcam 0 opened but delivered no frames — another application is holding the device (Windows allows the open but starves the reads). Close other camera apps, then press Start again."** in ~5 s |
| second SignBridge instance / preview script | generic failure, no owner info | **"Webcam 0 is already owned by another SignBridge process (PID 11744) — stop it, e.g. `Get-Process -Id 11744`. …"** in ~4 s |

---

## 5. How to reproduce (PowerShell, project root)

```powershell
# environment
.\venv\Scripts\pip install -r requirements.txt     # opencv-contrib-python only

# tests (all offline except the two measurement scripts)
.\venv\Scripts\python scripts\test_feature_extraction.py   # 42/42
.\venv\Scripts\python scripts\test_sentence_builder.py     # 40/40
.\venv\Scripts\python scripts\test_web_ui.py               # 39/39 (AppTest)
.\venv\Scripts\python scripts\test_camera_lifecycle.py     # 73/73 (new)
.\venv\Scripts\python -m compileall app.py test_camera.py src scripts

# measurements (webcam required; stop the app first)
.\venv\Scripts\python scripts\measure_web_performance.py   # headless FPS + stages
.\venv\Scripts\streamlit run app.py                        # manual/live check

# failure messages (from the temp harness used during this repair)
powershell -ExecutionPolicy Bypass -File "$env:TEMP\opencode\run_conflict_test.ps1"            # plain holder → "delivered no frames"
powershell -ExecutionPolicy Bypass -File "$env:TEMP\opencode\run_conflict_test.ps1" -HolderLock  # lock holder → named PID
```

## 6. Manual verification steps

1. `streamlit run app.py` → **Start camera** → preview is smooth with a
   hand in view (strip shows ~30 FPS, prediction a few ms).
2. **Stop camera** → button switches to *Stopping camera…*, page stays
   responsive, Start returns within ~1 s.
3. Start, then close the browser tab → within ~10 s another `Start`
   succeeds (keepalive released the camera), or start a second app
   instance in another terminal → it fails naming this process's PID.
4. With the app running, open `python scripts/predict_realtime.py` → it
   exits immediately with *"webcam 0 is already owned by another SignBridge
   process (PID …)"*; reverse the order to see the app's lock error.
5. Hold the camera with the Windows Camera app / a Teams call → Start shows
   the *"opened but delivered no frames"* message; closing that app lets
   Start succeed.

## 7. Branch, commit and push status

- Branch: **`fix/camera-fps-stability`** (created before any change, from
  `browser-webcam-support`).
- Commit: single repair commit on that branch (see `git log -1`).
- **Push status: NOT pushed.** No `git push`, force-push, reset, branch
  deletion or history rewrite was performed; the three untracked user
  backup files were left untouched.
