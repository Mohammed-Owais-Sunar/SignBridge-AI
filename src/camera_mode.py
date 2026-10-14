"""Camera-mode resolution for SignBridge AI (server vs browser webcam).

WHY THIS EXISTS
    Local Windows runs the server-side OpenCV pipeline (CameraWorker:
    cv2.VideoCapture -> MJPEG feed). Streamlit Community Cloud runs on a
    remote Linux server with NO camera hardware, so that pipeline fails
    there ("VIDEOIO(V4L2): can't open camera by index", "nb_devices is 0")
    before a single frame is processed. The fix is a second, browser-based
    pipeline: streamlit-webRTC captures the VISITOR'S camera in their
    browser and pipes frames into the same recognition pipeline.

    The two pipelines are mutually exclusive: exactly one is active per
    script run, chosen here ONCE (main-script scope, never inside a
    ticking st.fragment), so the mode is stable for the whole session and
    the two camera sources can never run at the same time.

MODE SELECTION (environment variable SIGNBRIDGE_CAMERA_MODE)
    auto   (default) - probe the configured OpenCV camera: open it, grab a
           couple of frames, release it. Opens and delivers frames ->
           "server". No camera device (Streamlit Community Cloud) ->
           "browser". The probe takes NO cross-process lock (a probe must
           never make a concurrent Start fail), releases the device
           before returning, and silences cv2's C-level stderr so a
           camera-less deployment logs no confusing V4L2 errors.
    server - always the OpenCV pipeline (local Windows). Explicit: no
           env override, just the documented worker behaviour.
    browser - always the browser webcam. The OpenCV probe is SKIPPED
           ENTIRELY (no /dev/video0 is ever opened), so nothing can
           produce Windows-style "camera in use" errors on Cloud.

    Unknown values fall back to auto with a one-line log note.

GUARANTEE
    The auto probe can never lock, leak or consume the camera: it holds
    the device only for the few frames it grabs, acquires no lock, and is
    fully released even when the probe itself raises (try/finally).
"""

import contextlib
import os
import sys
import tempfile

from src.config import CAMERA_BACKEND, CAMERA_INDEX

CAMERA_MODE_ENV = "SIGNBRIDGE_CAMERA_MODE"
MODE_SERVER = "server"
MODE_BROWSER = "browser"
MODE_AUTO = "auto"
VALID_MODES = (MODE_AUTO, MODE_SERVER, MODE_BROWSER)


@contextlib.contextmanager
def _silence_c_stderr():
    """Redirect C-level stderr (fd 2) to a temporary file while active.

    OpenCV prints probe failures ("VIDEOIO(V4L2): can't open camera by
    index", "nb_devices is 0") straight to file descriptor 2 from C++;
    contextlib.redirect_stderr cannot catch those. Auto mode swallows
    them so a camera-less deployment logs nothing confusing.
    """
    sys.stderr.flush()
    with tempfile.TemporaryFile(mode="w+b") as sink:
        saved_fd = os.dup(2)
        try:
            os.dup2(sink.fileno(), 2)
            yield
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            sys.stderr.flush()


def probe_server_camera(index=None, warmup_frames=2):
    """True when the configured OpenCV camera opens and delivers frames.

    Release-guarded: the device is released even when the probe raises, so
    the probe can never leak the camera. Takes NO cross-process lock (that
    is the worker's job at Start time) and silences cv2's stderr so the
    failure mode of a camera-less host stays quiet in the logs.
    """
    import cv2

    index = CAMERA_INDEX if index is None else int(index)
    backend = None
    if CAMERA_BACKEND:
        backend = getattr(cv2, f"CAP_{CAMERA_BACKEND.upper()}", None)
    with _silence_c_stderr():
        try:
            if backend is not None:
                camera = cv2.VideoCapture(index, backend)
            else:
                camera = cv2.VideoCapture(index)
        except Exception:  # noqa: BLE001 - any cv2 failure = no camera
            return False
        try:
            if not camera.isOpened():
                return False
            for _ in range(int(warmup_frames)):
                if not camera.grab():
                    return False
            return True
        finally:
            camera.release()


def resolve_camera_mode(env=None):
    """Return MODE_SERVER or MODE_BROWSER for this Streamlit script run.

    `env` defaults to os.environ (injectable for tests). Explicit modes
    are honoured verbatim - in particular, explicit MODE_BROWSER never
    touches the OpenCV camera at all. Auto mode probes once and falls
    back to the browser pipeline when no server camera exists.
    """
    env = os.environ if env is None else env
    requested = (env.get(CAMERA_MODE_ENV) or MODE_AUTO).strip().lower()
    if requested == MODE_SERVER:
        return MODE_SERVER
    if requested == MODE_BROWSER:
        return MODE_BROWSER          # OpenCV probe skipped entirely
    if requested != MODE_AUTO:
        print(f"[camera-mode] unknown {CAMERA_MODE_ENV}={requested!r} - "
              f"expected one of {', '.join(VALID_MODES)}; using auto",
              flush=True)
    return MODE_SERVER if probe_server_camera() else MODE_BROWSER
