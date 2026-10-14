"""Cross-process ownership lock for the physical webcam.

WHY THIS EXISTS (reproduced during the camera-in-use repair)
    OpenCV on Windows can report ``VideoCapture(index).isOpened() == True``
    even while ANOTHER process already owns the device - the first
    ``read()`` then fails and the page can only show a generic "failed to
    read" message. The user cannot tell WHO holds the camera.

    Every SignBridge tool that opens a webcam (the Streamlit background
    worker in src/camera_worker.py and the OpenCV scripts) first takes this
    advisory lock on a small file in the OS temp directory. A second
    SignBridge process then fails BEFORE touching the device and receives
    the holder's PID, so the error names the actual culprit.

DESIGN
    - The lock is an OS-level byte-range lock (msvcrt on Windows, flock on
      POSIX). The operating system releases it automatically when the
      holding process exits or crashes - a stale lock file can never block
      a later run (unlike a plain "pid exists" file).
    - The holder writes its PID into the file right after acquiring, so a
      losing process can name it in the error message.
    - A per-process registry also rejects a second lock from the SAME
      process (two worker instances in one Streamlit process), reporting
      this process's own PID.
    - Scope: SignBridge processes only. A camera held by a different
      application (Teams, Zoom, Windows Camera, a browser call) is
      detected separately by the open/read checks in camera_worker.
"""

import os
import sys
import tempfile
import threading
from pathlib import Path

# One lock file per camera index - shared by every copy/branch of the
# project on this machine, because they compete for the same physical
# device.
LOCK_DIR = Path(tempfile.gettempdir())

# The byte-range lock sits at this offset; the holder's PID is written at
# offset 0. They must NOT overlap: on Windows a locked byte range cannot
# even be read by another process, so locking byte 0 would hide the PID
# from exactly the process that needs to report it. Windows validates the
# WHOLE requested read range against locks (a buffered read(n) asks for
# 8 KB and would span a lock at byte 0), so PID readers must read to EOF -
# that issues an exact-size read of the short PID bytes only, which never
# reaches the locked byte here.
LOCK_OFFSET = 64

_local_locks = {}          # index -> CameraLock held by THIS process
_local_guard = threading.Lock()


class CameraBusyError(RuntimeError):
    """Raised when the webcam index is already owned by another process."""

    def __init__(self, index, holder_pid=None):
        self.index = int(index)
        self.holder_pid = holder_pid
        if holder_pid is None:
            detail = "unknown SignBridge process"
        elif int(holder_pid) == os.getpid():
            detail = f"this process (PID {os.getpid()})"
        else:
            detail = f"PID {holder_pid}"
        super().__init__(f"webcam {self.index} is locked by {detail}")


def lock_path(index) -> Path:
    return LOCK_DIR / f"signbridge_camera_{int(index)}.lock"


class CameraLock:
    """Advisory cross-process lock for one camera index (context manager)."""

    def __init__(self, index):
        self.index = int(index)
        self._handle = None

    # ----------------------------------------------------------- acquisition
    def acquire(self) -> "CameraLock":
        with _local_guard:
            if self.index in _local_locks:
                raise CameraBusyError(self.index, os.getpid())
            handle = lock_path(self.index).open("a+b")
            try:
                if sys.platform == "win32":
                    import msvcrt
                    handle.seek(LOCK_OFFSET)
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    except OSError:
                        raise CameraBusyError(self.index,
                                              self._read_holder(handle))
                else:
                    import fcntl
                    try:
                        fcntl.flock(handle.fileno(),
                                    fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError:
                        raise CameraBusyError(self.index,
                                              self._read_holder(handle))
            except CameraBusyError:
                handle.close()
                raise
            except Exception:
                handle.close()
                raise
            # We own the device lock -> publish our PID for future losers.
            try:
                handle.seek(0)
                handle.truncate()
                handle.write(str(os.getpid()).encode("ascii"))
                handle.flush()
            except OSError:
                pass  # PID is best-effort information only
            self._handle = handle
            _local_locks[self.index] = self
            return self

    def release(self) -> None:
        handle, self._handle = self._handle, None
        with _local_guard:
            if _local_locks.get(self.index) is self:
                del _local_locks[self.index]
        if handle is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                handle.seek(LOCK_OFFSET)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass  # closing the handle releases the lock anyway
        finally:
            handle.close()

    @staticmethod
    def _read_holder(handle):
        """Best effort: PID written by the current lock holder.

        Read TO EOF (not read(n)): while the holder's byte-range lock is
        active, Windows denies any buffered read whose requested range
        spans the locked byte - and read-to-EOF requests exactly the file
        size, i.e. only the short PID bytes at offset 0.
        """
        try:
            handle.seek(0)
            raw = handle.read().decode("ascii", "ignore").strip()
            return int(raw) if raw.isdigit() else None
        except OSError:
            return None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()
        return False


def acquire(index) -> CameraLock:
    """Lock the webcam index; raise CameraBusyError when already owned."""
    return CameraLock(index).acquire()


def holder_pid(index):
    """PID recorded in the lock file (does not acquire anything).

    Reads to EOF for the same Windows reason as CameraLock._read_holder:
    while another process holds the lock, an exact-size read of the short
    PID bytes succeeds where read(n) would be denied.
    """
    path = lock_path(index)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            raw = handle.read().decode("ascii", "ignore").strip()
        return int(raw) if raw.isdigit() else None
    except OSError:
        return None
