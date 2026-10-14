"""Phase 8 performance fix: MJPEG preview feed (Python standard library).

WHY THIS EXISTS
    The first Phase 8 version pushed video by re-running the ENTIRE
    Streamlit script once per frame (st.rerun() + st.image). Every frame
    paid for a full page render, which capped the preview at ~6 FPS
    (server log: 6 script runs per second). This module moves the video
    OUT of Streamlit's render loop:

        background worker --push(jpeg)--> this server --multipart feed-->
        the browser's <img src="http://localhost:8502/feed">

    The browser renders the multipart/x-mixed-replace (MJPEG) stream
    natively at the feed's frame rate, while the Streamlit page is
    rendered once and never re-runs for frames. Streamlit only embeds a
    plain <img> tag - no st.image, no per-frame deltas, no WebRTC.

DEPENDENCIES
    None: only http.server, threading and time from the standard
    library. This is transport only - no Phase 5/6 logic lives here.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8502          # first port tried (8502..8511 as fallback)
IDLE_TIMEOUT_SECONDS = 30.0  # close a feed connection after this silence
STARTUP_TIMEOUT_SECONDS = 30.0  # give a client this long to see first frame
WRITE_TIMEOUT_SECONDS = 10.0    # drop a client that stops reading frames


class _LatestFrame:
    """Thread-safe holder for the newest JPEG (last-writer-wins)."""

    def __init__(self):
        self._condition = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self._pushed_at = 0.0

    def push(self, jpeg: bytes) -> None:
        """Store the newest frame and wake every waiting feed client."""
        with self._condition:
            self._jpeg = jpeg
            self._seq += 1
            self._pushed_at = time.monotonic()
            self._condition.notify_all()

    def wait_new(self, last_seq: int, timeout: float = 1.0):
        """Return (jpeg, seq) of a frame newer than last_seq, else None.

        Blocks on the condition variable whenever there is NOTHING new to
        send - both "no frame exists yet" (a client that connected before
        the worker's first frame: last_seq is -1 while _seq is still 0)
        and "no frame newer than the last one written". Missing the first
        case made the handler return immediately in a tight loop, i.e.
        spin one core until the first frame or the startup timeout.
        """
        with self._condition:
            if self._jpeg is None or self._seq == last_seq:
                self._condition.wait(timeout)
            if self._jpeg is None or self._seq == last_seq:
                return None
            return self._jpeg, self._seq

    def idle_for(self) -> float:
        """Seconds since the last pushed frame (inf when nothing yet)."""
        with self._condition:
            if self._pushed_at == 0.0:
                return float("inf")
            return time.monotonic() - self._pushed_at

    def has_frame(self) -> bool:
        """True once at least one frame was ever pushed."""
        with self._condition:
            return self._jpeg is not None


LATEST = _LatestFrame()
_server = None
_port = None
_lock = threading.Lock()

# Connected feed clients (one per browser <img> pulling the preview).
# The worker uses this as a keepalive signal: a backgrounded tab stops
# sending page touches (browsers throttle JS timers), but its MJPEG
# connection stays open - so the camera must not be released while a
# viewer is still attached. When the tab closes, the socket drops,
# the counter reaches 0 and the touch-based keepalive takes over.
_clients = 0
_clients_lock = threading.Lock()


def connected_clients() -> int:
    """Number of browser tabs currently streaming the preview feed."""
    with _clients_lock:
        return _clients


def _client_enter() -> None:
    global _clients
    with _clients_lock:
        _clients += 1


def _client_leave() -> None:
    global _clients
    with _clients_lock:
        _clients = max(0, _clients - 1)


class _FeedHandler(BaseHTTPRequestHandler):
    """Streams the newest frame over and over (MJPEG multipart)."""

    def do_GET(self):
        if self.path.split("?")[0].rstrip("/") not in ("", "/feed"):
            self.send_error(404)
            return
        _client_enter()
        try:
            # A browser that stops reading (frozen tab, wedged socket) must
            # not pin this handler thread: without a deadline, wfile.write()
            # below can block forever once the TCP buffer fills. The timeout
            # surfaces as OSError and lands in the except clause here.
            try:
                self.connection.settimeout(WRITE_TIMEOUT_SECONDS)
            except OSError:
                pass
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control",
                             "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            connected_at = time.monotonic()
            last_seq = -1
            while True:
                item = LATEST.wait_new(last_seq, timeout=1.0)
                if item is None:
                    if LATEST.has_frame():
                        # Feed streamed before but went silent (camera
                        # stopped, no page attached) -> close eventually.
                        if LATEST.idle_for() > IDLE_TIMEOUT_SECONDS:
                            return
                    elif time.monotonic() - connected_at > \
                            STARTUP_TIMEOUT_SECONDS:
                        # Nothing was ever pushed (feed never started).
                        return
                    # else: waiting for the FIRST frame (the <img> can
                    # connect a moment before the worker's first frame) -
                    # keep the connection open instead of dropping it.
                    continue
                jpeg, seq = item
                head = (b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode()
                        + b"\r\n\r\n")
                # Separate writes: `jpeg` may be the cv2.imencode buffer
                # (bytes-like numpy) - no concatenation copy per frame.
                self.wfile.write(head)
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                last_seq = seq
        except (BrokenPipeError, ConnectionResetError, OSError):
            return  # browser closed the preview - completely normal
        finally:
            _client_leave()

    def log_message(self, *_args):
        """Keep the Streamlit console free of per-request noise."""
        pass


def ensure_running() -> int:
    """Start the feed server once per process; return its port."""
    global _server, _port
    with _lock:
        if _server is not None:
            return _port
        for port in range(DEFAULT_PORT, DEFAULT_PORT + 10):
            try:
                httpd = ThreadingHTTPServer(("127.0.0.1", port), _FeedHandler)
            except OSError:
                continue  # port in use - try the next one
            httpd.daemon_threads = True
            threading.Thread(target=httpd.serve_forever, daemon=True,
                             name="mjpeg-feed").start()
            _server, _port = httpd, port
            print(f"[feed] MJPEG preview server listening on "
                  f"http://localhost:{port}/feed", flush=True)
            return port
        raise RuntimeError("No free TCP port for the MJPEG preview "
                           "(tried 8502-8511).")


def feed_url() -> str:
    """URL the page embeds in its <img> element."""
    return f"http://localhost:{ensure_running()}/feed"


def push(jpeg) -> None:
    """Publish the newest frame (bytes or a cv2.imencode buffer)."""
    LATEST.push(jpeg)
