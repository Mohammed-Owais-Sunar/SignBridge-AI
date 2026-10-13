"""Phase 8: SignBridge AI web interface (Streamlit) - presentation redesign.

RUN (project folder, project virtual environment):

    streamlit run app.py        ->  http://localhost:8501

ARCHITECTURE - PERFORMANCE LOCK (unchanged in this redesign)
    The first Phase 8 version re-ran the ENTIRE script once per video
    frame (st.rerun + st.image), which capped the preview at ~6 FPS.
    The current version runs three independent rates:

      1. camera + Phase 5/6 pipeline : continuous background thread
         (src/camera_worker.py) - full speed, never re-renders
      2. video to the browser        : MJPEG HTTP feed (src/mjpeg_server.py)
         - one <img>; its markup changes only on Start/Stop, the browser
           pulls every frame itself (verified: the node survives ticks,
           so the feed connection is never reconnected)
      3. page regions                : every live region is its own
         st.fragment at STATS_INTERVAL_SECONDS (0.4 s) - status pill,
         camera control, gesture card, statistics + sentence. The main
         script runs once per page load and never per frame.

    Heavy resources (Random Forest + class list, MediaPipe detector,
    evaluation metrics) load once via st.cache_resource / st.cache_data.
    src/camera_worker.py and src/mjpeg_server.py are NOT part of this
    redesign - they stay exactly as they are.

PRESENTATION LAYER (the only part this file redesigns)
    PAGE_CSS, the layout/composition and the HTML helpers that FORMAT
    values the worker already computed. Recognition behaviour is
    untouched: labels come from the worker snapshot, the confidence
    threshold, the vote window and the SentenceBuilder instance are the
    same objects Phases 5/6 used.

    Formatting rule: underscores are hidden in the browser only
    (I_LOVE_YOU -> I LOVE YOU). The stored/model label stays
    I_LOVE_YOU and is still present in the markup (data-class/title).

SCOPE: presentation only. No text-to-speech (Phase 7), no deployment,
no retraining, no dataset change. Design rules: dark neutral background,
one accent colour, one success/warning/error colour, borders instead of
shadows, no gradients, no icons, no emoji, real numbers only
(evaluation/metrics.json + src/config.py), no marketing language.
"""

import html
import json
import sys
from pathlib import Path

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Make the project importable regardless of the working directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import browser_camera
from src.config import (MIN_PREDICTION_CONFIDENCE,
                        MODEL_PATH as HAND_MODEL_PATH, MODEL_URL,
                        PREDICTION_VOTE_WINDOW, SIGN_CLASSES)
from src.feature_extractor import MAX_FEATURE_LENGTH
from src.hand_detector import HandDetector
from src.sentence_builder import NO_HAND_LABEL
from scripts.predict_realtime import load_classifier

METRICS_PATH = PROJECT_ROOT / "evaluation" / "metrics.json"

# UI statistics refresh rate. The video does NOT depend on it - it
# streams through the MJPEG feed - so the panel can update several
# times per second without touching the frame pipeline at all.
STATS_INTERVAL_SECONDS = 0.4

# Short factual meaning of every predefined gesture - what it does in
# THIS project's vocabulary (not a claim about sign language in general).
GESTURE_MEANING = {
    "HELLO": "greeting",
    "YES": "affirmative answer",
    "NO": "negative answer",
    "THANK_YOU": "thanks",
    "PLEASE": "polite request",
    "HELP": "asking for help",
    "SORRY": "apology",
    "I_LOVE_YOU": "I-love-you sign",
    "GOOD": "positive evaluation",
    "BAD": "negative evaluation",
    "SPACE": "insert one space between words",
    "DELETE": "remove the last word",
}

# These two classes edit the message instead of adding a word: they are
# controls of this project's Sentence Builder (shown as such in the UI).
SENTENCE_CONTROLS = {"SPACE", "DELETE"}

# Header disclaimer (kept short and quiet on purpose).
SCOPE_NOTE = ("Predefined static hand gestures only - this is not"
              " continuous sign-language translation.")

# Full scope statement (Project Scope section).
PROJECT_SCOPE = ("This project recognizes a predefined vocabulary of"
                 " static hand gestures in real time. It is not intended"
                 " to perform continuous natural-language sign-language"
                 " translation.")

# How-it-works steps (single source, values from src/config.py)
HOW_IT_WORKS = [
    "Read one webcam frame and mirror it exactly once, before detection.",
    "MediaPipe Hand Landmarker finds up to 2 hands (21 landmarks each).",
    f"build_sample() normalizes the landmarks into a {MAX_FEATURE_LENGTH}"
    "-dimensional feature vector - the same function that created the"
    " training dataset.",
    "Random Forest Classifier predicts the gesture and its confidence;"
    f" below {MIN_PREDICTION_CONFIDENCE:.2f} the display shows"
    ' "Uncertain" instead of a gesture.',
    f"The displayed label is the majority of the last"
    f" {PREDICTION_VOTE_WINDOW} frames, and SentenceBuilder turns it into"
    " words, SPACE and DELETE.",
]

# ---------------------------------------------------------------------------
# Page CSS - one dark surface, 1 accent, borders not shadows, no gradients.
# ---------------------------------------------------------------------------
PAGE_CSS = """
<style>
/* tokens + page surface ------------------------------------------------ */
:root {
    --sb-bg: #0B0D10;
    --sb-surface: #11151A;
    --sb-surface-2: #151A21;
    --sb-border: #252B33;
    --sb-accent: #4F8CE8;
    --sb-ok: #35B779;
    --sb-warn: #E5B84B;
    --sb-err: #E05252;
    --sb-text: #E3E6EB;
    --sb-text-2: #C6CBD4;
    --sb-muted: #98A1AE;
    --sb-dim: #7C8493;
    --sb-mono: Consolas, Menlo, "Courier New", monospace;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stToolbar"] {
    background-color: var(--sb-bg) !important;
}
/* the decorative gradient bar is a gradient - it does not belong here */
[data-testid="stDecoration"] { display: none !important; }
/* the fixed header bar is 60px tall - keep the page content below it */
div.block-container { padding-top: 4rem; padding-bottom: 2.5rem; }

/* header -------------------------------------------------------------- */
.sb-header {
    border-left: 3px solid var(--sb-accent);
    padding: 0.15rem 0 0.25rem 0.9rem;
}
.sb-rule {
    border-top: 1px solid var(--sb-border);
    margin-top: 0.3rem;
}
.sb-title {
    font-size: 1.72rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--sb-text);
    line-height: 1.15;
}
.sb-sub {
    font-size: 0.96rem;
    font-weight: 500;
    color: var(--sb-text-2);
    margin-top: 0.22rem;
}
.sb-tagline {
    font-size: 0.88rem;
    color: var(--sb-muted);
    margin-top: 0.32rem;
}
.sb-scope {
    font-size: 0.76rem;
    color: var(--sb-dim);
    margin-top: 0.4rem;
}

/* system status pill (right of the header) ---------------------------- */
.sb-statusbar {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    min-height: 2.2rem;
    padding-top: 0.3rem;
}
.sb-status {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--sb-muted);
    background: var(--sb-surface);
    border: 1px solid var(--sb-border);
    border-radius: 999px;
    padding: 0.3rem 0.75rem;
}
.sb-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--sb-muted);
}
.sb-status-on {
    color: var(--sb-ok);
    border-color: rgba(53, 183, 121, 0.45);
}
.sb-status-on .sb-dot { background: var(--sb-ok); }
.sb-status-error {
    color: var(--sb-err);
    border-color: rgba(224, 82, 82, 0.45);
}
.sb-status-error .sb-dot { background: var(--sb-err); }

/* section dividers ----------------------------------------------------- */
.sb-section {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--sb-muted);
    border-bottom: 1px solid var(--sb-border);
    padding-bottom: 0.45rem;
    margin-top: 1.7rem;
    margin-bottom: 0.9rem;
}
.sb-section-first { margin-top: 1.15rem; }
.sb-section-sub {
    font-size: 0.85rem;
    color: var(--sb-muted);
    margin-top: -0.55rem;
    margin-bottom: 0.75rem;
}
.sb-panel-title {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--sb-text-2);
    /* same height as the Start/Stop control on the left, so both
       workspace cards start on the same line */
    min-height: 2.6rem;
    display: flex;
    align-items: center;
}

/* cards ---------------------------------------------------------------- */
.sb-card {
    background: var(--sb-surface);
    border: 1px solid var(--sb-border);
    border-radius: 8px;
    padding: 1.05rem 1.1rem;
}

/* live video ----------------------------------------------------------- */
.sb-video {
    display: block;
    width: 100%;
    background: #000;
    border: 1px solid var(--sb-border);
    border-radius: 6px;
}
.sb-video-off {
    height: 22rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--sb-surface-2);
    border: 1px dashed var(--sb-border);
    border-radius: 6px;
    color: var(--sb-muted);
    font-size: 0.88rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.sb-video-note {
    margin-top: 0.75rem;
    color: var(--sb-dim);
    font-size: 0.78rem;
    line-height: 1.5;
    max-width: 110ch;
}

/* current gesture ------------------------------------------------------- */
.sb-gesture {
    font-family: var(--sb-mono);
    font-size: 2rem;
    font-weight: 650;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    line-height: 1.12;
    word-break: break-word;
    margin-bottom: 0.85rem;
}
.sb-gesture-ok   { color: var(--sb-ok); }
.sb-gesture-warn { color: var(--sb-warn); }
.sb-gesture-idle { color: var(--sb-muted); }

/* confidence ------------------------------------------------------------ */
.sb-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--sb-muted);
}
.sb-conf-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-top: 0.3rem;
}
.sb-conf-value {
    font-size: 1.35rem;
    font-weight: 650;
    font-family: var(--sb-mono);
    color: var(--sb-text);
}
.sb-conf-hint {
    font-size: 0.74rem;
    font-family: var(--sb-mono);
    color: var(--sb-dim);
}
.sb-bar {
    height: 8px;
    background: var(--sb-surface-2);
    border: 1px solid var(--sb-border);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 0.45rem;
}
.sb-fill { height: 100%; }
.sb-fill-ok   { background: var(--sb-ok); }
.sb-fill-warn { background: var(--sb-warn); }

/* hand + state line ------------------------------------------------------ */
.sb-handline {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
    font-size: 0.82rem;
    color: var(--sb-text-2);
    background: var(--sb-surface-2);
    border: 1px solid var(--sb-border);
    border-radius: 5px;
    padding: 0.42rem 0.6rem;
    margin-top: 0.8rem;
}
.sb-sep { color: #3A424E; }
.sb-state { color: var(--sb-ok); }
.sb-state-warn { color: var(--sb-warn); }
.sb-state-idle { color: var(--sb-muted); }

.sb-dim {
    font-size: 0.76rem;
    color: var(--sb-dim);
    line-height: 1.45;
    margin-top: 0.5rem;
}

/* runtime strip ---------------------------------------------------------- */
.sb-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem 0.95rem;
    font-family: var(--sb-mono);
    font-size: 0.8rem;
    color: var(--sb-muted);
    background: var(--sb-surface-2);
    border: 1px solid var(--sb-border);
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
}
.sb-strip b { color: var(--sb-text); font-weight: 650; }
.sb-strip-sep { color: #333A45; }

/* sentence builder -------------------------------------------------------- */
.sb-sentence {
    font-family: var(--sb-mono);
    font-size: 1.3rem;
    line-height: 1.55;
    color: var(--sb-text);
    background: var(--sb-surface);
    border: 1px solid var(--sb-border);
    border-radius: 8px;
    padding: 1.15rem 1.2rem;
    min-height: 6.5rem;
    white-space: pre-wrap;
    word-break: break-word;
}
.sb-sentence-empty { color: var(--sb-dim); font-size: 1.05rem; }
.sb-sentence-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 1.4rem;
    font-family: var(--sb-mono);
    font-size: 0.78rem;
    color: var(--sb-muted);
    margin-top: 0.55rem;
}
.sb-control-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--sb-muted);
    margin-top: 0.9rem;
    margin-bottom: 0.35rem;
}

/* project overview strip --------------------------------------------------- */
.sb-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
}
.sb-metric {
    flex: 1 1 160px;
    min-width: min(160px, 100%);
    background: var(--sb-surface);
    border: 1px solid var(--sb-border);
    border-radius: 6px;
    padding: 0.7rem 0.8rem;
}
.sb-metric-value {
    font-family: var(--sb-mono);
    font-size: 1.3rem;
    font-weight: 650;
    color: var(--sb-text);
    line-height: 1.15;
}
.sb-metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--sb-muted);
    margin-top: 0.3rem;
}

/* supported gestures grid --------------------------------------------------- */
.sb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(195px, 100%), 1fr));
    gap: 0.6rem;
}
.sb-chip {
    background: var(--sb-surface);
    border: 1px solid var(--sb-border);
    border-radius: 6px;
    padding: 0.7rem 0.8rem;
}
.sb-chip-name {
    font-family: var(--sb-mono);
    font-size: 0.95rem;
    font-weight: 650;
    letter-spacing: 0.04em;
    color: var(--sb-accent);
}
.sb-chip-meaning {
    font-size: 0.8rem;
    color: var(--sb-text-2);
    margin-top: 0.3rem;
}
.sb-chip-tag {
    display: inline-block;
    margin-top: 0.5rem;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--sb-warn);
    border: 1px solid rgba(229, 184, 75, 0.4);
    border-radius: 3px;
    padding: 0.12rem 0.42rem;
}
.sb-chip-control { border-color: rgba(229, 184, 75, 0.35); }

/* expandable reference sections -------------------------------------------- */
div[data-testid="stExpander"] {
    background: var(--sb-surface);
    border: 1px solid var(--sb-border) !important;
    border-radius: 6px;
    margin-bottom: 0.55rem;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary span {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--sb-text-2) !important;
}

/* key/value + steps + scope text ------------------------------------------- */
.sb-kv {
    border: 1px solid var(--sb-border);
    border-radius: 6px;
    overflow: hidden;
    font-size: 0.85rem;
}
.sb-kv .row { display: flex; border-bottom: 1px solid var(--sb-border); }
.sb-kv .row:last-child { border-bottom: none; }
.sb-kv .k {
    width: 42%;
    padding: 0.5rem 0.65rem;
    color: var(--sb-muted);
    background: var(--sb-surface-2);
    font-size: 0.78rem;
}
.sb-kv .v {
    padding: 0.5rem 0.65rem;
    color: var(--sb-text);
    font-family: var(--sb-mono);
    word-break: break-word;
}
.sb-steps {
    font-size: 0.85rem;
    color: var(--sb-text-2);
    line-height: 1.55;
    padding-left: 1.15rem;
}
.sb-steps li { margin-bottom: 0.45rem; }
.sb-scope-text {
    font-size: 0.88rem;
    color: var(--sb-text-2);
    line-height: 1.6;
}

/* controls ------------------------------------------------------------------ */
[data-testid="stButton"] button {
    border: 1px solid var(--sb-border) !important;
    background: var(--sb-surface-2) !important;
    color: var(--sb-text-2) !important;
    border-radius: 5px;
    /* fixed height = .sb-panel-title min-height, so the two
       workspace cards start on the same line */
    height: 2.6rem;
    padding: 0.4rem 0.6rem;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    transition: border-color 0.15s ease, color 0.15s ease;
}
[data-testid="stButton"] button:hover:not(:disabled) {
    border-color: var(--sb-accent) !important;
    color: var(--sb-text) !important;
}
[data-testid="stButton"] button:focus { outline: none; }
[data-testid="stButton"] button:disabled { opacity: 0.4; }
button[data-testid="stButtonPrimary"],
[data-testid="stButtonPrimary"] button,
[data-testid="stButton"] button[kind="primary"] {
    background: var(--sb-accent) !important;
    border-color: var(--sb-accent) !important;
    color: #0B0D10 !important;
}
button[data-testid="stButtonPrimary"]:hover,
[data-testid="stButtonPrimary"] button:hover,
[data-testid="stButton"] button[kind="primary"]:hover:not(:disabled) {
    background: #5D99EC !important;
    border-color: #5D99EC !important;
    color: #0B0D10 !important;
}

/* captions, warnings, errors ----------------------------------------------- */
div[data-testid="stCaptionContainer"] {
    color: var(--sb-dim);
    font-size: 0.78rem;
    line-height: 1.5;
    /* keep full-width notes on a readable measure */
    max-width: 110ch;
}
div[data-testid="stError"] {
    background: rgba(224, 82, 82, 0.1);
    border: 1px solid rgba(224, 82, 82, 0.5);
    color: #F1B4B4;
}
div[data-testid="stWarning"] {
    background: rgba(229, 184, 75, 0.08);
    border: 1px solid rgba(229, 184, 75, 0.4);
    color: #EBD6A2;
}

/* responsive: stack the columns instead of squeezing them -------------------- */
/* both workspace panels share the row height, so the recognition card
   ends on the same line as the camera panel instead of leaving a hole.
   Streamlit nests extra stVerticalBlock layers inside a column, so the
   rule has to walk every level down to the card container. */
div[data-testid="stColumn"] [data-testid="stVerticalBlock"] {
    height: 100%;
    display: flex;
    flex-direction: column;
}
div[data-testid="stColumn"] [data-testid="stVerticalBlock"] > * {
    min-height: 0;
}
div[data-testid="stColumn"] [data-testid="stVerticalBlock"] > *:last-child {
    flex: 1 1 auto;
}
/* recognition card: streamlit wraps markdown in three nested divs, so the
   whole wrapper chain has to fill the stretched element container for the
   card's bottom edge to land on the same line as the camera panel */
div[data-testid="stColumn"]
        [data-testid="stElementContainer"]:has(.sb-card--center)
        [data-testid="stMarkdown"],
div[data-testid="stColumn"]
        [data-testid="stElementContainer"]:has(.sb-card--center)
        [data-testid="stMarkdown"] > *,
div[data-testid="stColumn"]
        [data-testid="stElementContainer"]:has(.sb-card--center)
        [data-testid="stMarkdownContainer"] {
    height: 100%;
}
.sb-card { height: 100%; }
.sb-card > .sb-video-off { height: auto; min-height: 22rem; }
/* the recognition card fills the shared row height: center its content
   vertically instead of leaving an empty lower half */
.sb-card--center {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

@media (max-width: 860px) {
    div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        flex-basis: 100% !important;
        min-width: 100%;
    }
    .sb-gesture { font-size: 1.6rem; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Cached resources: loaded once per process, never per frame.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_worker():
    """One process-wide CameraWorker (shared feed, one camera)."""
    return browser_camera.BrowserCameraProcessor()


@st.cache_resource(show_spinner=False)
def get_model():
    """(Random Forest, class names); load_classifier() validates both."""
    return load_classifier()


@st.cache_resource(show_spinner=False)
def get_detector():
    """MediaPipe hand landmarker (Phase 2.1: up to 2 hands)."""
    return HandDetector(model_path=HAND_MODEL_PATH, max_hands=2,
                        detection_confidence=0.5, tracking_confidence=0.5)


@st.cache_data
def get_metrics():
    """Phase 4 evaluation numbers (real, measured) or None when missing."""
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Start helper (runs in the page thread: model + detector already cached)
# ---------------------------------------------------------------------------
def start_worker() -> str:
    """Initialize cached recognition resources for the browser callback."""
    try:
        model, class_names = get_model()
        detector = get_detector()
        get_worker().configure(model, class_names, detector)
    except Exception as error:  # noqa: BLE001 - show initialization issue
        return f"Could not initialize recognition: {error}"
    return ""


# ---------------------------------------------------------------------------
# Presentation helpers: pure formatting of values computed elsewhere.
# ---------------------------------------------------------------------------
def present_label(value) -> str:
    """Browser-facing label: I_LOVE_YOU -> I LOVE YOU.

    Presentation ONLY - the model class, the worker snapshot and the
    SentenceBuilder keep the stored label with its underscore.
    """
    return str(value).replace("_", " ")


def present_sentence(sentence: str) -> str:
    """Same rule for the built message (collapses whitespace for reading)."""
    return present_label(" ".join(str(sentence).split()))


def worker_state() -> str:
    """ready | active | error - read from the worker, never invented."""
    worker = get_worker()
    if worker.error and not (worker.running and worker.is_alive()):
        return "error"
    return "active" if worker.running and worker.is_alive() else "ready"


def gesture_css_class(prediction):
    if prediction == NO_HAND_LABEL:
        return "sb-gesture-idle"
    if prediction in ("Uncertain", "Sample rejected"):
        return "sb-gesture-warn"
    return "sb-gesture-ok"


def header_html():
    return (
        '<div class="sb-header">'
        '<div class="sb-title">SignBridge AI</div>'
        '<div class="sb-sub">Real-Time Gesture Recognition</div>'
        '<div class="sb-tagline">Convert predefined hand gestures into'
        " readable text.</div>"
        f'<div class="sb-scope">{html.escape(SCOPE_NOTE)}</div>'
        "</div>")


def status_html(state: str):
    if state == "active":
        css, label = "sb-status sb-status-on", "Camera active"
    elif state == "error":
        css, label = "sb-status sb-status-error", "Camera error"
    else:
        css, label = "sb-status", "Ready"
    return (f'<div class="sb-statusbar"><span class="{css}">'
            f'<span class="sb-dot"></span>{label}</span></div>')


def confidence_block_html(confidence):
    """Label + value row + progress bar as ONE markdown block."""
    if confidence is None:
        value, hint, fill = "&ndash;", "", ""
    else:
        value = f"{confidence * 100:.1f}%"
        fill_css = ("sb-fill-ok" if confidence >= MIN_PREDICTION_CONFIDENCE
                    else "sb-fill-warn")
        width = max(0.0, min(100.0, confidence * 100.0))
        fill = f'<div class="sb-fill {fill_css}" style="width: {width:.1f}%;"></div>'
    return (
        '<div class="sb-label">Confidence</div>'
        f'<div class="sb-conf-row"><span class="sb-conf-value">{value}</span>'
        f'<span class="sb-conf-hint">threshold '
        f"{MIN_PREDICTION_CONFIDENCE:.2f}</span></div>"
        f'<div class="sb-bar">{fill}</div>')


def hand_status_html(snapshot, prediction) -> str:
    """Hand count + state line - values come from the snapshot only.

    The worker exposes the NUMBER of detected hands; no hand side is
    reported by the pipeline, so none is shown here.
    """
    hands = int(snapshot.get("hands", 0) or 0)
    if not hands:
        text = ("Camera is off" if not snapshot.get("running")
                else "Waiting for a hand in view")
        return (f'<div class="sb-handline"><span>{text}</span></div>')
    count = f"{hands} hand" + ("" if hands == 1 else "s") + " detected"
    if prediction == NO_HAND_LABEL:
        state, css = "Waiting for a hand", "sb-state-idle"
    elif prediction == "Uncertain":
        state, css = "Below confidence threshold", "sb-state-warn"
    elif prediction == "Sample rejected":
        state, css = "Sample rejected", "sb-state-warn"
    else:
        state, css = "Recognized", "sb-state"
    return (f'<div class="sb-handline"><span>{count}</span>'
            f'<span class="sb-sep">|</span>'
            f'<span class="{css}">{html.escape(state)}</span></div>')


def notes_html(snapshot, is_running) -> str:
    """Debug facts the pipeline already reports, plus the confidence note."""
    parts = []
    if snapshot.get("raw_text"):
        parts.append('<div class="sb-dim">This frame: '
                     + html.escape(snapshot["raw_text"]) + "</div>")
    if snapshot.get("note"):
        parts.append('<div class="sb-dim">'
                     + html.escape(snapshot["note"]) + "</div>")
    if is_running:
        parts.append(
            f'<div class="sb-dim">Displayed label = majority of the last'
            f" {PREDICTION_VOTE_WINDOW} frames (vote window).</div>")
    parts.append(
        '<div class="sb-dim">Confidence is the model probability for the'
        " displayed class - it is not an accuracy measure.</div>")
    return "".join(parts)


def stats_strip_html(snapshot) -> str:
    """FPS | prediction latency | hands | frames - snapshot values only."""
    fps = snapshot.get("fps")
    latency = snapshot.get("latency")
    hands = int(snapshot.get("hands", 0) or 0)
    frames = int(snapshot.get("frames", 0) or 0)

    def item(value, unit):
        return f'<span class="sb-strip-item"><b>{value}</b> {unit}</span>'

    sep = '<span class="sb-strip-sep">|</span>'
    return ('<div class="sb-strip">'
            + item(f"{fps:.1f}" if fps is not None else "&ndash;", "FPS")
            + sep
            + item(f"{latency:.1f}" if latency is not None else "&ndash;",
                   "ms prediction")
            + sep
            + item(str(hands), "hand" if hands == 1 else "hands")
            + sep
            + item(str(frames), "frames")
            + "</div>")


def sentence_block_html(sentence: str) -> str:
    shown = present_sentence(sentence)
    if shown:
        return f'<div class="sb-sentence">{html.escape(shown)}</div>'
    return ('<div class="sb-sentence sb-sentence-empty">'
            "Start signing to build your message...</div>")


def metrics_strip_html(metrics) -> str:
    """Project overview - every number read from evaluation/metrics.json
    and src/config.py (nothing invented for decoration)."""
    accuracy = metrics["metrics"]["accuracy"]
    items = [
        (str(len(SIGN_CLASSES)), "Gesture classes"),
        (str(metrics["dataset"]["features"]), "Features"),
        (f"{metrics['dataset']['samples']:,}", "Dataset samples"),
        (f"{accuracy * 100:.2f}%", "Held-out test accuracy"),
        (str(metrics["model"]["parameters"]["n_estimators"]),
         "Random forest trees"),
    ]
    cells = "".join(
        f'<div class="sb-metric"><div class="sb-metric-value">'
        f"{value}</div>"
        f'<div class="sb-metric-label">{label}</div></div>'
        for value, label in items)
    return f'<div class="sb-metrics">{cells}</div>'


def kv_html(pairs):
    rows = "".join(
        f'<div class="row"><div class="k">{html.escape(str(key))}</div>'
        f'<div class="v">{html.escape(str(value))}</div></div>'
        for key, value in pairs)
    return f'<div class="sb-kv">{rows}</div>'


def model_info_pairs(metrics):
    """Model facts - all read from evaluation/metrics.json + config."""
    params = metrics["model"]["parameters"]
    split = metrics["split"]
    counts = metrics["dataset"]["class_counts"]
    samples = metrics["dataset"]["samples"]
    per_class = samples // len(counts)
    accuracy = metrics["metrics"]["accuracy"]
    return [
        ("Classifier",
         f"Random Forest Classifier, {params['n_estimators']} trees"),
        ("Hand detection", "MediaPipe Hand Landmarker (21 per hand)"),
        ("Feature vector",
         f"{MAX_FEATURE_LENGTH} values (2 hands x 21 x 3)"),
        ("Gesture classes",
         f"{len(SIGN_CLASSES)} predefined static gestures"),
        ("Dataset", f"{samples:,} samples ({len(counts)} x {per_class})"),
        ("Train / test",
         f"{split['train_samples']:,} / {split['test_samples']:,}"
         f" ({split['test_size'] * 100:.0f}% held out)"),
        ("Held-out test accuracy", f"{accuracy * 100:.2f}%"),
    ]


def gestures_grid_html():
    """Responsive grid built from the SAME SIGN_CLASSES + meanings."""
    cards = []
    for label in SIGN_CLASSES:
        control = label in SENTENCE_CONTROLS
        css = "sb-chip sb-chip-control" if control else "sb-chip"
        tag = ('<span class="sb-chip-tag">sentence control</span>'
               if control else "")
        cards.append(
            f'<div class="{css}" data-class="{label}"'
            f' title="model class: {label}">'
            f'<div class="sb-chip-name">{html.escape(present_label(label))}'
            f"</div>"
            f'<div class="sb-chip-meaning">'
            f"{html.escape(GESTURE_MEANING[label])}</div>{tag}</div>")
    return f'<div class="sb-grid">{"".join(cards)}</div>'


def steps_html():
    items = "".join(f"<li>{html.escape(step)}</li>" for step in HOW_IT_WORKS)
    return f'<ol class="sb-steps">{items}</ol>'


# ---------------------------------------------------------------------------
# Page (main script runs once per load - live regions are fragments)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SignBridge AI - Real-Time Gesture Recognition",
                   layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

worker = get_worker()
worker.touch()
metrics = get_metrics()

# Live rate for every statistics/control region of the page. It is
# UNCONDITIONAL: the camera control lives in its own fragment, so the main
# script runs once per page load and cannot arm a refresh later (a fragment
# click never re-runs the app - verified). The video does not depend on this
# rate - it streams through the MJPEG feed - and an <img> whose markdown
# value is unchanged is not re-sent to the browser: React keeps the same
# node, so the feed connection survives every tick (verified).
run_every = STATS_INTERVAL_SECONDS

# --- header: identity on the left, real system status on the right ---------
head_col, status_col = st.columns([1, 0.26], gap="medium")

with head_col:
    st.markdown(header_html(), unsafe_allow_html=True)

with status_col:
    # Refreshed with the same rate as the other statistics, so the pill
    # always reflects the worker state (never a fake online flag).
    @st.fragment(run_every=run_every)
    def status_fragment():
        local_worker = get_worker()
        local_worker.touch()
        st.markdown(status_html(worker_state()), unsafe_allow_html=True)

    status_fragment()

# full-width rule: the header reads as one band across the page
st.markdown('<div class="sb-rule"></div>', unsafe_allow_html=True)

# --- live recognition workspace (camera 60% / recognition 40%) --------------
st.markdown('<div class="sb-section sb-section-first">Live Recognition'
            "</div>", unsafe_allow_html=True)
video_col, panel_col = st.columns([1.5, 1], gap="large")

with video_col:
    # WebRTC captures from the VISITOR'S browser. Unlike cv2.VideoCapture(0),
    # this works on Streamlit Community Cloud where the server has no webcam.
    @st.fragment(run_every=run_every)
    def camera_fragment():
        local_worker = get_worker()
        local_worker.touch()
        try:
            model, class_names = get_model()
            detector = get_detector()
            local_worker.configure(model, class_names, detector)
        except Exception as error:  # noqa: BLE001 - surface setup errors
            local_worker.set_running(False)
            st.error(f"Could not initialize recognition: {error}")
            return

        st.markdown('<div class="sb-panel-title">Browser Webcam</div>',
                    unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="signbridge-browser-webcam",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=local_worker.recv,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        local_worker.set_running(bool(ctx.state.playing))
        if ctx.state.playing:
            st.caption("Camera is streaming from this browser. Allow camera access "
                       "when prompted; gestures are processed on the app server.")
        else:
            st.caption("Press START above and allow browser camera permission. "
                       "HTTPS is required for deployed webcam access.")
        if local_worker.error and not ctx.state.playing:
            st.error(local_worker.error)

    camera_fragment()

with panel_col:
    # The gesture card is its own live region: it reads worker.snapshot on
    # every tick, so it can never show a state the worker already left.
    @st.fragment(run_every=run_every)
    def gesture_fragment():
        local_worker = get_worker()
        local_worker.touch()
        snap = local_worker.snapshot
        prediction = snap.get("prediction") or NO_HAND_LABEL
        confidence = snap.get("confidence")
        is_running = snap.get("running", False)

        st.markdown('<div class="sb-panel-title">Current Gesture</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="sb-card sb-card--center">'
            f'<div class="sb-gesture {gesture_css_class(prediction)}"'
            f' data-class="{html.escape(str(prediction))}">'
            f"{html.escape(present_label(prediction))}</div>"
            f"{confidence_block_html(confidence)}"
            f"{hand_status_html(snap, prediction)}"
            f"{notes_html(snap, is_running)}</div>",
            unsafe_allow_html=True)

    gesture_fragment()

# --- runtime strip + sentence builder (one live region, full width) --------
@st.fragment(run_every=run_every)
def output_fragment():
    local_worker = get_worker()
    local_worker.touch()
    snap = local_worker.snapshot
    st.markdown(stats_strip_html(snap), unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Sentence Builder</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sb-section-sub">Build a message using'
                " recognized gestures.</div>", unsafe_allow_html=True)

    sentence = local_worker.builder.sentence
    st.markdown(sentence_block_html(sentence), unsafe_allow_html=True)

    word_count = len(sentence.split())
    last_action = local_worker.last_action or "-"
    st.markdown(
        f'<div class="sb-sentence-meta"><span>Words: {word_count}</span>'
        f"<span>Last action: {html.escape(str(last_action))}</span></div>",
        unsafe_allow_html=True)

    st.markdown('<div class="sb-control-label">Message controls</div>',
                unsafe_allow_html=True)
    edit_columns = st.columns(3, gap="small")
    can_edit = bool(sentence)
    can_space = can_edit and not sentence.endswith(" ")
    if edit_columns[0].button("SPACE", disabled=not can_space,
                              width="stretch", key="sentence_space"):
        local_worker.manual("SPACE")
    if edit_columns[1].button("DELETE", disabled=not can_edit,
                              width="stretch", key="sentence_delete"):
        local_worker.manual("DELETE")
    if edit_columns[2].button("Clear", disabled=not can_edit,
                              width="stretch", key="sentence_clear"):
        local_worker.manual("CLEAR")
    st.caption("The buttons edit the message directly. The SPACE and "
               "DELETE gestures do the same while the camera runs; "
               "gestures are edge-triggered, so a held sign fires once "
               "and re-arms only after it is released.")


output_fragment()

# --- project overview (verified numbers only) ------------------------------
st.markdown('<div class="sb-section">Project Overview</div>',
            unsafe_allow_html=True)
if metrics is None:
    st.warning("evaluation/metrics.json not found - train the model "
               "first: python scripts/train_model.py")
else:
    st.markdown(metrics_strip_html(metrics), unsafe_allow_html=True)

# --- supported gestures: grid instead of a raw data table -------------------
st.markdown('<div class="sb-section">Supported Gestures</div>',
            unsafe_allow_html=True)
st.markdown(gestures_grid_html(), unsafe_allow_html=True)
st.caption("SPACE and DELETE are controls of this project's Sentence "
           "Builder (they edit the message); the other ten entries are "
           "the predefined gesture classes the model was trained on.")

# --- reference sections: collapsed so they do not dominate the page --------
with st.expander("Model Details"):
    if metrics is None:
        st.warning("evaluation/metrics.json not found - train the model "
                   "first: python scripts/train_model.py")
    else:
        st.markdown(kv_html(model_info_pairs(metrics)),
                    unsafe_allow_html=True)

with st.expander("How It Works"):
    st.markdown(steps_html(), unsafe_allow_html=True)

with st.expander("Project Scope"):
    st.markdown(f'<p class="sb-scope-text">{html.escape(PROJECT_SCOPE)}'
                "</p>", unsafe_allow_html=True)

st.caption("Phase 8 web interface. The recognition pipeline and the "
           "Sentence Builder are unchanged from Phases 5/6; there is no "
           "speech output (Phase 7 is a separate phase).")
