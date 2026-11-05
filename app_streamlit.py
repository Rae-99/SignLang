"""
=============================================================================
ASL TO SPEECH TRANSLATION — STREAMLIT WEB APPLICATION
=============================================================================
A modern, responsive Streamlit dashboard for real-time American Sign Language
(ASL) detection, gesture tracking, and text-to-speech translation.

This module reuses core classification, motion tracking, audio dispatch, and
visual HUD rendering directly from `inference_tts.py` without modifying any
existing project files.
=============================================================================
"""

import os
import time
import pickle
import threading
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit as st
from PIL import Image

# Import core functionality directly from existing inference_tts.py
from inference_tts import (
    speak_async,
    speech_queue,
    normalize_landmarks,
    DynamicGestureTracker,
    print_result_callback,
    draw_translucent_panel,
    draw_hand_skeleton,
    draw_corner_bbox,
    draw_progress_ring,
    draw_hud,
    draw_title_bar,
    COLOR_PANEL,
    COLOR_PANEL_EDGE,
    COLOR_ACCENT,
    COLOR_ACCENT_2,
    COLOR_WARN,
    COLOR_OK,
    COLOR_TEXT,
    COLOR_TEXT_DIM,
    HAND_CONNECTIONS,
    FINGER_COLOR_BY_JOINT,
    run_translation_system,
)
import inference_tts


# =============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & THEME STYLING
# =============================================================================
st.set_page_config(
    page_title="ASL -> Speech AI Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit reserves "C" for clearing caches. Stop that browser shortcut outside
# text inputs so "C" can still be used by the native OpenCV controls.
st.markdown(
    """
    <script>
    window.addEventListener("keydown", function(event) {
        const target = event.target;
        const isTextInput = target instanceof HTMLElement &&
            (target.closest("input, textarea") || target.isContentEditable);
        const isClearCacheShortcut = event.key &&
            event.key.toLowerCase() === "c" &&
            !event.ctrlKey && !event.metaKey && !event.altKey;

        if (isClearCacheShortcut && !isTextInput) {
            event.stopImmediatePropagation();
            event.preventDefault();
        }
    }, true);
    </script>
    """,
    unsafe_allow_html=True,
)

# Custom High-Tech Dark / Cyberpunk CSS styling
st.markdown(
    """
    <style>
    /* Main body background & typography */
    .stApp {
        background: radial-gradient(circle at top left, #121824 0%, #0a0d14 100%);
        color: #e6edf3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Top Header Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(20, 26, 40, 0.85) 0%, rgba(12, 16, 24, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    .badge-ok {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.3);
    }
    .badge-info {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-warn {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    /* Cards & Display Panels */
    .metric-card {
        background: rgba(18, 24, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .char-display {
        font-size: 3.4rem;
        font-weight: 900;
        color: #38bdf8;
        text-align: center;
        line-height: 1;
        padding: 10px 0;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.45);
    }
    .word-display {
        font-size: 1.6rem;
        font-weight: 700;
        color: #fbbf24;
        font-family: monospace;
        letter-spacing: 2px;
        min-height: 2.2rem;
    }
    .sentence-display {
        font-size: 1.15rem;
        color: #f1f5f9;
        min-height: 2.5rem;
        line-height: 1.5;
        background: rgba(10, 14, 22, 0.6);
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px dashed rgba(255, 255, 255, 0.12);
    }

    /* Button Customization */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.25);
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8 0%, #c084fc 100%);
    }

    /* Reference Card Grid */
    .ref-letter-card {
        background: rgba(22, 30, 48, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        transition: transform 0.15s ease;
    }
    .ref-letter-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 2. RESOURCE CACHING & MODELS
# =============================================================================
@st.cache_resource(show_spinner="Loading ASL Classifier Model...")
def load_classifier(model_path="asl_model.p"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found in project directory.")
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    return classifier


@st.cache_resource(show_spinner="Initializing MediaPipe Hand Landmarker Task...")
def get_task_path(task_path="hand_landmarker.task"):
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"MediaPipe task asset '{task_path}' not found.")
    return task_path


def open_robust_camera(selected_choice):
    """
    Intelligently opens the camera using the appropriate Windows video backend
    (DirectShow or Media Foundation) with retry warmup frames.
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    if selected_choice == "Auto Detect":
        candidates = [1, 0, 2]
    else:
        preferred = int(selected_choice)
        candidates = [preferred] + [i for i in [1, 0, 2] if i != preferred]

    for idx in candidates:
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                # Warm up read (some webcams require a couple of frames to negotiate exposure)
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        return cap, idx
                cap.release()
            except Exception:
                pass
    return None, None


# =============================================================================
# 3. SESSION STATE INITIALIZATION
# =============================================================================
if "current_char" not in st.session_state:
    st.session_state.current_char = ""
if "current_word" not in st.session_state:
    st.session_state.current_word = ""
if "full_sentence" not in st.session_state:
    st.session_state.full_sentence = []
if "transcript_history" not in st.session_state:
    st.session_state.transcript_history = []
if "desktop_hud_running" not in st.session_state:
    st.session_state.desktop_hud_running = False
if "desktop_hud_thread" not in st.session_state:
    st.session_state.desktop_hud_thread = None


# =============================================================================
# 4. SIDEBAR SETTINGS & CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")

    # Camera Selection
    camera_options = [
        "Camera 1 (DirectShow - Default)",
        "Camera 0 (Media Foundation)",
        "Auto Detect",
        "Camera 2",
    ]
    camera_choice = st.selectbox(
        "📹 Video Source",
        options=camera_options,
        index=0,
        help="Camera 1 with DirectShow is the default for this system. If unavailable, try Camera 0 or Auto Detect.",
    )
    cam_idx_map = {
        "Camera 1 (DirectShow - Default)": 1,
        "Camera 0 (Media Foundation)": 0,
        "Auto Detect": "Auto Detect",
        "Camera 2": 2,
    }
    selected_camera_val = cam_idx_map[camera_choice]

    st.markdown("---")
    st.markdown("### 🎛️ Detection & Timing")

    hold_duration = st.slider(
        "⏱️ Sign Hold Duration (s)",
        min_value=0.3,
        max_value=2.5,
        value=1.0,
        step=0.1,
        help="Time needed to hold a static gesture steadily before registering.",
    )

    min_detection_conf = st.slider(
        "🎯 Min Hand Detection Confidence",
        min_value=0.2,
        max_value=0.9,
        value=0.4,
        step=0.05,
    )

    min_tracking_conf = st.slider(
        "🔍 Min Hand Tracking Confidence",
        min_value=0.2,
        max_value=0.9,
        value=0.4,
        step=0.05,
    )

    dynamic_cooldown = st.slider(
        "⚡ Dynamic Gesture Cooldown (s)",
        min_value=0.5,
        max_value=2.5,
        value=1.2,
        step=0.1,
        help="Minimum delay between successive 'J' or 'Z' dynamic strokes.",
    )

    st.markdown("---")
    st.markdown("### 🎨 Visual HUD Overlays")

    show_skeleton = st.checkbox("Show Hand Skeleton (Neon)", value=True)
    show_bbox = st.checkbox("Show AR Corner Box", value=True)
    show_hud_panel = st.checkbox("Show In-Feed HUD Banner", value=True)
    mirror_camera = st.checkbox("Mirror Camera Horizontally", value=True)

    st.markdown("---")
    st.markdown("### 🔊 Text-To-Speech")

    voice_enabled = st.checkbox("Enable Audio Speech Output", value=True)
    speak_letters = st.checkbox("Vocalize Individual Letters", value=True)

    if st.button("🧪 Test TTS Voice", use_container_width=True):
        speak_async("American Sign Language Audio Engine Active")
        st.toast("Playing test audio sample...", icon="🔊")


# =============================================================================
# 5. HEADER SECTION
# =============================================================================
try:
    classifier = load_classifier("asl_model.p")
    task_asset = get_task_path("hand_landmarker.task")
    system_ready = True
except Exception as e:
    system_ready = False
    load_err = str(e)

st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">🤟 ASL -> Audio Real-Time Translator</h1>
        <p class="hero-subtitle">
            Deep-learning powered American Sign Language translation with dynamic gesture velocity tracking and instant voice synthesis.
        </p>
        <div style="margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap;">
            <span class="badge badge-ok">● ML Model: SVC Active</span>
            <span class="badge badge-info">● MediaPipe Landmarker: Loaded</span>
            <span class="badge badge-info">● Dynamic Strokes: J / Z Active</span>
            <span class="badge badge-ok">● Thread-Safe TTS: Ready</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not system_ready:
    st.error(f"Failed to load project models or assets: {load_err}")
    st.stop()


# =============================================================================
# 6. TABBED NAVIGATION INTERFACE
# =============================================================================
tab_live, tab_snapshot, tab_desktop, tab_guide = st.tabs(
    [
        "🎥 Live Video Translator",
        "📸 Snapshot Detector",
        "🚀 Standalone Desktop HUD",
        "📖 ASL Gesture Reference",
    ]
)


# =============================================================================
# TAB 1: REAL-TIME STREAMLIT CAMERA TRANSLATOR
# =============================================================================
with tab_live:
    col_video, col_controls = st.columns([1.6, 1.0], gap="large")

    with col_controls:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">CURRENT DETECTED SIGN</div>
                <div id="char-target" class="char-display">-</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        char_progress_bar = st.progress(0.0, text="Hold sign steadily...")

        st.markdown("#### 📝 Translation Buffer")

        word_placeholder = st.empty()
        word_placeholder.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">CURRENT WORD</div>
                <div class="word-display">{st.session_state.current_word or "..."}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sentence_placeholder = st.empty()
        sentence_str = " ".join(st.session_state.full_sentence) if st.session_state.full_sentence else "..."
        sentence_placeholder.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">FULL SENTENCE</div>
                <div class="sentence-display">{sentence_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### ⚡ Quick Actions")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔊 Speak Word", use_container_width=True):
                if st.session_state.current_word:
                    speak_async(st.session_state.current_word)
                    st.toast(f"Spoke: {st.session_state.current_word}")
            if st.button("⌫ Backspace", use_container_width=True):
                if st.session_state.current_word:
                    st.session_state.current_word = st.session_state.current_word[:-1]
                elif st.session_state.full_sentence:
                    last_w = st.session_state.full_sentence.pop()
                    if len(last_w) > 1:
                        st.session_state.full_sentence.append(last_w[:-1])
                st.rerun()

        with c2:
            if st.button("🔊 Speak Sentence", use_container_width=True):
                if st.session_state.full_sentence:
                    full_txt = " ".join(st.session_state.full_sentence)
                    speak_async(full_txt)
                    st.toast(f"Spoke sentence: {full_txt}")
            if st.button("🗑️ Clear Word", use_container_width=True):
                st.session_state.current_word = ""
                st.rerun()

        if st.button("🧹 Clear All (Word & Sentence)", use_container_width=True):
            st.session_state.current_word = ""
            st.session_state.full_sentence.clear()
            st.rerun()

        with st.expander("📜 Translation Transcript History", expanded=False):
            if st.session_state.transcript_history:
                for item in reversed(st.session_state.transcript_history[-10:]):
                    st.caption(f"`{item['time']}` **[{item['type']}]** {item['content']}")
            else:
                st.caption("No spoken events yet.")

    with col_video:
        # Camera Activation Controls
        cam_col1, cam_col2 = st.columns([1, 1])
        with cam_col1:
            live_active = st.toggle("🔴 Start Live Camera Stream", value=False)
        with cam_col2:
            st.caption("Hold sign 1s to register. Sign 'space' to commit word.")

        video_frame_placeholder = st.empty()

        if live_active:
            with st.spinner("Connecting to webcam device..."):
                cap, active_idx = open_robust_camera(selected_camera_val)

            if cap is None:
                st.error(
                    f"⚠️ Could not access camera ({camera_choice}).\n\n"
                    f"**Quick Fixes:**\n"
                    f"1. Make sure no other browser tab or application (Zoom, Teams, Camera app) is using the webcam.\n"
                    f"2. In Tab 2, ensure the snapshot camera is deactivated.\n"
                    f"3. In the sidebar, try switching **Video Source** to **Camera 0** or **Auto Detect**."
                )
            else:
                base_options = python.BaseOptions(model_asset_path=task_asset)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.LIVE_STREAM,
                    num_hands=1,
                    min_hand_detection_confidence=min_detection_conf,
                    min_tracking_confidence=min_tracking_conf,
                    result_callback=print_result_callback,
                )

                gesture_tracker = DynamicGestureTracker()
                prev_index_pt = None
                prev_pinky_pt = None

                # Debouncing & Timing
                current_char = ""
                char_hold_start = 0.0
                registered_flag = False
                last_dynamic_commit_time = 0.0
                flash_label = None
                flash_started_at = 0.0
                FLASH_DURATION = 0.45
                consecutive_fails = 0

                try:
                    with vision.HandLandmarker.create_from_options(options) as landmarker:
                        while live_active:
                            ret, frame = cap.read()
                            if not ret or frame is None or frame.size == 0:
                                consecutive_fails += 1
                                if consecutive_fails > 8:
                                    st.warning("Camera stream disconnected or frame grab failed after several attempts.")
                                    break
                                time.sleep(0.04)
                                continue
                            consecutive_fails = 0

                            if mirror_camera:
                                frame = cv2.flip(frame, 1)

                            h, w, _ = frame.shape
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                            landmarker.detect_async(mp_image, int(time.time() * 1000))

                            detected_label = None
                            x_coords, y_coords = [], []
                            dynamic_committed_this_frame = False

                            # Check MediaPipe Async Result from inference_tts module
                            if inference_tts.latest_result and inference_tts.latest_result.hand_landmarks:
                                hand_landmarks = inference_tts.latest_result.hand_landmarks[0]

                                index_pt = (hand_landmarks[8].x, hand_landmarks[8].y)
                                pinky_pt = (hand_landmarks[20].x, hand_landmarks[20].y)

                                norm_features = normalize_landmarks(hand_landmarks)
                                features = np.array(norm_features).reshape(1, -1)
                                predicted_label = classifier.predict(features)[0]

                                # Dynamic motion tracking for 'J' & 'Z'
                                dyn_label = gesture_tracker.update(index_pt, pinky_pt, prev_index_pt, prev_pinky_pt)
                                now_ts = time.time()

                                if dyn_label and (now_ts - last_dynamic_commit_time) > dynamic_cooldown:
                                    detected_label = dyn_label
                                    dynamic_committed_this_frame = True
                                    last_dynamic_commit_time = now_ts
                                elif gesture_tracker.active:
                                    detected_label = None
                                else:
                                    detected_label = predicted_label

                                prev_index_pt = index_pt
                                prev_pinky_pt = pinky_pt

                                for lm in hand_landmarks:
                                    x_coords.append(int(lm.x * w))
                                    y_coords.append(int(lm.y * h))
                            else:
                                gesture_tracker.reset()
                                prev_index_pt = None
                                prev_pinky_pt = None

                            now = time.time()

                            # --- Dynamic Gesture Commit (J / Z) ---
                            if dynamic_committed_this_frame:
                                st.session_state.current_word += detected_label
                                if voice_enabled and speak_letters:
                                    speak_async(detected_label)
                                st.session_state.transcript_history.append(
                                    {
                                        "time": time.strftime("%H:%M:%S"),
                                        "type": "DYNAMIC_CHAR",
                                        "content": detected_label,
                                    }
                                )
                                current_char = ""
                                registered_flag = False
                                flash_label = detected_label
                                flash_started_at = now

                            # --- Static Gesture Hold Commit ---
                            elif detected_label:
                                if detected_label == current_char:
                                    if not registered_flag and (now - char_hold_start) >= hold_duration:
                                        registered_flag = True

                                        if detected_label == "space":
                                            if st.session_state.current_word:
                                                word_committed = st.session_state.current_word
                                                if voice_enabled:
                                                    speak_async(word_committed)
                                                st.session_state.full_sentence.append(word_committed)
                                                st.session_state.transcript_history.append(
                                                    {
                                                        "time": time.strftime("%H:%M:%S"),
                                                        "type": "WORD",
                                                        "content": word_committed,
                                                    }
                                                )
                                                st.session_state.current_word = ""
                                            else:
                                                if st.session_state.full_sentence:
                                                    completed_sentence = " ".join(st.session_state.full_sentence)
                                                    if voice_enabled:
                                                        speak_async(completed_sentence)
                                                    st.session_state.transcript_history.append(
                                                        {
                                                            "time": time.strftime("%H:%M:%S"),
                                                            "type": "SENTENCE",
                                                            "content": completed_sentence,
                                                        }
                                                    )
                                                    st.session_state.full_sentence.clear()
                                        else:
                                            st.session_state.current_word += detected_label
                                            if voice_enabled and speak_letters:
                                                speak_async(detected_label)
                                            st.session_state.transcript_history.append(
                                                {
                                                    "time": time.strftime("%H:%M:%S"),
                                                    "type": "CHAR",
                                                    "content": detected_label,
                                                }
                                            )
                                else:
                                    current_char = detected_label
                                    char_hold_start = now
                                    registered_flag = False
                            else:
                                current_char = ""
                                registered_flag = False

                            # Calculate Progress
                            progress_ratio = 0.0
                            if detected_label and not registered_flag:
                                progress_ratio = min(1.0, (now - char_hold_start) / hold_duration)

                            flash_ratio = (
                                max(0.0, 1.0 - (now - flash_started_at) / FLASH_DURATION)
                                if flash_label
                                else 0.0
                            )

                            # --- Render Visual Overlays on Frame ---
                            if show_hud_panel:
                                draw_title_bar(frame, w)

                            if x_coords and y_coords:
                                if show_skeleton:
                                    draw_hand_skeleton(frame, x_coords, y_coords)
                                if show_bbox:
                                    pad = 25
                                    x_min = max(0, min(x_coords) - pad)
                                    x_max = min(w, max(x_coords) + pad)
                                    y_min = max(0, min(y_coords) - pad)
                                    y_max = min(h, max(y_coords) + pad)
                                    draw_corner_bbox(frame, x_min, y_min, x_max, y_max)

                            if show_hud_panel:
                                draw_hud(
                                    frame,
                                    w,
                                    h,
                                    current_char,
                                    st.session_state.current_word,
                                    st.session_state.full_sentence,
                                    progress_ratio,
                                    hand_tracked=bool(x_coords),
                                    flash_label=flash_label,
                                    flash_ratio=flash_ratio,
                                )

                            # Convert processed BGR frame to RGB for Streamlit rendering
                            display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            video_frame_placeholder.image(
                                display_rgb,
                                channels="RGB",
                                use_container_width=True,
                            )

                            # Update side panel progress & buffers smoothly
                            char_progress_bar.progress(
                                progress_ratio,
                                text=(
                                    f"Registering '{current_char}' ({int(progress_ratio * 100)}%)"
                                    if current_char
                                    else "Ready for hand sign..."
                                ),
                            )

                            # Update dynamic word/sentence text displays
                            word_placeholder.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-label">CURRENT WORD</div>
                                    <div class="word-display">{st.session_state.current_word or "..."}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            sentence_str = (
                                " ".join(st.session_state.full_sentence)
                                if st.session_state.full_sentence
                                else "..."
                            )
                            sentence_placeholder.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-label">FULL SENTENCE</div>
                                    <div class="sentence-display">{sentence_str}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Micro-sleep to allow UI threads to stay snappy
                            time.sleep(0.01)

                finally:
                    cap.release()
        else:
            video_frame_placeholder.info(
                "📹 Toggle **'Start Live Camera Stream'** above to activate real-time ASL sign detection.",
                icon="💡",
            )


# =============================================================================
# TAB 2: SNAPSHOT / PHOTO DETECTOR
# =============================================================================
with tab_snapshot:
    st.markdown("### 📸 Single-Frame Image Recognition")
    st.caption("Capture or upload a static photo of a hand sign for instant landmark detection and translation.")

    snap_mode = st.radio("Input Method", ["Upload File", "Webcam Snapshot"], horizontal=True, index=0)

    img_input = None
    if snap_mode == "Webcam Snapshot":
        st.info("ℹ️ *Note: The browser requests direct camera access for snapshots. Please turn off the live stream in Tab 1 if both use the same camera.*")
        img_input = st.camera_input("Take a sign photo")
    else:
        img_input = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if img_input is not None:
        pil_img = Image.open(img_input)
        raw_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # MediaPipe static image landmarker
        base_options = python.BaseOptions(model_asset_path=task_asset)
        img_options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

        with vision.HandLandmarker.create_from_options(img_options) as img_landmarker:
            rgb_snap = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            mp_snap = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_snap)
            result = img_landmarker.detect(mp_snap)

            res_col1, res_col2 = st.columns([1.3, 1])

            if result and result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                features = np.array(normalize_landmarks(landmarks)).reshape(1, -1)
                predicted = classifier.predict(features)[0]

                # Draw skeleton and bounding box on the snapshot
                h_img, w_img, _ = raw_bgr.shape
                x_pts = [int(lm.x * w_img) for lm in landmarks]
                y_pts = [int(lm.y * w_img) for lm in landmarks]  # landmark scaling

                draw_hand_skeleton(raw_bgr, x_pts, y_pts)
                pad = 20
                draw_corner_bbox(
                    raw_bgr,
                    max(0, min(x_pts) - pad),
                    max(0, min(y_pts) - pad),
                    min(w_img, max(x_pts) + pad),
                    min(h_img, max(y_pts) + pad),
                )

                display_snap = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                with res_col1:
                    st.image(display_snap, caption="Detected Hand Skeleton & Boundaries", use_container_width=True)

                with res_col2:
                    st.success(f"### Detected Sign: **{predicted}**")
                    st.caption("Normalized features extracted successfully.")

                    if st.button("🔊 Vocalize Sign", key="btn_speak_snap", use_container_width=True):
                        speak_async(predicted if predicted != "space" else "space")
                        st.toast(f"Spoke: {predicted}")

                    if st.button("➕ Append to Current Word Buffer", key="btn_add_snap", use_container_width=True):
                        if predicted != "space":
                            st.session_state.current_word += predicted
                        st.toast(f"Added '{predicted}' to word buffer!")
            else:
                with res_col1:
                    st.image(pil_img, caption="Original Image", use_container_width=True)
                with res_col2:
                    st.warning("No hand landmarks detected in this image. Please ensure good lighting and clear hand pose.")


# =============================================================================
# TAB 3: STANDALONE DESKTOP HUD LAUNCHER
# =============================================================================
with tab_desktop:
    st.markdown("### 🚀 High-FPS Desktop OpenCV HUD Window")
    st.markdown(
        """
        Run the native desktop OpenCV HUD window from `inference_tts.py`.
        This provides high FPS, direct GPU/display rendering, and native keyboard shortcuts:
        - **`space`**: Register space or speak current word / sentence.
        - **`c`**: Backspace last letter.
        - **`Shift + C`**: Clear entire word and sentence.
        - **`q`**: Close desktop window.
        """
    )

    def _run_hud_worker():
        try:
            st.session_state.desktop_hud_running = True
            run_translation_system(model_path="asl_model.p", task_path="hand_landmarker.task")
        finally:
            st.session_state.desktop_hud_running = False

    hud_c1, hud_c2 = st.columns([1, 1.5])
    with hud_c1:
        if not st.session_state.desktop_hud_running:
            if st.button("▶️ Launch Native Desktop Window", type="primary", use_container_width=True):
                thread = threading.Thread(target=_run_hud_worker, daemon=True)
                thread.start()
                st.session_state.desktop_hud_thread = thread
                st.session_state.desktop_hud_running = True
                st.toast("Desktop HUD window launched!", icon="🚀")
                st.rerun()
        else:
            st.info("🟢 Desktop HUD Window is currently active on your screen.")
            st.caption("Press 'q' in the OpenCV window to exit and release camera.")


# =============================================================================
# TAB 4: ASL REFERENCE & GUIDE
# =============================================================================
with tab_guide:
    st.markdown("### 📖 ASL Alphabet & Gesture Reference Guide")
    st.caption("All 26 letters of the American Sign Language alphabet and control gestures supported by the model.")

    guide_col1, guide_col2 = st.columns([1.5, 1])

    with guide_col1:
        st.markdown("#### 🖐️ Static Alphabet Signs (Hold 1.0s to Register)")
        static_classes = [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "space"
        ]

        # Render responsive grid of cards
        cols_per_row = 6
        for row_idx in range(0, len(static_classes), cols_per_row):
            row_classes = static_classes[row_idx : row_idx + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, cls_name in zip(cols, row_classes):
                with col:
                    st.markdown(
                        f"""
                        <div class="ref-letter-card">
                            <div style="font-size: 1.4rem; font-weight: 800; color: #38bdf8;">{cls_name}</div>
                            <div style="font-size: 0.7rem; color: #64748b;">Static Sign</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown("#### ⚡ Dynamic Motion Signs (Instant Trigger)")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                """
                <div class="metric-card" style="border-left: 4px solid #fbbf24;">
                    <div style="font-size: 1.4rem; font-weight: 800; color: #fbbf24;">J</div>
                    <p style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">
                        <strong>Pinky Hook Motion:</strong> Pinky tip traces downward, then hooks smoothly left.
                        Classified instantly by <code>DynamicGestureTracker</code>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                """
                <div class="metric-card" style="border-left: 4px solid #fbbf24;">
                    <div style="font-size: 1.4rem; font-weight: 800; color: #fbbf24;">Z</div>
                    <p style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">
                        <strong>Index Zig-Zag Motion:</strong> Index tip traces downward with horizontal zig-zags.
                        Classified instantly without holding.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with guide_col2:
        st.markdown("#### 🎨 Finger Skeleton Color Guide")
        finger_legend = [
            ("Wrist", "#f5f5f5", "Anchor coordinate for normalization"),
            ("Thumb", "#3c96eb", "Joints 1 to 4"),
            ("Index Finger", "#28bee1", "Joints 5 to 8 (Tracks 'Z' motion)"),
            ("Middle Finger", "#5ad28c", "Joints 9 to 12"),
            ("Ring Finger", "#eba046", "Joints 13 to 16"),
            ("Pinky Finger", "#eb5aaf", "Joints 17 to 20 (Tracks 'J' motion)"),
        ]

        for fname, color_hex, desc in finger_legend:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <div style="width: 14px; height: 14px; border-radius: 50%; background-color: {color_hex};"></div>
                    <div>
                        <strong style="color: {color_hex}; font-size: 0.85rem;">{fname}</strong>:
                        <span style="color: #94a3b8; font-size: 0.8rem;">{desc}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 💡 Tips for Best Recognition")
        st.markdown(
            """
            - Ensure good ambient lighting with minimum backlight.
            - Keep your entire hand inside the frame.
            - Hold static signs for **1.0 second** until the progress ring completes.
            - To speak a completed word, sign **`space`**.
            - Signing **`space`** again will speak the entire completed sentence!
            """
        )
