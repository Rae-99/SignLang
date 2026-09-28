import time
import pickle
import threading
import queue
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import pythoncom
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =============================================================================
# 1. THREAD-SAFE ONE-SHOT TTS AUDIO DISPATCHER
# =============================================================================
speech_queue = queue.Queue()

def tts_worker():
    """
    Background worker that initializes COM and an engine instance per speech task.
    This keeps TTS playback independent from the camera loop.
    """
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            pythoncom.CoInitialize()
            engine = pyttsx3.init()
            engine.setProperty('rate', 100)
            engine.setProperty('volume', 1.0)

            # Set voice to Microsoft Mark if available
            voices = engine.getProperty('voices')
            if len(voices) > 3:
                engine.setProperty('voice', voices[4].id)

            engine.say(text)
            engine.runAndWait()

            del engine
            pythoncom.CoUninitialize()
        except Exception as e:
            print(f"TTS playback error: {e}")
        finally:
            speech_queue.task_done()

audio_thread = threading.Thread(target=tts_worker, daemon=True)
audio_thread.start()

def speak_async(phrase):
    if phrase and phrase.strip():
        speech_queue.put(phrase)

# =============================================================================
# 2. FEATURE NORMALIZATION (WRIST-ANCHORED)
# =============================================================================
def normalize_landmarks(hand_landmarks):
    raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = raw_coords[0]
    shifted_coords = raw_coords - wrist
    max_val = np.max(np.abs(shifted_coords))
    return (shifted_coords / max_val if max_val > 0 else shifted_coords).flatten().tolist()

# =============================================================================
# 3. DYNAMIC MOTION STATE MACHINE ('J' AND 'Z') — velocity-based
# =============================================================================
# Tunable thresholds (all in normalized 0-1 landmark units, per-frame)
MOTION_START_THRESH = 0.012   # per-frame speed needed to call a stroke "started"
MOTION_STOP_THRESH  = 0.006   # per-frame speed below which the hand counts as "still"
STOP_FRAMES_NEEDED  = 4       # consecutive still frames before the stroke is "done"
MIN_STROKE_FRAMES   = 6       # strokes shorter than this are treated as noise
MAX_STROKE_FRAMES   = 45      # abandon a stroke that never settles (~1.5s @ 30fps)


class DynamicGestureTracker:
    """
    Tracks ONE continuous motion stroke for J / Z using frame-to-frame
    velocity instead of a fixed-size sliding window.

    A stroke begins the instant speed crosses MOTION_START_THRESH, and it
    ends — and is classified — only once speed drops back under
    MOTION_STOP_THRESH and stays there for STOP_FRAMES_NEEDED consecutive
    frames. That means classification always happens right when the hand
    has actually stopped moving, regardless of how many frames the stroke
    took, instead of relying on an arbitrary window that may or may not
    still contain the motion.
    """

    def __init__(self):
        self.active = False
        self.stroke_index = []   # (x, y) of index tip, recorded during the stroke
        self.stroke_pinky = []   # (x, y) of pinky tip, recorded during the stroke
        self.still_count = 0
        self.frame_count = 0

    def reset(self):
        self.__init__()

    def update(self, index_pt, pinky_pt, prev_index_pt, prev_pinky_pt):
        """
        Call once per frame with the current and previous tip positions
        (prev_* is None on the first tracked frame, or right after the
        hand was lost). Returns a committed label ("J"/"Z") on the frame
        the stroke completes, else None.
        """
        if prev_index_pt is None:
            return None

        idx_speed = np.hypot(index_pt[0] - prev_index_pt[0], index_pt[1] - prev_index_pt[1])
        pky_speed = np.hypot(pinky_pt[0] - prev_pinky_pt[0], pinky_pt[1] - prev_pinky_pt[1])
        speed = max(idx_speed, pky_speed)

        if not self.active:
            if speed > MOTION_START_THRESH:
                self.active = True
                self.stroke_index = [index_pt]
                self.stroke_pinky = [pinky_pt]
                self.still_count = 0
                self.frame_count = 0
            return None

        # Mid-stroke: keep recording the path
        self.stroke_index.append(index_pt)
        self.stroke_pinky.append(pinky_pt)
        self.frame_count += 1
        self.still_count = self.still_count + 1 if speed < MOTION_STOP_THRESH else 0

        stroke_settled = self.still_count >= STOP_FRAMES_NEEDED
        stroke_timed_out = self.frame_count >= MAX_STROKE_FRAMES

        if stroke_settled or stroke_timed_out:
            label = self._classify_stroke() if self.frame_count >= MIN_STROKE_FRAMES else None
            self.reset()
            return label

        return None

    def _classify_stroke(self):
        ix = [p[0] for p in self.stroke_index]
        px = [p[0] for p in self.stroke_pinky]
        py = [p[1] for p in self.stroke_pinky]
        iy = [p[1] for p in self.stroke_index]

        # 'J': pinky moves down, then hooks left
        if (py[-1] - py[0]) > 0.15 and (px[0] - px[-1]) > 0.05:
            return "J"

        # 'Z': index moves down with a horizontal zig-zag
        if (iy[-1] - iy[0]) > 0.10 and (max(ix) - min(ix)) > 0.07:
            return "Z"

        return None

# =============================================================================
# 4. MEDIAPIPE ASYNC CALLBACK
# =============================================================================
latest_result = None

def print_result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# =============================================================================
# 5. VISUAL THEME + RENDERING HELPERS  (new look — no detection/game logic here)
# =============================================================================

# --- Palette (BGR) -----------------------------------------------------
COLOR_PANEL      = (26, 22, 20)     # translucent dark panel base
COLOR_PANEL_EDGE = (70, 60, 55)
COLOR_ACCENT     = (235, 200, 60)   # soft cyan-blue accent
COLOR_ACCENT_2   = (200, 110, 255)  # magenta/pink accent
COLOR_WARN       = (70, 190, 255)   # amber
COLOR_OK         = (140, 235, 130)  # green
COLOR_TEXT       = (240, 240, 240)
COLOR_TEXT_DIM   = (165, 165, 165)

# 21-point hand skeleton connections (standard MediaPipe hand topology)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),              # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),              # index
    (5, 9), (9, 10), (10, 11), (11, 12),         # middle
    (9, 13), (13, 14), (14, 15), (15, 16),       # ring
    (13, 17), (17, 18), (18, 19), (19, 20),      # pinky
    (0, 17),                                     # palm base
]

# Per-finger color grouping, keyed by the joint index range each line belongs to
FINGER_COLOR_BY_JOINT = {}
for j in range(0, 5):
    FINGER_COLOR_BY_JOINT[j] = (235, 150, 60)     # thumb - blue
for j in range(5, 9):
    FINGER_COLOR_BY_JOINT[j] = (225, 190, 40)     # index - cyan
for j in range(9, 13):
    FINGER_COLOR_BY_JOINT[j] = (140, 210, 90)     # middle - green
for j in range(13, 17):
    FINGER_COLOR_BY_JOINT[j] = (70, 160, 235)     # ring - orange
for j in range(17, 21):
    FINGER_COLOR_BY_JOINT[j] = (170, 90, 235)     # pinky - purple/red
FINGER_COLOR_BY_JOINT[0] = (245, 245, 245)        # wrist - white


def draw_translucent_panel(frame, x1, y1, x2, y2, alpha=0.55, color=COLOR_PANEL, border=COLOR_PANEL_EDGE):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)


def draw_hand_skeleton(frame, x_coords, y_coords):
    """Colored skeleton with a soft glow, drawn from the same landmark pixel
    coordinates the original code collected — pure rendering, no new data."""
    points = list(zip(x_coords, y_coords))

    # Glow pass (thicker, dim) then crisp pass (thin, bright)
    for a, b in HAND_CONNECTIONS:
        if a >= len(points) or b >= len(points):
            continue
        color = FINGER_COLOR_BY_JOINT.get(b, COLOR_ACCENT)
        cv2.line(frame, points[a], points[b], color, 4, cv2.LINE_AA)
        cv2.line(frame, points[a], points[b], (255, 255, 255), 1, cv2.LINE_AA)

    tip_ids = {4, 8, 12, 16, 20}
    for i, (cx, cy) in enumerate(points):
        color = FINGER_COLOR_BY_JOINT.get(i, COLOR_ACCENT)
        radius = 7 if i in tip_ids else 5
        cv2.circle(frame, (cx, cy), radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), max(1, radius - 3), (255, 255, 255), -1, cv2.LINE_AA)


def draw_corner_bbox(frame, x_min, y_min, x_max, y_max, color=COLOR_ACCENT, length_ratio=0.18, thickness=2):
    """Futuristic 'AR-style' corner-bracket bounding box instead of a full rectangle."""
    bw, bh = x_max - x_min, y_max - y_min
    lx = max(12, int(bw * length_ratio))
    ly = max(12, int(bh * length_ratio))
    corners = [
        ((x_min, y_min), (1, 1)),
        ((x_max, y_min), (-1, 1)),
        ((x_min, y_max), (1, -1)),
        ((x_max, y_max), (-1, -1)),
    ]
    for (cx, cy), (sx, sy) in corners:
        cv2.line(frame, (cx, cy), (cx + sx * lx, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + sy * ly), color, thickness, cv2.LINE_AA)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (color[0], color[1], color[2]), 1, cv2.LINE_AA)


def draw_progress_ring(frame, center, radius, ratio, color=COLOR_ACCENT, thickness=4):
    cv2.ellipse(frame, center, (radius, radius), -90, 0, 360, (60, 60, 60), thickness, cv2.LINE_AA)
    if ratio > 0:
        cv2.ellipse(frame, center, (radius, radius), -90, 0, 360 * min(1.0, ratio), color, thickness, cv2.LINE_AA)


def draw_hud(frame, w, h, current_char, current_word, full_sentence, progress_ratio, hand_tracked,
             flash_label=None, flash_ratio=0.0):
    panel_h = 108
    draw_translucent_panel(frame, 0, h - panel_h, w, h, alpha=0.6)

    # --- status chip, top-left of panel ---
    chip_color = COLOR_OK if hand_tracked else COLOR_TEXT_DIM
    chip_label = "TRACKING" if hand_tracked else "NO HAND"
    cv2.circle(frame, (24, h - panel_h + 20), 6, chip_color, -1, cv2.LINE_AA)
    cv2.putText(frame, chip_label, (38, h - panel_h + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, chip_color, 1, cv2.LINE_AA)

    # --- progress ring + current letter badge ---
    # While a dynamic (J/Z) commit is fresh, flash_ratio fades 1.0 -> 0.0 and
    # we swap in the committed letter + a gold ring instead of the normal
    # hold-progress ring, since a motion commit has no "hold" to show.
    is_flashing = flash_label is not None and flash_ratio > 0.0
    ring_center = (54, h - panel_h + 66)

    if is_flashing:
        flash_color = tuple(int(c * flash_ratio + 60 * (1 - flash_ratio)) for c in COLOR_WARN)
        draw_progress_ring(frame, ring_center, 26, 1.0, flash_color, 5)
        letter_text = flash_label
    else:
        draw_progress_ring(frame, ring_center, 26, progress_ratio, COLOR_ACCENT_2, 4)
        letter_text = current_char if current_char else "-"

    (tw, th_), _ = cv2.getTextSize(letter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, letter_text, (ring_center[0] - tw // 2, ring_center[1] + th_ // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2, cv2.LINE_AA)

    if is_flashing:
        tag = "MOTION"
        cv2.putText(frame, tag, (ring_center[0] - 24, ring_center[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WARN, 1, cv2.LINE_AA)

    # --- word + sentence text ---
    text_x = 100
    cv2.putText(frame, "WORD", (text_x, h - panel_h + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, current_word if current_word else "...", (text_x, h - panel_h + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_WARN, 2, cv2.LINE_AA)

    cv2.putText(frame, "SENTENCE", (text_x, h - panel_h + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    sentence_preview = " ".join(full_sentence[-6:]) if full_sentence else "..."
    cv2.putText(frame, sentence_preview, (text_x, h - panel_h + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)

    # --- control hints, right side of panel ---
    hints = "Hold sign  |  'space' = speak word  |  c = backspace  |  Shift+C = clear  |  q = quit"
    (hw, _), _ = cv2.getTextSize(hints, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, hints, (max(text_x, w - hw - 20), h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)


def draw_title_bar(frame, w):
    draw_translucent_panel(frame, 0, 0, w, 40, alpha=0.5)
    cv2.putText(frame, "ASL -> AUDIO TRANSLATOR", (16, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 1, cv2.LINE_AA)


# =============================================================================
# 6. MAIN LOOP  (all detection / debouncing / TTS logic unchanged)
# =============================================================================
def run_translation_system(model_path="asl_model.p", task_path="hand_landmarker.task"):
    global latest_result

    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        result_callback=print_result_callback
    )

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not access laptop webcam.")
        return

    gesture_tracker = DynamicGestureTracker()
    prev_index_pt = None
    prev_pinky_pt = None

    # Debouncing, hold timers, and auto-complete buffer variables
    current_char = ""
    char_hold_start = 0.0
    HOLD_DURATION = 1.0
    registered_flag = False

    # Dynamic-gesture (J/Z) event tracking — these fire once, not held
    DYNAMIC_COOLDOWN = 1.2
    last_dynamic_commit_time = 0.0

    # HUD flash shown briefly when a J/Z commit fires
    FLASH_DURATION = 0.45
    flash_label = None
    flash_started_at = 0.0

    current_word = ""
    full_sentence = []

    print("\nASL translation active.")
    print(" - Hold a sign to register it")
    print(" - Sign 'space' to speak the current word")
    print(" - Press 'c' to clear word | Shift+'C' to clear sentence | 'q' to quit\n")

    window_name = "ASL to Audio Translation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            landmarker.detect_async(mp_image, int(time.time() * 1000))

            detected_label = None
            x_coords, y_coords = [], []
            dynamic_committed_this_frame = False

            if latest_result and latest_result.hand_landmarks:
                hand_landmarks = latest_result.hand_landmarks[0]

                index_pt = (hand_landmarks[8].x, hand_landmarks[8].y)
                pinky_pt = (hand_landmarks[20].x, hand_landmarks[20].y)

                norm_features = normalize_landmarks(hand_landmarks)
                features = np.array(norm_features).reshape(1, -1)
                predicted_label = classifier.predict(features)[0]

                # Fires only on the frame a stroke actually settles (velocity-based),
                # not on an arbitrary fixed window.
                dyn_label = gesture_tracker.update(index_pt, pinky_pt, prev_index_pt, prev_pinky_pt)
                now_ts = time.time()

                if dyn_label and (now_ts - last_dynamic_commit_time) > DYNAMIC_COOLDOWN:
                    detected_label = dyn_label
                    dynamic_committed_this_frame = True
                    last_dynamic_commit_time = now_ts
                elif gesture_tracker.active:
                    # Mid-stroke: suppress static ML classifier so 'X' doesn't sneak in
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

            # =====================================================================
            # 7. RENDER SKELETON + BOUNDING BOX  (visuals only)
            # =====================================================================
            draw_title_bar(frame, w)

            if x_coords and y_coords:
                draw_hand_skeleton(frame, x_coords, y_coords)

                pad = 25
                x_min, x_max = max(0, min(x_coords) - pad), min(w, max(x_coords) + pad)
                y_min, y_max = max(0, min(y_coords) - pad), min(h, max(y_coords) + pad)
                draw_corner_bbox(frame, x_min, y_min, x_max, y_max)

            # =====================================================================
            # 8. DEBOUNCING LOGIC WITH CONSECUTIVE SPACE SENTENCE COMMIT (unchanged)
            # =====================================================================
            now = time.time()

            if dynamic_committed_this_frame:
                # J/Z: commit instantly, skip the hold timer entirely
                current_word += detected_label
                print(f"[Speaking Char]: {detected_label}")
                speak_async(detected_label)
                current_char = ""
                registered_flag = False
                flash_label = detected_label
                flash_started_at = now

            elif detected_label:
                if detected_label == current_char:
                    if not registered_flag and (now - char_hold_start) >= HOLD_DURATION:
                        registered_flag = True

                        if detected_label == "space":
                            if current_word:
                                print(f"[Speaking Word]: {current_word}")
                                speak_async(current_word)
                                full_sentence.append(current_word)
                                current_word = ""
                            else:
                                if full_sentence:
                                    completed_sentence = " ".join(full_sentence)
                                    print(f"[Speaking Sentence]: {completed_sentence}")
                                    speak_async(completed_sentence)
                                    full_sentence.clear()
                        else:
                            current_word += detected_label
                            print(f"[Speaking Char]: {detected_label}")
                            speak_async(detected_label)
                else:
                    current_char = detected_label
                    char_hold_start = now
                    registered_flag = False
            else:
                current_char = ""
                registered_flag = False

            # =====================================================================
            # 9. HUD STATUS PANEL  (visuals only)
            # =====================================================================
            progress_ratio = 0.0
            if detected_label and not registered_flag:
                progress_ratio = min(1.0, (now - char_hold_start) / HOLD_DURATION)

            flash_ratio = max(0.0, 1.0 - (now - flash_started_at) / FLASH_DURATION) if flash_label else 0.0

            draw_hud(frame, w, h, current_char, current_word, full_sentence,
                     progress_ratio, hand_tracked=bool(x_coords),
                     flash_label=flash_label, flash_ratio=flash_ratio)

            cv2.imshow(window_name, frame)

            # Granular Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if current_word:
                    current_word = current_word[:-1]  # Remove only the last character
                elif full_sentence:
                    last_word = full_sentence.pop()  # Remove last word from sentence
                    if last_word:
                        full_sentence.append(last_word[:-1])  # Trim last character from that word
            elif key == ord('C'):
                current_word = ""
                full_sentence.clear()  # Shift+C clears complete sentence history

    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)

if __name__ == "__main__":
    run_translation_system()