import time
import pickle
import threading
import queue
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import pythoncom
from collections import deque
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
# 3. DYNAMIC MOTION HEURISTICS ('J' AND 'Z')
# =============================================================================
def check_dynamic_gestures(index_buf, pinky_buf):
    if len(index_buf) < 20:
        return None

    ix = [p[0] for p in index_buf]
    iy = [p[1] for p in index_buf]
    px = [p[0] for p in pinky_buf]
    py = [p[1] for p in pinky_buf]

    # 'J': Pinky moves down and hooks left
    if (py[-1] - py[0]) > 0.15 and (px[0] - px[-1]) > 0.05:
        return "J"

    # 'Z': Index moves down with horizontal zig-zag
    if (iy[-1] - iy[0]) > 0.15 and (max(ix) - min(ix)) > 0.15:
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


def draw_hud(frame, w, h, current_char, current_word, full_sentence, progress_ratio, hand_tracked):
    panel_h = 108
    draw_translucent_panel(frame, 0, h - panel_h, w, h, alpha=0.6)

    # --- status chip, top-left of panel ---
    chip_color = COLOR_OK if hand_tracked else COLOR_TEXT_DIM
    chip_label = "TRACKING" if hand_tracked else "NO HAND"
    cv2.circle(frame, (24, h - panel_h + 20), 6, chip_color, -1, cv2.LINE_AA)
    cv2.putText(frame, chip_label, (38, h - panel_h + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, chip_color, 1, cv2.LINE_AA)

    # --- progress ring + current letter badge ---
    ring_center = (54, h - panel_h + 66)
    draw_progress_ring(frame, ring_center, 26, progress_ratio, COLOR_ACCENT_2, 4)
    letter_text = current_char if current_char else "-"
    (tw, th_), _ = cv2.getTextSize(letter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, letter_text, (ring_center[0] - tw // 2, ring_center[1] + th_ // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2, cv2.LINE_AA)

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

    index_history = deque(maxlen=30)
    pinky_history = deque(maxlen=30)

    # Debouncing, hold timers, and auto-complete buffer variables
    current_char = ""
    char_hold_start = 0.0
    HOLD_DURATION = 1.0
    registered_flag = False

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

            if latest_result and latest_result.hand_landmarks:
                hand_landmarks = latest_result.hand_landmarks[0]

                index_history.append((hand_landmarks[8].x, hand_landmarks[8].y))
                pinky_history.append((hand_landmarks[20].x, hand_landmarks[20].y))

                norm_features = normalize_landmarks(hand_landmarks)
                features = np.array(norm_features).reshape(1, -1)
                predicted_label = classifier.predict(features)[0]

                dyn_label = check_dynamic_gestures(index_history, pinky_history)
                detected_label = dyn_label if dyn_label else predicted_label

                for lm in hand_landmarks:
                    x_coords.append(int(lm.x * w))
                    y_coords.append(int(lm.y * h))
            else:
                index_history.clear()
                pinky_history.clear()

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
            if detected_label:
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

            draw_hud(frame, w, h, current_char, current_word, full_sentence,
                     progress_ratio, hand_tracked=bool(x_coords))

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