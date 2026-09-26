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
    This guarantees that SAPI5 does not lock up after the first utterance.
    """
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            pythoncom.CoInitialize()
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
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
# 4. MEDIAPIPE ASYNC CALLBACK & MAIN LOOP
# =============================================================================
latest_result = None

def print_result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

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

    # Debouncing and hold timer variables
    current_char = ""
    char_hold_start = 0.0
    HOLD_DURATION = 0.8       # Hold a pose steady for 0.8s to register
    registered_flag = False

    current_word = ""
    full_sentence = []

    print("\nPhase 2 Translation Active.")
    print(" - Hold sign for 0.8s to type and speak character")
    print(" - Sign 1st 'space' to speak the completed word")
    print(" - Sign 2nd 'space' to speak the full sentence")
    print(" - Press 'c' to clear word | Shift+'C' to clear sentence | 'q' to quit\n")

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
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
            else:
                index_history.clear()
                pinky_history.clear()

            # Debouncing logic with double-space detection
            now = time.time()
            if detected_label:
                if detected_label == current_char:
                    if not registered_flag and (now - char_hold_start) >= HOLD_DURATION:
                        registered_flag = True
                        
                        if detected_label == "space":
                            if current_word:
                                # 1st Space: Finishes and speaks the word
                                print(f"[Speaking Word]: {current_word}")
                                speak_async(current_word)
                                full_sentence.append(current_word)
                                current_word = ""
                            else:
                                # 2nd Space: Speaks the entire sentence if words exist
                                if full_sentence:
                                    completed_sentence = " ".join(full_sentence)
                                    print(f"[Speaking Sentence]: {completed_sentence}")
                                    speak_async(completed_sentence)
                                    full_sentence.clear()
                        else:
                            # Standard letter registration
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

            # Draw Interface
            cv2.rectangle(frame, (0, h - 85), (w, h), (35, 35, 35), -1)

            progress_ratio = 0.0
            if detected_label and not registered_flag:
                progress_ratio = min(1.0, (now - char_hold_start) / HOLD_DURATION)
            cv2.rectangle(frame, (0, h - 90), (int(w * progress_ratio), h - 85), (0, 255, 0), -1)

            display_status = f"Sign: {current_char if current_char else 'None'} | Word: {current_word}"
            cv2.putText(frame, display_status, (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            sentence_display = "Sentence: " + " ".join(full_sentence[-4:])
            cv2.putText(frame, sentence_display, (20, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 1)

            cv2.imshow("ASL to Audio Translation - Phase 2", frame)

            # Granular keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Lowercase 'c' removes only one character
                if current_word:
                    current_word = current_word[:-1]
                elif full_sentence:
                    last_word = full_sentence.pop()
                    if last_word:
                        full_sentence.append(last_word[:-1])
            elif key == ord('C'):
                # Uppercase 'C' (Shift+c) erases everything
                current_word = ""
                full_sentence.clear()

    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)

if __name__ == "__main__":
    run_translation_system()