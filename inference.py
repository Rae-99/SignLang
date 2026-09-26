import time
import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

latest_result = None

def print_result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

def normalize_landmarks(hand_landmarks):
    """
    Translates hand landmarks so the wrist is the origin (0,0,0),
    and scales them to a uniform size between -1 and 1.
    """
    raw_coords = []
    for lm in hand_landmarks:
        raw_coords.append([lm.x, lm.y, lm.z])
    raw_coords = np.array(raw_coords)
    
    # The wrist at index 0 serves as the anchor point
    wrist = raw_coords[0]
    shifted_coords = raw_coords - wrist
    
    max_value = np.max(np.abs(shifted_coords))
    if max_value > 0:
        normalized_coords = shifted_coords / max_value
    else:
        normalized_coords = shifted_coords
        
    return normalized_coords.flatten().tolist()

def check_dynamic_gestures(index_buf, pinky_buf):
    if len(index_buf) < 20:
        return None
    ix = [p[0] for p in index_buf]
    iy = [p[1] for p in index_buf]
    px = [p[0] for p in pinky_buf]
    py = [p[1] for p in pinky_buf]

    # J: Pinky moves downward then hooks left
    if (py[-1] - py[0]) > 0.15 and (px[0] - px[-1]) > 0.05:
        return "J"
    # Z: Index sweeps horizontally while moving downward
    if (iy[-1] - iy[0]) > 0.15 and (max(ix) - min(ix)) > 0.15:
        return "Z"
    return None

def run_realtime_inference(model_path="asl_model.p", task_path="hand_landmarker.task"):
    global latest_result
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=print_result_callback
    )

    cap = cv2.VideoCapture(1)  # Laptop built-in webcam
    if not cap.isOpened():
        print("Error: Could not access laptop webcam.")
        return
    
    index_history = deque(maxlen=30)
    pinky_history = deque(maxlen=30)

    print("\nWebcam feed started. Press 'q' to quit.")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            if latest_result and latest_result.hand_landmarks:
                hand_landmarks = latest_result.hand_landmarks[0]

                index_history.append((hand_landmarks[8].x, hand_landmarks[8].y))
                pinky_history.append((hand_landmarks[20].x, hand_landmarks[20].y))

                # Normalize features to match training data
                normalized_features = normalize_landmarks(hand_landmarks)

                x_coords = [int(lm.x * w) for lm in hand_landmarks]
                y_coords = [int(lm.y * h) for lm in hand_landmarks]
                
                x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
                y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                for cx, cy in zip(x_coords, y_coords):
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                features = np.array(normalized_features).reshape(1, -1)
                predicted_label = classifier.predict(features)[0]

                dynamic_label = check_dynamic_gestures(index_history, pinky_history)
                final_label = dynamic_label if dynamic_label else predicted_label

                cv2.putText(frame, f"Gesture: {final_label}", (x_min, max(35, y_min - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                index_history.clear()
                pinky_history.clear()

            cv2.imshow("ASL Real-Time Translation - Phase 1", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_inference()