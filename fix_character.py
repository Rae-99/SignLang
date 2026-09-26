import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def normalize_landmarks(hand_landmarks):
    """
    Translates hand landmarks so the wrist is origin (0,0,0),
    and scales them uniformly between -1 and 1[cite: 3].
    """
    raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = raw_coords[0]
    shifted_coords = raw_coords - wrist
    max_value = np.max(np.abs(shifted_coords))
    return (shifted_coords / max_value if max_value > 0 else shifted_coords).flatten().tolist()

def fix_specific_gesture(task_path="hand_landmarker.task", csv_path="custom_asl_landmarks.csv"):
    
    # =========================================================================
    # 1. UPDATE THE TARGET LETTER HERE
    # Change 'S' to whichever character needs re-recording (e.g., 'A', 'M', 'R')
    # =========================================================================
    TARGET_LETTER = 'A'  # <--- CHANGE THIS LETTER
    
    # =========================================================================
    # 2. UPDATE THE FRAME COUNT HERE
    # 200 matches your standard collect_data.py batch[cite: 3]
    # =========================================================================
    TARGET_FRAMES = 200  # <--- CHANGE THIS NUMBER IF NEEDED

    # Verify existing dataset first
    try:
        df_existing = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # Check if the target letter exists
    prior_count = len(df_existing[df_existing['label'] == TARGET_LETTER])
    print(f"Targeting '{TARGET_LETTER}'. Existing entries found: {prior_count}")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=task_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3
    )

    recorded_rows = []

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(1)  # Laptop built-in webcam
        if not cap.isOpened():
            print("Error: Could not access laptop webcam.")
            return

        print(f"Webcam ready. Hold sign for '{TARGET_LETTER}' and press 'R' to record.")
        recording = False
        frames_collected = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)
            hand_landmarks = result.hand_landmarks[0] if result.hand_landmarks else None

            # Render live skeleton preview
            if hand_landmarks:
                for lm in hand_landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

            if not recording:
                cv2.putText(frame, f"Show '{TARGET_LETTER}', press 'R' to record", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'Q' to cancel", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                cv2.putText(frame, f"Recording '{TARGET_LETTER}': {frames_collected}/{TARGET_FRAMES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if hand_landmarks:
                    normalized_row = [TARGET_LETTER] + normalize_landmarks(hand_landmarks)
                    recorded_rows.append(normalized_row)
                    frames_collected += 1

                if frames_collected >= TARGET_FRAMES:
                    break

            cv2.imshow("Replace ASL Character", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                recording = True
            elif key == ord('q'):
                print("Recording canceled. No changes made to the dataset.")
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

    # Apply atomic replacement
    if frames_collected >= TARGET_FRAMES:
        print(f"Successfully captured {TARGET_FRAMES} frames for '{TARGET_LETTER}'.")
        print(f"Updating {csv_path}...")

        # Remove previous data for this character[cite: 1, 5]
        df_cleaned = df_existing[df_existing['label'] != TARGET_LETTER]

        # Convert newly recorded samples to DataFrame
        df_new = pd.DataFrame(recorded_rows, columns=df_existing.columns)

        # Concatenate and rewrite safely
        df_final = pd.concat([df_cleaned, df_new], ignore_index=True)
        df_final.to_csv(csv_path, index=False)

        print(f"Update complete! Character '{TARGET_LETTER}' safely replaced.")
        print(f"Total dataset size: {len(df_final)} rows across {df_final['label'].nunique()} classes.")

if __name__ == "__main__":
    fix_specific_gesture()