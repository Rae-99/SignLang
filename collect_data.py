import cv2
import mediapipe as mp
import csv
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def normalize_landmarks(hand_landmarks):
    """
    Translates hand landmarks so the wrist is the origin (0,0,0),
    and scales them to a uniform size between -1 and 1.
    """
    raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = raw_coords[0]
    shifted_coords = raw_coords - wrist
    
    max_value = np.max(np.abs(shifted_coords))
    if max_value > 0:
        normalized_coords = shifted_coords / max_value
    else:
        normalized_coords = shifted_coords
        
    return normalized_coords.flatten().tolist()

def collect_custom_data(task_path="hand_landmarker.task", output_csv="custom_asl_landmarks.csv"):
    classes = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in ["J", "Z"]] + ["space"]
    
    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3  # Lowered for better laptop camera pickup
    )
    
    header = ['label']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])
        
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not access laptop webcam.")
                return
            
            print("Webcam initialized. Live skeleton preview active.")
            
            for label in classes:
                recording = False
                frames_collected = 0
                target_frames = 200
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    
                    # Run detection live on every frame for preview
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = landmarker.detect(mp_image)
                    
                    hand_landmarks = None
                    if result.hand_landmarks:
                        hand_landmarks = result.hand_landmarks[0]
                        # Draw green skeleton points live
                        for lm in hand_landmarks:
                            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

                    if not recording:
                        cv2.putText(frame, f"Show '{label}', press 'R' to record", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, "Press 'S' to skip, 'Q' to quit", (10, 65), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    else:
                        cv2.putText(frame, f"Recording '{label}': {frames_collected}/{target_frames}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        if hand_landmarks:
                            row = [label]
                            row.extend(normalize_landmarks(hand_landmarks))
                            writer.writerow(row)
                            frames_collected += 1
                        
                        if frames_collected >= target_frames:
                            break
                            
                    cv2.imshow("Custom Dataset Collection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('r') and not recording:
                        recording = True
                    elif key == ord('s') and not recording:
                        break
                    elif key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        print(f"Collection aborted. Saved to {output_csv}.")
                        return
                        
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nDataset collection complete! Data saved to {output_csv}")

if __name__ == "__main__":
    collect_custom_data()