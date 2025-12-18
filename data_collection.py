import cv2
import mediapipe as mp
import csv
import os
import numpy as np


DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


gesture_name = input("Enter gesture name (e.g., Hello, Thanks, A, B): ").strip()
gesture_dir = os.path.join(DATA_DIR, gesture_name)
if not os.path.exists(gesture_dir):
    os.makedirs(gesture_dir)

cap = cv2.VideoCapture(0)
count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            
            if count % 5 == 0:
                csv_path = os.path.join(gesture_dir, f"{count}.csv")
                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks)

            count += 1

    cv2.imshow("Data Collection - Press Q to Quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Data collection completed for '{gesture_name}'. Saved {count//5} samples.")
