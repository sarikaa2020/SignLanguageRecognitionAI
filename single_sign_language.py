import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

# ---------------------------
# CONFIG: Update if you add gestures
# ---------------------------
labels = ['A', 'B']  # Change this list to match your dataset folders

# ---------------------------
# Load trained model
# ---------------------------
if not os.path.exists('sign_language_model.h5'):
    print("Error: Model file 'sign_language_model.h5' not found!")
    exit()

model = load_model('sign_language_model.h5')

# ---------------------------
# Initialize MediaPipe Hands
# ---------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------------------
# Start webcam
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            # Predict gesture
            features = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(features, verbose=0)
            class_index = np.argmax(prediction)
            predicted_label = labels[class_index]

            # Display prediction
            cv2.putText(frame, f"Sign: {predicted_label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------
# Clean up
# ---------------------------
cap.release()
cv2.destroyAllWindows()
hands.close()
