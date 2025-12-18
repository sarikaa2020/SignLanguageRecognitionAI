import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('sign_language_model.h5')

# Define labels (update if you add more gestures)
labels = ['A', 'B']  # change this list based on your trained dataset folders

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert frame
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks as numpy array
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            
            # Convert to numpy and reshape
            features = np.array(landmarks).reshape(1, -1)

            # Predict using the trained model
            prediction = model.predict(features)
            class_index = np.argmax(prediction)
            predicted_label = labels[class_index]

            # Display prediction
            cv2.putText(frame, f"Sign: {predicted_label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
