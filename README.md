Sign Language Recognition AI:

An AI-powered Sign Language Recognition system that uses Computer Vision and Deep Learning to recognize hand gestures in real time through a webcam.
Built using Python, MediaPipe, OpenCV, and TensorFlow, this project aims to improve communication accessibility for the hearing- and speech-impaired community.

Features:

-> Real-time hand detection using MediaPipe

-> Webcam-based gesture recognition

-> Deep learning model for sign classification

-> Dataset collection and preprocessing

-> Live prediction and testing

-> Modular and well-structured codebase

Tech Stack:

| Category        | Technology         |
| --------------- | ------------------ |
| Language        | Python             |
| Computer Vision | OpenCV, MediaPipe  |
| Deep Learning   | TensorFlow, Keras  |
| Data Handling   | NumPy, Pandas      |
| Model Type      | Neural Network     |
| Input           | Webcam (Real-time) |

Project Structure:

SignLanguageRecognitionAI/
│
├── data/                     # Collected gesture datasets
│
├── data_collection.py        # Collects hand landmark data
├── hand_detection.py         # Hand detection using MediaPipe
├── model_training.py         # Model training script
├── sign_language_recognition.py  # Main recognition logic
├── real_time_test.py         # Real-time sign testing
├── single_sign_language.py   # Single sign prediction
│
├── webcam_test.py            # Webcam testing
├── test_mediapipe.py         # MediaPipe testing
├── check_csv.py              # Dataset validation
│
├── requirements.txt          # Project dependencies
├── .gitignore
└── README.md

Installation & Setup:
🔹 Prerequisites

->Python 3.8 or above

->Webcam

->Git

🔹 Clone the Repository
git clone https://github.com/sarikaa2020/SignLanguageRecognitionAI.git
cd SignLanguageRecognitionAI

🔹 Install Dependencies
pip install -r requirements.txt

How to Run the Project:
-Test Webcam
Collect Dataset:
-python data_collection.py
Train the Model:
->python model_training.py
Real-Time Sign Recognition:
->python real_time_test.py

How It Works:

->Webcam captures hand movements

->MediaPipe extracts hand landmarks

->Landmarks are processed into features

->Trained deep learning model predicts the sign

->Output is displayed in real time

Current Status:

✔ Hand detection implemented
✔ Dataset collection completed
✔ Model training completed
✔ Real-time prediction working

Future Enhancements:

-> Support for full sign language alphabets & words

-> Mobile application integration

-> Multi-language sign support

-> Speech output for recognized signs

-> Advanced CNN / LSTM models

Use Cases:

->Assistive technology for the hearing impaired

->Human–computer interaction

->AI-based gesture control systems

->Educational tools for sign language learning

Contribution:

Contributions are welcome!
Feel free to fork the repository and submit pull requests.
.
License:

This project is developed for educational and research purposes.

Author-Sarikaa Ashree
🔗 GitHub: https://github.com/sarikaa2020

Acknowledgements:

->MediaPipe by Google

->TensorFlow & Keras

->OpenCV community

Support:

If you like this project, don’t forget to star the repository!