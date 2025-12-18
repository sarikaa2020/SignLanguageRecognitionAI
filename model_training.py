import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# Load CSV data
def load_data(data_dir):
    X = []
    y = []
    classes = sorted(os.listdir(data_dir))
    for idx, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                path = os.path.join(folder, file)
                df = pd.read_csv(path, header=None)   # ✅ FIXED: no header
                data = df.values.flatten().astype(float)
                X.append(data)
                y.append(idx)
    return np.array(X), np.array(y)

# Train model
X, y = load_data("data")
print(f"✅ Feature shape: {X.shape}, Labels: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(set(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
model.save("sign_language_model.h5")
print("✅ Model training completed and saved as 'sign_language_model.h5'.")
