# train_model.py
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from feature_extraction import extract_features

ravdess_data=r"C:\Users\HP\Favorites\Downloads\Audio_Speech_Actors_01-24\Actor_01"

emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

X, y = [], []

for file in os.listdir(ravdess_data):
    if file.endswith(".wav"):
        emotion_code = file.split("-")[2]
        emotion = emotions.get(emotion_code)
        if emotion:
            features = extract_features(os.path.join(ravdess_data, file))
            X.append(features)
            y.append(emotion)

X_train, X_test, y_train, y_test = train_test_split(np.array(X), y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
joblib.dump(model, "emotion_model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
