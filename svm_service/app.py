from flask import Flask, request, jsonify
import base64
from io import BytesIO
import numpy as np
import pickle
import librosa

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load pre-trained SVM model and scaler
def load_svm_model():
    with open('../models_train/models/svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../models_train/models/svm_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Load the model and scaler once
model, scaler = load_svm_model()

@app.route('/classify', methods=['POST'])
def classify_music():
    try:
        # Get base64 encoded audio
        base64_string = request.json.get('wav_music')
        if not base64_string:
            raise ValueError("Missing 'wav_music' in request")

        # Decode base64
        audio_data = base64.b64decode(base64_string)

        # Process audio and extract features
        features = extract_features(audio_data)
        if features is None:
            raise ValueError("Feature extraction failed")

        # Scale features and predict genre
        scaled_features = scaler.transform([features])
        genre = model.predict(scaled_features)[0]
        confidence = model.predict_proba(scaled_features)[0].max()

        return jsonify({'genre': genre, 'confidence': confidence})

    except Exception as e:
        print(f"Error during classification: {e}")
        return jsonify({'error': str(e)}), 500

def extract_features(wav_data):
    try:
        # Wrap raw binary data in BytesIO
        audio_file = BytesIO(wav_data)

        # Load audio with librosa
        y, sr = librosa.load(audio_file, sr=22050, duration=30)

        # Extract features (MFCC, Chroma, Spectrogram, etc.)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)

        # Flatten the features and return as a single vector
        feature_vector = np.concatenate([
            np.mean(mfccs, axis=1),
            np.var(mfccs, axis=1),
            [np.mean(chroma)],
            [np.mean(mel)],
            [np.mean(spec_cent)],
            [np.mean(spec_bw)],
            [np.mean(zcr)]
        ])
        return feature_vector

    except Exception as e:
        print(f"Error in extract_features: {e}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
