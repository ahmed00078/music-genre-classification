from flask import Flask, request, jsonify
import base64
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tensorflow.image import resize
from io import BytesIO
import warnings
import pickle

app = Flask(__name__)

# Load the pre-trained VGG19 model and label encoder
def load_vgg19_model():
    model = load_model('../models_train/models/vgg19_model.h5')
    with open('../models_train/models/vgg19_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

# Global variables for model and encoder
vgg19_model, label_encoder = load_vgg19_model()

@app.route('/classify', methods=['POST'])
def classify_music():
    try:
        # Get base64 encoded audio
        base64_string = request.json.get('wav_music')
        if not base64_string:
            raise ValueError("Missing 'wav_music' in request")

        # Decode base64
        audio_data = base64.b64decode(base64_string)

        # Process the audio and create a spectrogram
        spectrogram = create_spectrogram(audio_data)
        if spectrogram is None:
            raise ValueError("Spectrogram creation failed")

        # Predict genre
        predictions = vgg19_model.predict(spectrogram[np.newaxis, ...])
        predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])[0]
        confidence = np.max(predictions)
        confidence = float(confidence)

        return jsonify({'genre': predicted_label, 'confidence': confidence})

    except Exception as e:
        print(f"Error during classification: {e}")
        return jsonify({'error': 'Server error'}), 500

def create_spectrogram(audio_data):
    try:
        # Convert binary audio data to a numpy array
        audio_file = BytesIO(audio_data)
        y, sr = librosa.load(audio_file, sr=22050, duration=30)

        # Create a mel spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        # Resize spectrogram to match VGG19 input size (224x224x3)
        resized_spectrogram = resize(
            spectrogram_db[..., np.newaxis], 
            (224, 224)
        )
        spectrogram_rgb = np.repeat(resized_spectrogram, 3, axis=-1)  # Convert grayscale to RGB
        print("spectrogram_rgb",spectrogram_rgb)
        return spectrogram_rgb

    except Exception as e:
        print(f"Error in create_spectrogram: {e}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)