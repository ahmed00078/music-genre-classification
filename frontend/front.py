import streamlit as st
import requests
import base64
import librosa
import numpy as np
import io
import os
import matplotlib.pyplot as plt  # Added for waveform plot

# Set page config
st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide"
)

# Define service URLs
SVM_SERVICE_URL = os.getenv('SVM_SERVICE_URL', 'http://localhost:5001')
VGG19_SERVICE_URL = os.getenv('VGG19_SERVICE_URL', 'http://localhost:5002')

def encode_audio(audio_file):
    """Convert audio file to base64"""
    return base64.b64encode(audio_file.read()).decode('utf-8')

def create_waveform_image(audio_data):
    """Create waveform visualization"""
    y, sr = librosa.load(io.BytesIO(audio_data))
    plt.figure(figsize=(10, 2))
    plt.plot(y)
    plt.axis('off')
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return buf

def predict_genre(audio_file):
    """Get predictions from both models"""
    # Encode audio file
    audio_data = encode_audio(audio_file)
    
    results = {
        'SVM': None,
        'VGG19': None
    }
    
    # Get SVM prediction
    try:
        response = requests.post(
            f"{SVM_SERVICE_URL}/classify",
            json={"wav_music": audio_data}  # Updated key to match backend
        )
        if response.status_code == 200:
            results['SVM'] = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to SVM service: {str(e)}")
    
    # Get VGG19 prediction
    try:
        response = requests.post(
            f"{VGG19_SERVICE_URL}/classify",
            json={"wav_music": audio_data}  # Updated key to match backend
        )
        if response.status_code == 200:
            results['VGG19'] = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to VGG19 service: {str(e)}")
    
    return results

def main():
    # Header
    st.title("🎵 Music Genre Classification")
    st.write("""
    Upload a music file (WAV format) and our AI models will predict its genre!
    We use two different models (SVM and VGG19) for classification.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Audio Waveform")
            # Display waveform
            waveform_buf = create_waveform_image(uploaded_file.read())
            st.image(waveform_buf)
            uploaded_file.seek(0)  # Reset file pointer
        
        # Make prediction when button is clicked
        if st.button("Classify Genre"):
            with st.spinner('Analyzing music...'):
                results = predict_genre(uploaded_file)
                
                # Display results
                with col2:
                    st.subheader("Classification Results")
                    
                    # SVM Results
                    if results['SVM']:
                        st.markdown("### SVM Model")
                        st.markdown(f"""
                        - Genre: **{results['SVM']['genre']}**
                        - Confidence: **{results['SVM']['confidence']*100:.2f}%**
                        """)
                    
                    # VGG19 Results
                    if results['VGG19']:
                        st.markdown("### VGG19 Model")
                        st.markdown(f"""
                        - Genre: **{results['VGG19']['genre']}**
                        - Confidence: **{results['VGG19']['confidence']*100:.2f}%**
                        """)
                    
                    # Compare results
                    if results['SVM'] and results['VGG19']:
                        st.markdown("### Model Agreement")
                        if results['SVM']['genre'] == results['VGG19']['genre']:
                            st.success("✅ Both models agree on the genre!")
                        else:
                            st.warning("⚠️ Models predict different genres")

if __name__ == "__main__":
    main()