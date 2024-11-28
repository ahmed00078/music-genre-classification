import streamlit as st
import requests
import base64
import librosa
import numpy as np
import io
import os
import matplotlib.pyplot as plt
print(plt.style.available)

# Enhanced UI Configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for improved styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5em;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .subtitle {
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        width: 100%;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .prediction-card {
        background-color: #f7f9fc;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Service URL Configuration
SVM_SERVICE_URL = os.getenv('SVM_SERVICE_URL', 'http://localhost:5001')
VGG19_SERVICE_URL = os.getenv('VGG19_SERVICE_URL', 'http://localhost:5002')

def encode_audio(audio_file):
    """Convert audio file to base64"""
    return base64.b64encode(audio_file.read()).decode('utf-8')

def create_enhanced_waveform(audio_data):
    """Create an enhanced waveform visualization"""
    try:
        y, sr = librosa.load(io.BytesIO(audio_data), duration=10)  # Load first 10 seconds
        
        plt.figure(figsize=(12, 4))
        plt.title('Audio Waveform', fontsize=15)
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
        
        # Create a more detailed waveform visualization
        plt.plot(np.linspace(0, len(y)/sr, num=len(y)), y, linewidth=1, color='#3498db', alpha=0.7)
        plt.xlabel('Time (seconds)', fontsize=10)
        plt.ylabel('Amplitude', fontsize=10)
        plt.grid(True, linestyle='--', linewidth=0.5)
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
        plt.close()
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error creating waveform: {e}")
        return None

def predict_genre(audio_file):
    """Get predictions from both models with improved error handling"""
    # Encode audio file
    audio_data = encode_audio(audio_file)
    
    results = {
        'SVM': None,
        'VGG19': None
    }
    
    # Improved error handling for service calls
    with st.spinner('🔍 Analyzing music...'):
        # Get SVM prediction
        try:
            response = requests.post(
                f"{SVM_SERVICE_URL}/classify",
                json={"wav_music": audio_data},
                timeout=10  # Add timeout
            )
            response.raise_for_status()  # Raise exception for bad status codes
            results['SVM'] = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error connecting to SVM service: {str(e)}")
        
        # Get VGG19 prediction
        try:
            response = requests.post(
                f"{VGG19_SERVICE_URL}/classify",
                json={"wav_music": audio_data},
                timeout=10  # Add timeout
            )
            response.raise_for_status()  # Raise exception for bad status codes
            results['VGG19'] = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error connecting to VGG19 service: {str(e)}")
    
    return results

def main():
    # Enhanced Header with Custom Styling
    st.markdown('<h1 class="main-title">🎵 Music Genre Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a music file and discover its genre using advanced AI models</p>', unsafe_allow_html=True)
    
    # Improved File Uploader with Validation
    uploaded_file = st.file_uploader(
        "Choose a WAV file", 
        type=['wav'], 
        help="Only WAV files are supported. Max file size: 10MB"
    )
    
    if uploaded_file is not None:
        # Validate file size (optional)
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
            st.error("File too large. Please upload a file smaller than 10MB.")
            return
        
        # Audio Preview
        st.audio(uploaded_file, format='audio/wav')
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Audio Waveform")
            # Display enhanced waveform
            waveform_buf = create_enhanced_waveform(uploaded_file.read())
            if waveform_buf:
                st.image(waveform_buf)
            uploaded_file.seek(0)  # Reset file pointer
        
        # Classification Button
        if st.button("🔮 Classify Genre", use_container_width=True):
            with col2:
                st.subheader("🎯 Classification Results")
                
                # Get predictions
                results = predict_genre(uploaded_file)
                
                # Display results in a styled container
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                
                # SVM Results
                if results['SVM']:
                    st.markdown("### 📊 SVM Model")
                    st.markdown(f"""
                    - **Genre**: {results['SVM']['genre']}
                    - **Confidence**: {results['SVM']['confidence']*100:.2f}%
                    """)
                
                # VGG19 Results
                if results['VGG19']:
                    st.markdown("### 🧠 VGG19 Model")
                    st.markdown(f"""
                    - **Genre**: {results['VGG19']['genre']}
                    - **Confidence**: {results['VGG19']['confidence']*100:.2f}%
                    """)
                
                # Model Agreement
                if results['SVM'] and results['VGG19']:
                    st.markdown("### 🤝 Model Agreement")
                    if results['SVM']['genre'] == results['VGG19']['genre']:
                        st.success("✅ Both models agree on the genre!")
                    else:
                        st.warning("⚠️ Models predict different genres")
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()