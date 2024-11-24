# **Music Genre Classification**

🎵 **Music Genre Classification** is a machine learning project that classifies audio files into genres using two distinct models:  
- **Support Vector Machine (SVM)**: Based on extracted audio features (e.g., MFCCs).  
- **VGG19**: A deep learning model trained on spectrograms.  

The project is structured as a microservices-based application, leveraging Docker containers for seamless deployment and scalability.

---

## **Features**
- Upload audio files (WAV format) via a **Streamlit frontend**.
- Predict genres using:
  - **SVM Service**: Classifies based on extracted audio features.
  - **VGG19 Service**: Classifies using spectrograms processed by a VGG19 model.
- Confidence scores for predictions from both models.
- Visualization of audio waveforms in the frontend.

---

## **Project Architecture**
The application comprises three main Dockerized services:
1. **Frontend**: Built with **Streamlit** for user interaction.
2. **SVM Service**: Uses a pre-trained SVM model (via Flask) for genre classification.
3. **VGG19 Service**: Uses a pre-trained VGG19 deep learning model (via Flask) for genre classification.

### **Data Flow**
1. **User** uploads an audio file via the frontend.
2. The audio is encoded and sent to both the **SVM Service** and **VGG19 Service**.
3. Both services return the predicted genre and confidence scores to the frontend.
4. The frontend displays the results and visualizations.

---

## **Technologies Used**
- **Frontend**:
  - Streamlit
  - Python
- **Backend**:
  - Flask
  - TensorFlow/Keras (VGG19)
  - Scikit-learn (SVM)
  - Librosa (Audio processing)
- **Orchestration**:
  - Docker
  - Docker Compose

---

## **Installation and Setup**

### **Prerequisites**
- Docker and Docker Compose installed on your system.
- The following files must be present in `models_train/models`:
  - `svm_model.pkl`: Pre-trained SVM model.
  - `svm_scaler.pkl`: Scaler for SVM preprocessing.
  - `vgg19_model.h5`: Pre-trained VGG19 model.
  - `vgg19_label_encoder.pkl`: Label encoder for VGG19 model predictions.

### **Directory Structure**
```
music-genre-classification/
├── models_train/
│   └── models/
│       ├── svm_model.pkl
│       ├── svm_scaler.pkl
│       ├── vgg19_model.h5
│       └── vgg19_label_encoder.pkl
├── svm_service/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── vgg19_service/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── front.py
│   ├── Dockerfile
│   └── requirements.txt
└── docker-compose.yml
```

### **Steps to Run**
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/music-genre-classification.git
   cd music-genre-classification
   ```

2. **Ensure Model Files are in Place**
   Place the pre-trained models in `models_train/models/`.

3. **Build Docker Images**
   ```bash
   docker-compose build
   ```

4. **Run the Application**
   ```bash
   docker-compose up
   ```

5. **Access the Application**
   Open your browser and navigate to:
   - **Frontend**: [http://localhost:8501](http://localhost:8501)

---

## **Usage**
1. Upload a WAV file via the frontend.
2. View the audio waveform and classification results.
3. Compare predictions from the **SVM** and **VGG19** models, including confidence scores.

---

## **API Endpoints**

### **SVM Service**
- **Endpoint**: `POST http://localhost:5001/classify`
- **Request Body**:
  ```json
  {
      "wav_music": "base64_encoded_audio_data"
  }
  ```
- **Response**:
  ```json
  {
      "genre": "jazz",
      "confidence": 0.85
  }
  ```

### **VGG19 Service**
- **Endpoint**: `POST http://localhost:5002/classify`
- **Request Body**:
  ```json
  {
      "wav_music": "base64_encoded_audio_data"
  }
  ```
- **Response**:
  ```json
  {
      "genre": "rock",
      "confidence": 0.78
  }
  ```

---

## **Training the Models**
If you need to retrain the models:

1. **SVM Training**:
   - Run the training script:
     ```bash
     python models_train/train_svm.py
     ```

2. **VGG19 Training**:
   - Run the training script:
     ```bash
     python models_train/train.py
     ```

3. Place the updated models in `models_train/models/`.

---

## **Contributing**
Feel free to contribute to this project by:
- Reporting bugs.
- Suggesting new features.
- Submitting pull requests.

---

## **Future Enhancements**
- Add support for more audio formats (e.g., MP3).
- Improve accuracy with advanced feature engineering.
- Integrate additional models for ensemble predictions.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.