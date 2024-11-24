import os
import time
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import pickle
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, duration=30):
        self.sample_rate = sample_rate
        self.duration = duration
        self.total_files = 0
        self.processed_files = 0
        
    def extract_features(self, file_path):
        """Extract audio features from a file"""
        start_time = time.time()
        try:
            print(f"Extracting features from: {file_path}")
            
            # Load audio file
            y, sr = librosa.load(file_path, duration=self.duration, sr=self.sample_rate)
            print(f"Audio loaded in {time.time() - start_time:.2f} seconds")
            
            # Extract features
            feature_start = time.time()
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            print(f"MFCC extraction: {time.time() - feature_start:.2f} seconds")
            
            feature_start = time.time()
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            print(f"Chroma extraction: {time.time() - feature_start:.2f} seconds")
            
            feature_start = time.time()
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            print(f"Mel spectrogram extraction: {time.time() - feature_start:.2f} seconds")
            
            feature_start = time.time()
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            print(f"Other features extraction: {time.time() - feature_start:.2f} seconds")
            
            # Calculate statistics
            features = {
                'mfcc_mean': np.mean(mfccs, axis=1),
                'mfcc_var': np.var(mfccs, axis=1),
                'chroma_mean': np.mean(chroma),
                'mel_mean': np.mean(mel),
                'spec_cent_mean': np.mean(spec_cent),
                'spec_bw_mean': np.mean(spec_bw),
                'zcr_mean': np.mean(zcr)
            }
            
            self.processed_files += 1
            print(f"Processed {self.processed_files}/{self.total_files} files")
            print(f"Total extraction time: {time.time() - start_time:.2f} seconds\n")
            
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def create_spectrogram(self, file_path, output_shape=(224, 224)):
        """Create spectrogram for VGG19 model"""
        start_time = time.time()
        try:
            print(f"Creating spectrogram for: {file_path}")
            
            # Load audio
            y, sr = librosa.load(file_path, duration=self.duration)
            print(f"Audio loaded in {time.time() - start_time:.2f} seconds")
            
            # Create spectrogram
            spec_start = time.time()
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
            print(f"Spectrogram created in {time.time() - spec_start:.2f} seconds")
            
            # Resize the spectrogram to match VGG19 input size (224, 224)
            resize_start = time.time()
            spectrogram_resized = tf.image.resize(spectrogram_db[..., np.newaxis], output_shape)  # Add channel dimension
            
            # Convert grayscale (1 channel) to RGB (3 channels) without adding an extra dimension
            spectrogram_rgb = tf.image.grayscale_to_rgb(spectrogram_resized)
            print(f"Resizing and conversion: {time.time() - resize_start:.2f} seconds")
            
            self.processed_files += 1
            print(f"Processed {self.processed_files}/{self.total_files} files")
            print(f"Total spectrogram creation time: {time.time() - start_time:.2f} seconds\n")
            
            return spectrogram_rgb
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None


class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.feature_extractor = AudioFeatureExtractor()
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
                      'jazz', 'metal', 'pop', 'reggae', 'rock']
        
    def prepare_svm_data(self):
        """Prepare data for SVM model"""
        features = []
        labels = []
        
        print("\nPreparing SVM data...")
        start_time = time.time()
        
        # Count total files
        total_files = sum([len([f for f in os.listdir(os.path.join(self.data_path, 'genres_original', genre)) 
                              if f.endswith('.wav')]) for genre in self.genres])
        self.feature_extractor.total_files = total_files
        self.feature_extractor.processed_files = 0
        
        print(f"Total files to process: {total_files}")
        
        for genre in self.genres:
            genre_start = time.time()
            print(f"\nProcessing genre: {genre}")
            
            genre_path = os.path.join(self.data_path, 'genres_original', genre)
            for file_name in os.listdir(genre_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(genre_path, file_name)
                    audio_features = self.feature_extractor.extract_features(file_path)
                    
                    if audio_features is not None:
                        feature_vector = np.concatenate([
                            audio_features['mfcc_mean'],
                            audio_features['mfcc_var'],
                            [audio_features['chroma_mean']],
                            [audio_features['mel_mean']],
                            [audio_features['spec_cent_mean']],
                            [audio_features['spec_bw_mean']],
                            [audio_features['zcr_mean']]
                        ])
                        
                        features.append(feature_vector)
                        labels.append(genre)
            
            print(f"Genre {genre} processed in {time.time() - genre_start:.2f} seconds")
        
        print(f"\nTotal data preparation time: {time.time() - start_time:.2f} seconds")
        return np.array(features), np.array(labels)
    
    def prepare_vgg19_data(self):
        """Prepare data for VGG19 model"""
        spectrograms = []
        labels = []
        
        print("\nPreparing VGG19 data...")
        start_time = time.time()
        
        # Count total files
        total_files = sum([len([f for f in os.listdir(os.path.join(self.data_path, 'genres_original', genre)) 
                              if f.endswith('.wav')]) for genre in self.genres])
        self.feature_extractor.total_files = total_files
        self.feature_extractor.processed_files = 0
        
        print(f"Total files to process: {total_files}")
        
        for genre in self.genres:
            genre_start = time.time()
            print(f"\nProcessing genre: {genre}")
            
            genre_path = os.path.join(self.data_path, 'genres_original', genre)
            for file_name in os.listdir(genre_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(genre_path, file_name)
                    spectrogram = self.feature_extractor.create_spectrogram(file_path)
                    
                    if spectrogram is not None:
                        spectrograms.append(spectrogram)
                        labels.append(genre)
            
            print(f"Genre {genre} processed in {time.time() - genre_start:.2f} seconds")
        
        print(f"\nTotal data preparation time: {time.time() - start_time:.2f} seconds")
        return np.array(spectrograms), np.array(labels)
    
    def train_svm(self):
        """Train SVM model"""
        total_start = time.time()
        
        print("\n=== Starting SVM Training ===")
        X, y = self.prepare_svm_data()
        
        # Split data
        print("\nSplitting data...")
        split_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Data split complete in {time.time() - split_start:.2f} seconds")
        
        # Scale features
        print("\nScaling features...")
        scale_start = time.time()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(f"Feature scaling complete in {time.time() - scale_start:.2f} seconds")
        
        # Train SVM
        print("\nTraining SVM model...")
        train_start = time.time()
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(X_train_scaled, y_train)
        print(f"Model training complete in {time.time() - train_start:.2f} seconds")
        
        # Evaluate
        print("\nEvaluating model...")
        eval_start = time.time()
        train_score = svm.score(X_train_scaled, y_train)
        test_score = svm.score(X_test_scaled, y_test)
        print(f"Evaluation complete in {time.time() - eval_start:.2f} seconds")
        
        print(f"\nSVM Train accuracy: {train_score:.4f}")
        print(f"SVM Test accuracy: {test_score:.4f}")
        
        # Save model and scaler
        print("\nSaving model and scaler...")
        save_start = time.time()
        with open('models/svm_model.pkl', 'wb') as f:
            pickle.dump(svm, f)
        with open('models/svm_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Model saved in {time.time() - save_start:.2f} seconds")
        
        print(f"\nTotal SVM training time: {time.time() - total_start:.2f} seconds")
    
    def train_vgg19(self):
        """Train VGG19 model"""
        total_start = time.time()
        
        print("\n=== Starting VGG19 Training ===")
        X, y = self.prepare_vgg19_data()
        
        # Encode labels
        print("\nEncoding labels...")
        encode_start = time.time()
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"Label encoding complete in {time.time() - encode_start:.2f} seconds")
        
        # Split data
        print("\nSplitting data...")
        split_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        print(f"Data split complete in {time.time() - split_start:.2f} seconds")
        
        # Create VGG19 model
        print("\nCreating VGG19 model...")
        model_start = time.time()
        base_model = VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(self.genres), activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        print(f"Model creation complete in {time.time() - model_start:.2f} seconds")
        
        # Compile model
        print("\nCompiling model...")
        compile_start = time.time()
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"Model compilation complete in {time.time() - compile_start:.2f} seconds")
        
        # Train model
        print("\nTraining VGG19 model...")
        train_start = time.time()
        history = model.fit(
            X_train,
            y_train,
            batch_size=32,
            epochs=20,
            validation_data=(X_test, y_test)
        )
        print(f"Model training complete in {time.time() - train_start:.2f} seconds")
        
        # Save model and label encoder
        print("\nSaving model and encoder...")
        save_start = time.time()
        model.save('models/vgg19_model.h5')
        with open('models/vgg19_label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"Model saved in {time.time() - save_start:.2f} seconds")
        
        print(f"\nTotal VGG19 training time: {time.time() - total_start:.2f} seconds")

if __name__ == "__main__":
    total_script_start = time.time()
    print("=== Starting Music Genre Classification Training ===")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize trainer with path to GTZAN dataset
    trainer = ModelTrainer('../data/')
    
    # Train both models
    # trainer.train_svm()
    trainer.train_vgg19()
    
    print(f"\nTotal script execution time: {time.time() - total_script_start:.2f} seconds")