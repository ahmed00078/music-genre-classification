import os
import time
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

class VGG19MusicGenreClassifier:
    def __init__(self, data_path, genres=None):
        """
        Initialize the VGG19 Music Genre Classifier
        
        :param data_path: Path to the directory containing audio files
        :param genres: List of music genres (optional)
        """
        self.data_path = data_path
        self.genres = genres or ['blues', 'classical', 'country', 'disco', 'hiphop', 
                                 'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.model = None
        self.label_encoder = None
    
    def prepare_data(self, create_spectrogram_func):
        """
        Prepare spectrogram data for training
        
        :param create_spectrogram_func: Function to create spectrograms from audio files
        :return: Tuple of spectrograms and encoded labels
        """
        spectrograms = []
        labels = []
        
        start_time = time.time()
        
        total_files = sum([len([f for f in os.listdir(os.path.join(self.data_path, 'genres_original', genre)) 
                              if f.endswith('.wav')]) for genre in self.genres])
        
        
        for genre in self.genres:
            genre_start = time.time()
            
            genre_path = os.path.join(self.data_path, 'genres_original', genre)
            for file_name in os.listdir(genre_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(genre_path, file_name)
                    spectrogram = create_spectrogram_func(file_path)
                    
                    if spectrogram is not None:
                        spectrograms.append(spectrogram)
                        labels.append(genre)
            
        return np.array(spectrograms), np.array(labels)
    
    def create_model(self, input_shape=(224, 224, 3)):
        """
        Create VGG19 model for music genre classification
        
        :param input_shape: Input shape of the spectrograms
        :return: Compiled Keras model
        """
        model_start = time.time()
        
        base_model = VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        for layer in base_model.layers:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(self.genres), activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
                
        self.model = model
        return model
    
    def train(self, create_spectrogram_func, test_size=0.2, batch_size=32, epochs=20):
        """
        Train the VGG19 model
        
        :param create_spectrogram_func: Function to create spectrograms from audio files
        :param test_size: Proportion of data to use for testing
        :param batch_size: Training batch size
        :param epochs: Number of training epochs
        :return: Training history
        """
        total_start = time.time()
                
        X, y = self.prepare_data(create_spectrogram_func)
        
        encode_start = time.time()
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        split_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )
        
        if self.model is None:
            self.create_model(input_shape=X_train[0].shape)
        
        train_start = time.time()
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test)
        )
        
        return history
    
    def save_model(self, model_path='vgg19_model.h5', encoder_path='vgg19_label_encoder.pkl'):
        """
        Save trained model and label encoder
        
        :param model_path: Path to save the model
        :param encoder_path: Path to save the label encoder
        """
        save_start = time.time()
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.model.save(model_path)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        :param X_test: Test data
        :param y_test: Test labels
        :return: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        return self.model.evaluate(X_test, y_test)

# Example usage
if __name__ == "__main__":
    def create_spectrogram(file_path, output_shape=(224, 224)):
        """
        Example spectrogram creation function
        Replace this with your actual spectrogram creation logic
        """
        import librosa
        import tensorflow as tf
        
        y, sr = librosa.load(file_path, duration=30)
        
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        spectrogram_resized = tf.image.resize(spectrogram_db[..., np.newaxis], output_shape)
        
        spectrogram_rgb = tf.image.grayscale_to_rgb(spectrogram_resized)
        
        return spectrogram_rgb.numpy()

    # Initialize and train the model
    classifier = VGG19MusicGenreClassifier('../data/')
    
    # Train the model
    history = classifier.train(create_spectrogram)
    
    # Save the model
    classifier.save_model()