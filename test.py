# import numpy as np
# # Load the saved feature file
# data = np.load("combined_features.npz")
# print(f"Features shape: {data['features'].shape}")
# print(f"Labels shape: {data['labels'].shape}")

# # Display the first few rows of the CSV data to understand its structure
# data.head()

# # Get the total number of rows and column names
# total_rows = data.shape[0]
# column_names = data.columns.tolist()

# total_rows, column_names


# # import pandas as pd

# # Load the uploaded CSV file to check its contents
# file_path = 'C:/Users/visha/Downloads/archive/voice.csv'
# data = pd.read_csv(file_path)

# # Get the feature shape (excluding the label column) and the label shape
# feature_shape = data.iloc[:, :-1].shape  # Assuming the last column is the label
# label_shape = data.iloc[:, -1].shape  # Assuming the last column is the label

# # Print the shapes
# feature_shape, label_shape

# print(f"Features shape: {feature_shape}")
# print(f"Labels shape: {label_shape}")

# import numpy as np

# Load the uploaded .npz file to inspect its content
# file_path = 'combined_features.npz'
# data = np.load(file_path)

# # List the keys in the file to identify the rows/columns
# data_keys = data.keys()
# data_keys

# print(data_keys)



import tensorflow as tf
import librosa
import numpy as np
import sounddevice as sd
import joblib

# Load the trained model
model_path = 'voice1_recognition_model.h5'
model = tf.keras.models.load_model(model_path)

# Load the fitted scaler
scaler_file = 'scaler.pkl'
scaler = joblib.load(scaler_file)

# Function to extract MFCC features
def extract_mfcc(file_path=None, duration=3, sample_rate=44100, n_mfcc=20):
    """
    Extract MFCC features from an audio file or real-time input.
    
    Args:
        file_path (str): Path to the audio file. If None, captures audio.
        duration (int): Duration for real-time audio capture (if file_path is None).
        sample_rate (int): Sampling rate for audio.
        n_mfcc (int): Number of MFCC coefficients.
        
    Returns:
        numpy array: MFCC features.
    """
    if file_path:
        # Load audio from file
        y, sr = librosa.load(file_path, sr=sample_rate)
    else:
        # Record audio in real-time
        print(f"Recording {duration} seconds of audio...")
        y = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        y = y.flatten()
        sr = sample_rate

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

# Testing function
def test_voice(model, file_path=None, scaler=None, duration=3):
    """
    Test the voice model with a given audio file or real-time input.
    
    Args:
        model (tf.keras.Model): Trained voice classification model.
        file_path (str): Path to the audio file (optional for real-time testing).
        scaler (StandardScaler): Scaler used during training for feature normalization.
        duration (int): Duration of real-time recording (if file_path is None).
    """
    # Extract MFCC features
    features = extract_mfcc(file_path, duration=duration)
    
    # Normalize features if a scaler is provided
    if scaler:
        features = scaler.transform(features)
    
    # Predict using the model
    prediction = model.predict(features)
    if prediction[0] > 0.5:
        print("Recognized Voice: Vishal")
    else:
        print("Recognized Voice: Other")

# Test the model with real-time audio
test_voice(model, scaler=scaler)
