import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
from noisereduce import reduce_noise

# Function to extract MFCC features
def extract_mfcc_from_audio(audio, sample_rate, n_mfcc=13):
    """
    Extract MFCC features from audio data.
    
    Args:
        audio (numpy array): Audio data.
        sample_rate (int): Sample rate of the audio.
        n_mfcc (int): Number of MFCC coefficients to extract.
    
    Returns:
        numpy array: MFCC feature vector.
    """
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio.flatten(), sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Function to reduce noise in audio
def reduce_audio_noise(audio, sample_rate):
    """
    Reduce noise from the audio signal using noise reduction.
    
    Args:
        audio (numpy array): Audio data.
        sample_rate (int): Sample rate of the audio.
    
    Returns:
        numpy array: Noise-reduced audio data.
    """
    noise_sample = audio[:sample_rate]  # Use the first second of audio as noise profile
    return reduce_noise(y=audio.flatten(), sr=sample_rate, y_noise=noise_sample)

# Load the trained model
model = tf.keras.models.load_model("voice_recognition_model.h5")

# Real-time inference function
def real_time_inference(model, duration=5, sample_rate=44100, threshold=0.5):
    """
    Captures real-time audio, extracts MFCC features, and classifies using the model.
    
    Args:
        model (tf.keras.Model): Pretrained voice recognition model.
        duration (int): Duration of audio recording in seconds.
        sample_rate (int): Sample rate for audio recording.
        threshold (float): Decision threshold for classification.
    """
    print("Listening...")

    # Record real-time audio
    audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Audio recorded.")

    # Reduce noise in the audio
    print("Reducing noise...")
    audio_denoised = reduce_audio_noise(audio, sample_rate)

    # Extract features
    print("Extracting features...")
    mfcc_features = extract_mfcc_from_audio(audio_denoised, sample_rate).reshape(1, -1)

    # Normalize features
    mfcc_features = (mfcc_features - np.mean(mfcc_features)) / np.std(mfcc_features)

    # Predict using the model
    print("Classifying audio...")
    prediction = model.predict(mfcc_features)
    confidence = prediction[0][0]

    if confidence > threshold:
        print(f"Hello Vishal, How are you? (Confidence: {confidence:.2f})")
    else:
        print(f"Do I know you? (Confidence: {confidence:.2f})")

# Perform real-time inference
real_time_inference(model, duration=3, threshold=0.4)
