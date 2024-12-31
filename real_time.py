import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf

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

# Load the trained model
model = tf.keras.models.load_model("voice_recognition_model_hema_sir1.h5")

# Real-time inference function
def real_time_inference(model, duration=5, sample_rate=44100):
    """
    Captures real-time audio, extracts MFCC features, and classifies using the model.
    
    Args:
        model (tf.keras.Model): Pretrained voice recognition model.
        duration (int): Duration of audio recording in seconds.
        sample_rate (int): Sample rate for audio recording.
    """
    print("Listening...")

    # Record real-time audio
    audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished

    print("Audio recorded. Extracting features...")
    # Extract features
    mfcc_features = extract_mfcc_from_audio(audio, sample_rate).reshape(1, -1)

    print("Classifying audio...")
    # Predict using the model
    prediction = model.predict(mfcc_features)
    if prediction[0] > 0.5:
        print("Hello Director, How are you?")
    else:
        print("Hello Vishal, How are you?")

# Perform real-time inference
real_time_inference(model, duration=5)
