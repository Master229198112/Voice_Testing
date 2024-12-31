import librosa
import numpy as np

def extract_mfcc(file_name, n_mfcc=13):
    # Load the audio file
    audio, sample_rate = librosa.load(file_name, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    
    return np.mean(mfccs.T, axis=0)

mfcc_features = extract_mfcc("D:/VS Code/Voice_Testing/recordings/your_voice_sample_1.wav")
print("MFCC Features:", mfcc_features)
