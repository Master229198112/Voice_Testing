import os
import librosa
import pandas as pd
import numpy as np

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio file
        # Compute features
        meanfreq = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)) / sr
        sd = np.std(librosa.feature.spectral_centroid(y=y, sr=sr)) / sr
        median = np.median(librosa.feature.spectral_centroid(y=y, sr=sr)) / sr
        Q25 = np.percentile(librosa.feature.spectral_centroid(y=y, sr=sr), 25) / sr
        Q75 = np.percentile(librosa.feature.spectral_centroid(y=y, sr=sr), 75) / sr
        IQR = Q75 - Q25
        skew = pd.Series(librosa.feature.spectral_centroid(y=y, sr=sr)[0]).skew()
        kurt = pd.Series(librosa.feature.spectral_centroid(y=y, sr=sr)[0]).kurt()
        sp_ent = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)) / sr
        sfm = np.mean(librosa.feature.spectral_flatness(y=y))
        mode = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)) / sr
        centroid = meanfreq  # Alias for spectral centroid mean
        harmonic = librosa.effects.harmonic(y)
        meanfun = np.mean(librosa.feature.zero_crossing_rate(y=harmonic))
        minfun = np.min(librosa.feature.zero_crossing_rate(y=harmonic))
        maxfun = np.max(librosa.feature.zero_crossing_rate(y=harmonic))
        meandom = np.mean(librosa.feature.tonnetz(y=y, sr=sr))
        mindom = np.min(librosa.feature.tonnetz(y=y, sr=sr))
        maxdom = np.max(librosa.feature.tonnetz(y=y, sr=sr))
        dfrange = maxdom - mindom
        modindx = np.mean(librosa.feature.rms(y=y))
        return [meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp_ent, sfm, mode,
                centroid, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Directory containing .wav files
audio_dir = "recordings\your_voice"
output_csv = "my_voice_dataset.csv"

# Columns for the dataset
columns = ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent',
           'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun', 'meandom', 
           'mindom', 'maxdom', 'dfrange', 'modindx']

# Process all .wav files and extract features
data = []
for file_name in os.listdir(audio_dir):
    if file_name.endswith(".wav"):
        file_path = os.path.join(audio_dir, file_name)
        features = extract_features(file_path)
        if features:
            data.append(features)

# Create a DataFrame and save as CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)
print(f"Dataset saved as {output_csv}")
