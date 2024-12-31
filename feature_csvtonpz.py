import pandas as pd
import librosa
import numpy as np
import os

# Load the CSV file
csv_path = 'voice.csv'
data = pd.read_csv(csv_path)

# Assuming the CSV has columns 'file_path' and 'label'
# Update these column names if they differ in your CSV
file_column = 'file_path'
label_column = 'label'

# Check the loaded data
print("CSV Data Head:")
print(data.head())

# Function to extract MFCC features
def extract_mfcc(file_path, n_mfcc=13):
    """
    Extract MFCC features from an audio file.
    
    Args:
        file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCC coefficients to extract.
        
    Returns:
        numpy array: Extracted MFCC features (mean of coefficients over time).
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Initialize a dictionary to store features grouped by labels
features_by_label = {}

# Process each file in the dataset
for _, row in data.iterrows():
    file_path = row[file_column]
    label = row[label_column]
    
    # Extract MFCC features
    features = extract_mfcc(file_path)
    
    if features is not None:
        if label not in features_by_label:
            features_by_label[label] = []
        features_by_label[label].append(features)

# Save features grouped by label
output_path = 'features_by_label.npz'
np.savez(output_path, **features_by_label)
print(f"Features grouped by label saved to {output_path}")
