import librosa
import numpy as np
import os

def combine_features(files, output_file):
    """
    Combine multiple .npz feature files into one dataset.
    
    Args:
        files (list): List of .npz file paths.
        output_file (str): Path to save the combined dataset.
    """
    features = []
    labels = []
    
    for file in files:
        data = np.load(file)
        features.append(data['features'])
        labels.append(data['labels'])
    
    # Combine and shuffle
    features = np.vstack(features)
    labels = np.hstack(labels)
    permutation = np.random.permutation(len(labels))
    features, labels = features[permutation], labels[permutation]
    
    # Save combined data
    np.savez(output_file, features=features, labels=labels)
    print(f"Combined dataset saved to {output_file}")

# Example usage
combine_features(["features_hema_sir2.npz", "features_my2.npz"], "combined_features_hema_sir2.npz")
