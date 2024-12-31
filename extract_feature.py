import librosa
import numpy as np
import os


def extract_features(input_folder, output_file, n_mfcc=13):
    """
    Extract MFCC features from all audio files and save to a .npz file.
    
    Args:
        input_folder (str): Folder with audio files.
        output_file (str): Path to save the features and labels.
        n_mfcc (int): Number of MFCC coefficients to extract.
    """
    features = []
    labels = []
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(input_folder, file_name)
            try:
                y, sr = librosa.load(file_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                mfcc_mean = np.mean(mfcc.T, axis=0)  # Average across time
                features.append(mfcc_mean)
                
                # Label assignment based on folder structure
                label = 1 if "your_voice" in input_folder else 0
                labels.append(label)
            except Exception as e:
                print(f"Error extracting features from {file_name}: {e}")

    # Save features and labels
    np.savez(output_file, features=np.array(features), labels=np.array(labels))
    print(f"Features saved to {output_file}")

# Example usage
extract_features("D:\Patent\Hema_Sir\Audio\converted_wav_files", "features_hema_sir2.npz")
extract_features("recordings\your_voice", "features_my2.npz")
