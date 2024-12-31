import os
import librosa
import soundfile as sf  # Use soundfile to write audio files

def preprocess_audio(input_folder, output_folder, sample_rate=44100):
    """
    Preprocess audio files: normalize, trim silence, and resample.
    
    Args:
        input_folder (str): Path to folder containing raw audio files.
        output_folder (str): Path to save preprocessed audio files.
        sample_rate (int): Target sample rate for resampling.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(input_folder, file_name)
            try:
                # Load audio file
                y, sr = librosa.load(file_path, sr=None)
                
                # Resample audio to the target sample rate
                y_resampled = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
                
                # Trim leading and trailing silence
                y_trimmed, _ = librosa.effects.trim(y_resampled)
                
                # Save the preprocessed audio
                output_path = os.path.join(output_folder, file_name)
                sf.write(output_path, y_trimmed, samplerate=sample_rate)
                print(f"Processed: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Example usage
input_folder = "recordings\others"
output_folder = "processed\others"
preprocess_audio(input_folder, output_folder)
