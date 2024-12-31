import os
from pydub import AudioSegment

# Set the input and output folders
input_folder = r"D:\Patent\Hema_Sir\Audio"  # Path to your audio files
output_folder = os.path.join(input_folder, "converted_wav_files")  # Output folder for WAV files

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Convert all .waptt files to .wav with sample rate 44100
total_files = 0
for file in os.listdir(input_folder):
    if file.endswith(".waptt"):
        input_file_path = os.path.join(input_folder, file)
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.wav")
        try:
            # Load the WAPTT file (OGG format)
            audio = AudioSegment.from_file(input_file_path, format="ogg")
            
            # Set the sample rate to 44100 Hz
            audio = audio.set_frame_rate(44100)
            
            # Export the file as WAV
            audio.export(output_file_path, format="wav")
            print(f"Converted and resampled: {file} -> {output_file_path}")
            total_files += 1
        except Exception as e:
            print(f"Error converting {file}: {e}")

print(f"Conversion complete! {total_files} files were successfully converted.")
