import os
import pyaudio
import wave

def record_audio(file_name, duration=5, sample_rate=44100, channels=1):
    chunk = 1024  # Buffer size
    format = pyaudio.paInt16  # 16-bit audio format

    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk)

    print(f"Recording {file_name} for {duration} seconds...")
    frames = []

    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording complete.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the audio file
    os.makedirs("recordings", exist_ok=True)  # Create a directory for recordings
    file_path = os.path.join("recordings", file_name)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Saved: {file_path}")

# Record multiple samples
for i in range(5):  # Record 5 samples
    record_audio(f"your_voice_sample_{i + 1}.wav", duration=5)
