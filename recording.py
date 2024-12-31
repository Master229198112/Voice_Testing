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
    os.makedirs("recordings/others", exist_ok=True)
    file_path = os.path.join("recordings", file_name)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Saved: {file_path}")

# Example of recording all content systematically
texts = {
    "greetings_1": "Hello, how are you today?",
    "greetings_2": "Good morning! It's a pleasure to meet you.",
    "greetings_3": "Hey there! What's going on?",
    "Common_Questions_1": "Can you tell me the time?",
    "Common_Questions_2": "What's the weather like outside?",
    "Common_Questions_3": "Do you know where the nearest coffee shop is?",
    "Statements_1": "I am feeling great today!",
    "Statements_2": "This is a beautiful moment to capture.",
    "Statements_3": "I'd love to help you with that.",
    "tongue_twister_1": "She sells sea shells by the sea shore.",
    "tongue_twister_2": "Peter Piper picked a peck of pickled peppers.",
    "tongue_twister_3": "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "tongue_twister_4": "The quick brown fox jumps over the lazy dog.",
    "numbers_1": "One, two, three, four, five, six, seven, eight, nine, ten.",
    "numbers_2": "Eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty.",
    "numbers_3": "Thirty-three, forty-four, fifty-five, sixty-six, seventy-seven, eighty-eight, ninety-nine, one hundred.",
    "happiness_1": "I can't believe this is happening! This is amazing!",
    "happiness_2": "Wow, what a fantastic day!",
    "sadness_1": "I'm feeling really down today.",
    "sadness_2": "I can't believe this has happened.",
    "excitement_1": "Let's do this! I'm so ready!",
    "excitement_2": "I can't wait to get started on this project.",
    "calmness_1": "Let's take a moment to relax and breathe.",
    "calmness_2": "Everything will be okay in the end.",
    "news_style_1": "Today's weather forecast predicts sunny skies with a slight chance of rain in the evening. The temperature is expected to be around 75 degrees Fahrenheit.",
    "storytelling_1" : "Once upon a time, in a land far away, there lived a brave knight who fought dragons and rescued villages. Everyone admired his courage and kindness.",
    "descriptive_1" : "The gentle waves kissed the shore, creating a soothing rhythm that echoed across the quiet beach. The golden sand sparkled under the morning sun, and a soft breeze carried the salty aroma of the sea.",
    "single_letters_1" : "A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z.",
    "vowels_1" : "Education is essential for growth.",
    "vowels_2": "Umbrellas are useful in the rain.",
    "vowels_3": "Oranges are rich in vitamin C.",
    "clusters_1" : "Strength, glimpse, crunch.",
    "clusters_2" : "Blizzard, thrust, crisp.",
    "technology_1" : "Artificial intelligence is transforming the way we live and work.",
    "technology_2" : "Please reboot your computer and try again.",
    "shopping_1" : "Can I pay with a credit card?",
    "shopping_2" : "Do you have this item in stock?",
    "travel_1" : "Where is the nearest airport?",
    "travel_2" : "I'd like to book a ticket to New York.",
    "importance" :"Effective communication is one of the most important skills a person can possess. It allows us to connect with others, express our ideas, and build meaningful relationships. Listening is just as crucial as speaking, as it fosters understanding and empathy.",
    "relaxation" : "There is something truly magical about being in nature. The rustling leaves, the chirping birds, and the fresh air create a sense of peace and tranquility. It's the perfect escape from the hustle and bustle of everyday life."
}

for file_name, content in texts.items():
    print(f"Please read: {content}")
    record_audio(file_name + ".wav", duration=5)
