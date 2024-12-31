import pandas as pd

# File paths for the datasets
your_voice_file = 'my_voice_dataset.csv'
other_voice_file = 'voice.csv'

# Load the datasets
your_voice_data = pd.read_csv(your_voice_file)
other_voice_data = pd.read_csv(other_voice_file)

# Add labels (assuming label column doesn't already exist)
your_voice_data['label'] = 1  # 1 for your voice
other_voice_data['label'] = 0  # 0 for other voices

# Combine the datasets
combined_data = pd.concat([your_voice_data, other_voice_data], ignore_index=True)

# Shuffle the data to mix voices
combined_data = combined_data.sample(frac=1).reset_index(drop=True)

# Save the combined dataset
output_file = 'combined_voice_features.csv'
combined_data.to_csv(output_file, index=False)
print(f"Combined dataset saved to {output_file}")
