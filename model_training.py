import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Provided MFCC features (as an example dataset)
mfcc_features = [-465.0564, 159.07498, 23.731998, -6.018441, 14.932665,
                 5.794623, -4.031026, 6.1005263, 4.826407, -7.6310987,
                 -5.465929, 2.4863517, -1.3770064]

# Example dataset
X = np.array([mfcc_features])  # Features for training (expand with more data)
y = np.array([1])  # Labels: 1 for your voice, 0 for others (expand as needed)

# Expanding the dataset for demonstration (you would replace this with actual data)
X = np.vstack([X, X + np.random.normal(0, 1, size=(13,)), X + np.random.normal(0, 2, size=(13,))])
y = np.array([1, 1, 0])  # Labels for the expanded dataset

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save("voice_recognition_model.h5")

# Load the model for inference
loaded_model = tf.keras.models.load_model("voice_recognition_model.h5")
