import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import classification_report

# Step 3: Load Combined Data and Split into Train/Test Sets
# File path to the combined dataset
combined_data_file = 'combined_voice_features.csv'

# Load the combined dataset
combined_data = pd.read_csv(combined_data_file)

# Separate features (X) and labels (y)
X = combined_data.drop(columns=['label'])  # Drop the label column to get features
y = combined_data['label']  # Use the label column as the target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Step 4: Preprocess Features (Optional Scaling)
# Scale the features for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train a Neural Network Model
# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Step 6: Evaluate the Model
# Evaluate the model on the test set
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Step 7: Validate the Model
# Predict the labels for the test set
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Other Voice', 'Your Voice']))

# Step 8: Save the Trained Model
model_file = 'voice_classification_model.h5'
model.save(model_file)
print(f"Model saved to {model_file}")

# Step 9: Visualize Training History (Optional)
import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()
