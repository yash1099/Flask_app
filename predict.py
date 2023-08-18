import pandas as pd
import numpy as np
import librosa
import pywt

from getaudiofe import calculate_features
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('parkinsons.data')

# Define data augmentation parameters
num_augmentations = 7  # Number of augmented samples to generate
noise_mean = 0          # Mean of the random noise
noise_std = 0.05        # Standard deviation of the random noise

# Create a list to store augmented data
augmented_data_list = []


# Perform data augmentation
for _ in range(num_augmentations):
    augmented_sample = data.copy()
    
    # Add random noise to numerical columns
    for column in data.columns:
        if data[column].dtype == np.float64:
            noise = np.random.normal(noise_mean, noise_std, size=len(data))
            augmented_sample[column] += noise
# Append augmented sample to the list
    augmented_data_list.append(augmented_sample)

# Concatenate the list of augmented data into a DataFrame
augmented_data = pd.concat(augmented_data_list, ignore_index=True)


# Separate features and labels
X = augmented_data.drop(['status', 'name'], axis=1)  # Assuming 'df' is your DataFrame
y = augmented_data['status']

# Normalize features
scaler = StandardScaler()
scaler = scaler.fit(X)

# Load the audio file
audio_file = "Speaker_Diarization_Example.mp3"
feature_df = calculate_features(audio_file)

sample_input_normalized = scaler.transform([feature_df.values.tolist()[0]])

# Load the saved model
loaded_model = tf.keras.models.load_model("trained_model.keras")

# Predict with the loaded model
predictions = loaded_model.predict(sample_input_normalized)

# Print the predicted class probabilities
print("Predicted class probabilities:", predictions)


