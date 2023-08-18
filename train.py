import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('parkinsons.data')


def get_augmented(df):
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

    return augmented_data

def train_model(X_train, X_test, y_train, y_test):
    
    # Create a simple model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1)

    return model

def eval_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)

    return loss,accuracy

# Perform Augmentation on data
augmented_data = data

# Separate features and labels
X = augmented_data.drop(['status', 'name'], axis=1)  # Assuming 'df' is your DataFrame
y = augmented_data['status']

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Convert labels to one-hot encoding
num_classes = 2  # Assuming you have 2 classes
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_one_hot, test_size=0.2, random_state=42)

model = train_model(X_train, X_test, y_train, y_test)

# Evaluate the model
loss, accuracy = eval_model(model, X_test, y_test)
print("Test accuracy:", accuracy)
print("Test Loss:",loss)


# Save the trained model
model.save("trained_model.keras")  # Save the model to a file named "trained_model.h5"
print("Model saved as trained_model.keras")

