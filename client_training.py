import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

from encryption_decryption import encrypt_weights, decrypt_weights, reconstruct_weights
from preprocessing import load_and_preprocess_images
from client_testing import client_testing

client_test_data = {}

def train_model(model, client_data_path, epochs=3, loss_weights=[1.0, 0.5]):
    global client_test_data

    # Extract client identifier from the dataset path
    client_id = os.path.basename(client_data_path)
    
    categories = [f'f{i}-{client_data_path[-1]}' for i in range(5)]  

    print(f"Loading data for client: {client_id}...")

    try:
        images, labels = load_and_preprocess_images(client_data_path, categories)
        if len(images) < 10:  # Ensure dataset is large enough
            print(f"Warning: Not enough images in {client_data_path} for training.")
            return None
    except Exception as e:
        print(f"Error loading images: {e}")
        return None

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    client_test_data[client_data_path[-1]] = (X_test, y_test)

    print(f"Training model for client: {client_id} with {len(X_train)} training images and {len(X_test)} test images.")

    # Compile the model
    model.compile(
        optimizer='adam',
        loss={'decoder_output': 'mse', 'classifier_output': 'categorical_crossentropy'},
        loss_weights=loss_weights,
        metrics={'classifier_output': 'accuracy'}
    )

    # Train the model
    model.fit(
        X_train, [X_train, y_train],
        validation_data=(X_test, [X_test, y_test]),
        epochs=epochs
    )

    print(f"Training completed for client: {client_id}")

    return model.get_weights()  # Returning trained weights

