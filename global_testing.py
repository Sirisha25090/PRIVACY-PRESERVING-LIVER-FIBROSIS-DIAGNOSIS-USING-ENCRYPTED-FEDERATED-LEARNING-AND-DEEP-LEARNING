import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from models import create_vae_cnn_model  # Updated to match VAE-CNN
from preprocessing import load_and_preprocess_images
from sklearn.metrics import classification_report
from models import sampling
from client_testing import dice_coefficient

# Path to the test dataset
TEST_DATA_PATH = "Dataset/global_test"

# Load global model
def global_testing():
    print("Loading test dataset...")
    try:
        categories = [f"f{i}" for i in range(5)]
        X_test, y_test = load_and_preprocess_images(TEST_DATA_PATH, categories)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("Loading global VAE-CNN model...")
    # Register the function
    # keras.utils.get_custom_objects().update({"sampling": sampling})
    global_model = tf.keras.models.load_model('global_model.h5', custom_objects={"sampling": sampling})

    print("Compiling global model...")
    global_model.compile(
        optimizer='adam',
        loss={
            "decoder_output": "mse",  # Autoencoder Loss
            "classifier_output": "categorical_crossentropy"  # Classification Loss
        },
        metrics={"classifier_output": "accuracy"}
    )

    print("Evaluating global model...")
    try:
        # Predict using the VAE-CNN (returns two outputs: reconstructed images & class probabilities)
        reconstructed_images, y_pred_probs = global_model.predict(X_test)

        # Convert class probabilities to labels
        y_pred = y_pred_probs.argmax(axis=1)  
        y_true = y_test.argmax(axis=1)  # Assuming y_test is one-hot encoded

        # Compute classification loss
        loss_fn = CategoricalCrossentropy()
        class_loss = loss_fn(y_test, y_pred_probs).numpy()

        # Compute Dice Coefficient
        dice = dice_coefficient(y_test, y_pred_probs)

        # Classification Report
        report = classification_report(y_true, y_pred, target_names=categories)

        accuracy = (y_pred == y_true).mean() * 100
        
        print(report)
        print(f"Global model classification accuracy: {accuracy:.2f}%")
        print(f"\nGlobal Model Classification Loss: {class_loss:.4f}")
        print(f"Global Model Dice Coefficient: {dice:.4f}")
        return report

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

if __name__ == "__main__":
    global_testing()
