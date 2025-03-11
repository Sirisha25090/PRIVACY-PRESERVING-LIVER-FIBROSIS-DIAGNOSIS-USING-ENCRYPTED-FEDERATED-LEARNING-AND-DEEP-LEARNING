import numpy as np
from sklearn.metrics import classification_report
from preprocessing import load_and_preprocess_images
from tensorflow.keras.losses import CategoricalCrossentropy

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice coefficient for classification tasks.

    Parameters:
    y_true -- One-hot encoded true labels
    y_pred -- Probability predictions from the model

    Returns:
    dice -- Computed Dice coefficient
    """
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)

    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
    return dice

def client_testing(model, test_data_path):
    """
    Evaluates the model on the test data and prints a classification report.
    
    Parameters:
    model -- Trained model with weights loaded
    test_data_path -- Path to the test dataset
    """
    print(f"Loading test dataset from {test_data_path}...")
    
    try:
        categories = [f"f{i}" for i in range(5)]
        X_test, y_test = load_and_preprocess_images(test_data_path,categories)
        
        if X_test is None or y_test is None or len(X_test) == 0 or len(y_test) == 0:
            print(f"Error: No valid test data found in {test_data_path}.")
            return
        
        
        if y_test.ndim == 1:  # If not one-hot encoded
            print("Error: y_test is not one-hot encoded. Check preprocessing.")
            return

        num_classes = y_test.shape[1]  # Dynamically determine number of classes
        categories = [f"f{i}" for i in range(num_classes)]
        
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print("Testing model on client data...")
    
    try:
        
        _, y_pred = model.predict(X_test)  # Extract only classification output

        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices
        y_true_classes = np.argmax(y_test, axis=1)  # Convert one-hot labels to class indices
        
        # Compute classification loss
        loss_fn = CategoricalCrossentropy()
        loss = loss_fn(y_test, y_pred).numpy()

        # Compute Dice Coefficient
        dice = dice_coefficient(y_test, y_pred)
        

        report = classification_report(y_true_classes, y_pred_classes, target_names=categories)
        accuracy = report.split("\n")[-2].strip().split()[-2]  # Extract accuracy from the report text

        
        print("\nClassification Report:\n")
        print(report)
        print(f"\nClient Model Accuracy: {accuracy}%")
        print(f"Client Model Dice Coefficient: {dice:.4f}")
        print(f"Client Model Classification Loss: {loss:.4f}")

    except Exception as e:
        print(f"Error during model evaluation: {e}")