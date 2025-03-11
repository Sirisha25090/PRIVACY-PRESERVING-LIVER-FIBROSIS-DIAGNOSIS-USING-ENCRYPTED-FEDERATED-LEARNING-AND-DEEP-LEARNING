import tensorflow as tf
import numpy as np
import os

from encryption_decryption import encrypt_weights, decrypt_weights, reconstruct_weights
from preprocessing import load_and_preprocess_images
from weights_aggregation import aggregate_weights
from client_training import train_model, client_test_data
from global_testing import global_testing
from client_testing import client_testing
from models import create_vae_cnn_model

# Initialize separate models for each client
client_models = {i: create_vae_cnn_model() for i in range(1, 11)}

client_data_paths = ["Dataset/client 1", "Dataset/client 2", "Dataset/client 3", "Dataset/client 4", "Dataset/client 5", "Dataset/client 6", "Dataset/client 7", "Dataset/client 8", "Dataset/client 9"]


global_model = create_vae_cnn_model()
global_weights = global_model.get_weights()

tolerance = 0.0001
max_rounds = 7
previous_loss = float('inf')

for round_num in range(max_rounds):
    print(f"\nRound {round_num + 1} of federated training...")
    client_weights = []
    client_losses = []
    
    # Distribute global weights to clients
    print("Enccypting global model weights...")
    encrypted_global_weights = encrypt_weights(global_weights)
    print("Decrypting global model weights...")
    decrypted_global_weights = reconstruct_weights(decrypt_weights(encrypted_global_weights), [w.shape for w in global_weights])
    
    for i, client_path in enumerate(client_data_paths):
        print(f"Training client {i + 1}...")
        client_model = client_models[i + 1]
        client_model.set_weights(decrypted_global_weights)
        trained_weights = train_model(client_model, client_path)
        print("Encrypting client model weights...")
        encrypted_client_weights = encrypt_weights(trained_weights)
        client_weights.append(encrypted_client_weights)
        X_test, y_test = client_test_data[client_path[-1]]

        client_losses.append(client_model.evaluate(X_test, [X_test,y_test], verbose=0)[0]) #the output of evaluate is a list of loss and accuracy
    #decrypting the client weights
    print("Decrypting client model weights...")
    client_weights = [reconstruct_weights(decrypt_weights(weights), [w.shape for w in global_weights]) for weights in client_weights]
    print("Aggregating global model weights...")
    global_weights = aggregate_weights(client_weights)
    global_model.set_weights(global_weights)
    
    avg_loss = np.mean(client_losses)
    print(f"Average client loss: {avg_loss}")
    if abs(previous_loss - avg_loss) < tolerance:
        print("Convergence reached. Stopping training.")
        break
    previous_loss = avg_loss

print("Federated training completed.")
global_model.save('global_model.h5')

# Local client testing
test_data_paths = [
    "Dataset/client 1/test 1", "Dataset/client 2/test 2", "Dataset/client 3/test 3", "Dataset/client 4/test 4","Dataset/client 5/test 5", "Dataset/client 6/test 6", "Dataset/client 7/test 7", "Dataset/client 8/test 8", "Dataset/client 9/test 9"]


for i, client_test_path in enumerate(test_data_paths):
    client_testing(client_models[i + 1], client_test_path)


# Global model testing
categories = [f"f{i}" for i in range(5)]
global_testing_data = load_and_preprocess_images("Dataset/global_test",categories)
print("Global Model Testing:")
global_testing()