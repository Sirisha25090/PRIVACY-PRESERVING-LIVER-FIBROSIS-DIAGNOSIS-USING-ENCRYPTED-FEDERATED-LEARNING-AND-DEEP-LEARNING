import numpy as np
from encryption_decryption import decrypt_weights

def aggregate_weights(client_weights):
    """
    Aggregates model weights from multiple clients by averaging corresponding layers.

    :param client_weights: List of lists containing model weights from each client.
    :return: Aggregated global weights.
    """
    if not client_weights:
        print("Error: No client weights provided for aggregation.")
        return None

    print("Starting weight aggregation...")

    # Ensure all clients provided valid weights
    client_weights = [w for w in client_weights if w is not None]

    if len(client_weights) == 0:
        print("Error: No valid client weights to aggregate.")
        return None

    # Decrypt weights if they are encrypted
    client_weights = [decrypt_weights(w) if isinstance(w, bytes) else w for w in client_weights]

    # Ensure all clients have the same number of layers
    num_layers = len(client_weights[0])
    if any(len(weights) != num_layers for weights in client_weights):
        print("Error: Inconsistent number of layers across client models.")
        return None

    new_weights = []
    for idx, weights_list_tuple in enumerate(zip(*client_weights)):
        try:
            new_weights.append(
                np.mean([np.array(w) for w in weights_list_tuple], axis=0)
            )
        except ValueError as e:
            print(f"Error during aggregation at layer {idx}: {e}")
            raise ValueError(f"Inconsistent weight shapes at layer {idx}: {[np.array(w).shape for w in weights_list_tuple]}")

    print("Weight aggregation completed successfully.")
    return new_weights
