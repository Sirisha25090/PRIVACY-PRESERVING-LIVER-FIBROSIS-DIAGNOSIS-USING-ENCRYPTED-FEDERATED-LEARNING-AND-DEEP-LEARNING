import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64

SECRET_KEY = b'my_secret_key_16'  # Must be exactly 16, 24, or 32 bytes long

# Encrypt weights
def encrypt_weights(weights):
    serialized_weights = np.concatenate([w.flatten() for w in weights])  # Flatten all weights
    padded_length = (len(serialized_weights) + 15) // 16 * 16  # Ensure it's a multiple of 16
    padded_weights = np.pad(serialized_weights, (0, padded_length - len(serialized_weights)), 'constant')

    cipher = Cipher(algorithms.AES(SECRET_KEY), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_weights = encryptor.update(padded_weights.astype(np.float32).tobytes()) + encryptor.finalize()

    return base64.b64encode(encrypted_weights).decode('utf-8')

# Decrypt weights
def decrypt_weights(encrypted_weights):
    encrypted_weights = base64.b64decode(encrypted_weights)  # Convert string back to bytes

    cipher = Cipher(algorithms.AES(SECRET_KEY), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_weights = decryptor.update(encrypted_weights) + decryptor.finalize()

    # Convert decrypted bytes back to NumPy array
    decrypted_array = np.frombuffer(decrypted_weights, dtype=np.float32)

    return decrypted_array  # Ensure this is a NumPy array

# Reconstruct weights
def reconstruct_weights(decrypted_weights, original_shapes):
    weights = []
    start = 0
    for shape in original_shapes:
        size = np.prod(shape)
        weights.append(decrypted_weights[start:start + size].reshape(shape))  # ✅ Now this won't fail
        start += size
    return weights
