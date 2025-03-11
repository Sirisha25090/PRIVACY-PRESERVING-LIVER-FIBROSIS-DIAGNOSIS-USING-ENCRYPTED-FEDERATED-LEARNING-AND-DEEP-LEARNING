import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def load_and_preprocess_images(data_path,categories):
    images = []
    labels = []
    # Update categories to match client folder structures
    
    for category in categories:
        category_path = os.path.join(data_path, category)
        label = categories.index(category)

        if not os.path.exists(category_path):
            print(f"Warning: Category folder {category_path} does not exist.")
            continue  # Skip if category folder does not exist

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = load_img(img_path, target_size=(128, 128))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    if not images:
        print(f"No images found in the path: {data_path}")
    else:
        print(f"Loaded {len(images)} images from {data_path}")

    images = np.array(images)
    labels = tf.keras.utils.to_categorical(labels, num_classes=5)

    if len(images) == 0 or len(labels) == 0:
        raise ValueError(f"Dataset at {data_path} is empty. Ensure the dataset contains valid images.")

    return images, labels