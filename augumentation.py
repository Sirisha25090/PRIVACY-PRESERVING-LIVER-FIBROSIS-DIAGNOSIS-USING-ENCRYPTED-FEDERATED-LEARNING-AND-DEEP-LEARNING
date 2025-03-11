import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tqdm import tqdm

# Define proper augmentation techniques
datagen = ImageDataGenerator(
    rotation_range=30,       # Rotate images by up to 30 degrees
    width_shift_range=0.2,   # Shift image horizontally
    height_shift_range=0.2,  # Shift image vertically
    shear_range=0.2,         
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True,    # Flip images horizontally
    fill_mode='nearest'      # Fill in missing pixels
) 

# Function to balance dataset
def augment_images(data_path):
    clients = [os.path.join(data_path, d) for d in os.listdir(data_path) 
               if os.path.isdir(os.path.join(data_path, d)) and "test" not in d.lower()]  # Exclude test data

    for client in clients:
        print(f"Processing {client}...")
        categories = [f'f{i}-{client[-1]}' for i in range(5)]  # Generate category names
        category_paths = {cat: os.path.join(client, cat) for cat in categories}

        # Get max number of images in any category
        max_images = max(len(os.listdir(path)) for path in category_paths.values() if os.path.exists(path))

        for category, path in category_paths.items():
            if not os.path.exists(path):
                print(f"Warning: {path} does not exist. Skipping...")
                continue

            image_paths = [os.path.join(path, img) for img in os.listdir(path) if img.endswith(('png', 'jpg' 'jpeg'))]
            num_images = len(image_paths)

            if num_images < max_images:
                print(f"Augmenting {category}: {num_images} → {max_images}")
                existing_images = [load_img(img_path, target_size=(128, 128)) for img_path in image_paths]

                for i in tqdm(range(max_images - num_images)):
                    img = random.choice(existing_images)  # Select a random image
                    img_array = img_to_array(img) / 255.0  # Normalize

                    # Generate a random transformation
                    transform_params = datagen.get_random_transform(img_array.shape)
                    aug_img_array = datagen.apply_transform(img_array, transform_params)

                    # Convert back to image
                    aug_img = array_to_img(aug_img_array)

                    # Save new image
                    new_filename = os.path.join(path, f"aug_{i}.jpg")
                    aug_img.save(new_filename)

        print(f"Finished processing {client}!\n")

# Run script
if __name__ == "__main__":
    dataset_path = "dataset"  # Update this if needed
    augment_images(dataset_path)
