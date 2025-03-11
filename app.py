from flask import Flask, render_template, request, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import random
import threading
import webbrowser
import cv2
from models import sampling

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['REPORT_FOLDER'] = 'static/reports'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained global model
global_model = tf.keras.models.load_model('global_model.h5', custom_objects={'sampling': sampling})

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((128, 128))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to check if image is grayscale (Most ultrasounds are grayscale)
def is_grayscale(image):
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:  # Check if RGB
        if np.all(img_array[:, :, 0] == img_array[:, :, 1]) and np.all(img_array[:, :, 1] == img_array[:, :, 2]):
            return True  # Grayscale image
    elif len(img_array.shape) == 2:  # Already grayscale
        return True
    return False  # RGB color image

# Reference links
reference_links = [
    {"title": "🧪 What is Liver Fibrosis?", "url": "https://www.merckmanuals.com/home/liver-and-gallbladder-disorders/fibrosis-and-cirrhosis-of-the-liver/fibrosis-of-the-liver"},
    {"title": "🧪 Treatment & Management", "url": "https://www.fda.gov/news-events/press-announcements/fda-approves-first-treatment-patients-liver-scarring-due-fatty-liver-disease"},
    {"title": "👨🏻‍🏫 Patient Stories", "url": "https://britishlivertrust.org.uk/information-and-support/support-for-you/your-stories/mikes-story-i-was-seeing-colours-differently"},
    {"title": "📑 Research Papers", "url": "https://pubmed.ncbi.nlm.nih.gov/28051792/"},
    {"title": "📊 Latest News on Liver Health", "url": "https://www.verywellhealth.com/how-a-liver-elastography-test-works-8736281"}
]

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact page
@app.route('/contact')
def contact():
    return render_template('contactus.html')

# Explore page
@app.route('/explore')
def explore():
    return render_template('explore.html')

# Prediction page
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html')

# Handle image prediction request
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('prediction.html', error="No file part")

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('prediction.html', error="Invalid file type. Please upload a PNG, JPG, or JPEG file.")

    try:
        # Read image in-memory
        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Check if the image is grayscale (Most ultrasounds are grayscale)
        if not is_grayscale(image):
            return render_template('prediction.html', error="Invalid image uploaded. Please upload a valid liver ultrasound image.")

        # Preprocess image
        img_array = preprocess_image(image)

        # Predict using the VAE-CNN model
        _, y_pred_probs = global_model.predict(img_array)
        fibrosis_stage = f"F{np.argmax(y_pred_probs)}"

        # Select 3 random reference links
        selected_links = random.sample(reference_links, 3)
        return render_template('prediction.html', fibrosis_stage=fibrosis_stage, reference_links=selected_links)

    except Exception as e:
        return render_template('prediction.html', error=str(e))

# Handle file download request
@app.route('/download')
def download_file():
    file_path = "static/understanding-your-fibroscan-results.pdf"  # Path to your file
    return send_file(file_path, as_attachment=True)

# Function to open browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

# Run the Flask application
if __name__ == '__main__':
    threading.Timer(1.5, open_browser).start()  # Open browser after 1.5 seconds
    app.run(debug=True, use_reloader=False)
