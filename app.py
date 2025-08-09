import os
import numpy as np
import cv2
import requests
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image

# Initialize Flask app (default: looks in 'templates/' folder for HTML)
app = Flask(__name__)

# Model download setup
MODEL_URL = "https://github.com/Adinathmk/Brain-Tumour-Detection/releases/download/Model/unet_brain.h5"
MODEL_PATH = "/tmp/unet_brain.h5"  # Render allows writing to /tmp

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from GitHub...")
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded successfully.")

# Load trained U-Net model once at startup
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# Configure upload and result folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((256, 256))  # Convert to RGB
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 256, 256, 3)
    return img_array

# Function to post-process and save the segmented result
def postprocess_and_save(result, filename):
    result_img = (result.squeeze() * 255).astype(np.uint8)  # Convert to image format
    result_pil = Image.fromarray(result_img)
    result_path = os.path.join(RESULT_FOLDER, filename)
    result_pil.save(result_path)
    return result_path

# Function for tumor severity analysis
def analyze_tumor(input_image_path, result_image_path):
    mri_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    segmentation_mask = cv2.imread(result_image_path, cv2.IMREAD_GRAYSCALE)

    if mri_image is None or segmentation_mask is None:
        return "Error loading images", "N/A"

    # Threshold to binary mask
    _, tumor_mask = cv2.threshold(segmentation_mask, 127, 255, cv2.THRESH_BINARY)

    tumor_volume = np.count_nonzero(tumor_mask)
    brain_volume = np.count_nonzero(mri_image)
    tbr = (tumor_volume / brain_volume) * 10 if brain_volume > 0 else 0

    if tbr == 0.0:
        severity = "No Tumor"
    elif tbr < 0.1:
        severity = "Low Tumor Burden"
    elif 0.1 <= tbr < 0.3:
        severity = "Moderate Tumor Burden"
    else:
        severity = "High Tumor Burden"

    return round(tbr, 4), severity

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Predict using model
    img_array = preprocess_image(filepath)
    result = model.predict(img_array)
    result_path = postprocess_and_save(result, filename)

    # Analyze tumor severity
    tumor_brain_ratio, severity_level = analyze_tumor(filepath, result_path)

    return render_template('result.html',
                           input_image=filepath,
                           result_image=result_path,
                           tumor_brain_ratio=tumor_brain_ratio,
                           severity_level=severity_level)

# Run app locally (or on Render with gunicorn)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
