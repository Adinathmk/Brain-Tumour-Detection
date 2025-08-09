import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

import numpy as np
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image

# Flask app
app = Flask(__name__)

# Folders for uploads/results
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model
MODEL_PATH = 'model.h5'  # Make sure this is in your root folder
model = load_model(MODEL_PATH)

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess for model
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Postprocess mask
def postprocess_mask(mask):
    mask = (mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255
    return mask

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if not allowed_file(file.filename):
            return "Invalid file format", 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Predict mask
        img_array = preprocess_image(filepath)
        mask = model.predict(img_array)
        mask = postprocess_mask(mask)

        # Save result
        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, mask)

        # Calculate tumor-brain ratio
        tumor_pixels = np.sum(mask > 0)
        brain_pixels = mask.size
        tumor_brain_ratio = round((tumor_pixels / brain_pixels) * 100, 2)

        # Severity
        if tumor_brain_ratio < 1:
            severity = "No Tumor"
        elif tumor_brain_ratio < 5:
            severity = "Low"
        elif tumor_brain_ratio < 15:
            severity = "Moderate"
        else:
            severity = "High"

        return render_template(
            'result.html',
            input_image=filename,
            result_image=result_filename,
            tumor_brain_ratio=tumor_brain_ratio,
            severity_level=severity
        )

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
