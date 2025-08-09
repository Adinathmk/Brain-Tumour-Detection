import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TensorFlow info/warnings

import numpy as np
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import requests

# -------------------- CONFIG --------------------
MODEL_URL = "https://github.com/<your-username>/<your-repo>/releases/download/v1.0/model.h5"
MODEL_PATH = "/tmp/model.h5"

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -------------------- DOWNLOAD MODEL --------------------
if not os.path.exists(MODEL_PATH):
    print("Downloading model from GitHub...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded.")

# -------------------- LOAD MODEL --------------------
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# -------------------- FLASK APP --------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# -------------------- HELPERS --------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def postprocess_mask(mask):
    mask = (mask[0] * 255).astype(np.uint8)
    return mask

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return "Invalid file type", 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Preprocess
    img_array = preprocess_image(file_path)

    # Predict
    mask = model.predict(img_array)
    mask = postprocess_mask(mask)

    # Save mask result
    result_filename = f"result_{filename}"
    result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
    cv2.imwrite(result_path, mask)

    # Dummy calculation for tumor ratio & severity (replace with your logic)
    tumor_pixels = np.sum(mask > 128)
    total_pixels = mask.size
    tumor_brain_ratio = round((tumor_pixels / total_pixels) * 100, 2)
    severity_level = "High" if tumor_brain_ratio > 30 else "Low"

    return render_template(
        "result.html",
        input_image=f"/{file_path}",
        result_image=f"/{result_path}",
        tumor_brain_ratio=tumor_brain_ratio,
        severity_level=severity_level
    )

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
