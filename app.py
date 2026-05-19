import os

# Fix OpenMP Error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import easyocr
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load OCR model
reader = easyocr.Reader(['en'])


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    processed_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        'processed.png'
    )

    cv2.imwrite(processed_path, thresh)

    return processed_path


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(file_path)

    # Preprocess image
    processed_path = preprocess_image(file_path)

    # OCR Extraction
    results = reader.readtext(processed_path, detail=0)

    extracted_text = '\n'.join(results)

    return jsonify({
        'text': extracted_text
    })


if __name__ == '__main__':
    app.run(debug=True)