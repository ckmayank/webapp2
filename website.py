import os
import cv2
import numpy as np
import easyocr
from autocorrect import Speller
from flask import Flask, render_template, request, jsonify
from PIL import Image
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

spell = Speller(lang='en')
reader = easyocr.Reader(['en'], gpu=False)

def ocr_process(image_data):
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_image = process_frame(image_np)
    text_results = reader.readtext(processed_image)

    corrected_text = ''
    for result in text_results:
        corrected_text += spell(result[1]) + '\n'

    return {'detected_text': corrected_text}

def process_frame(frame):
    pil_image = Image.fromarray(frame)
    gray_image = pil_image.convert('L')
    opencv_image = np.array(gray_image)

    return opencv_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent.'}), 400

    image_data = request.files['image'].read()
    if not image_data:
        return jsonify({'error': 'No image data received.'}), 400

    result = ocr_process(image_data)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
