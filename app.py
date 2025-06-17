import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import numpy as np # Import NumPy

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/analizar', methods=['POST'])
def analizar_imagen():
    if 'image' not in request.files:
        return "No image part in the request", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if file:
        try:
            filename = secure_filename(file.filename)
            original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_image_path)

            img = cv2.imread(original_image_path)
            if img is None:
                return "Could not read image", 500

            # Convert to float32 for calculations
            img_float = img.astype(np.float32)

            # Split into B, G, R channels
            # OpenCV loads images as BGR by default
            B, G, R = cv2.split(img_float)

            # Calculate NDVI: (G - R) / (G + R)
            # Add a small epsilon to the denominator to prevent division by zero if G+R is very close to 0,
            # though explicit check is better.
            denominator = G + R
            ndvi = np.zeros_like(G, dtype=np.float32) # Initialize NDVI array with zeros

            # Avoid division by zero by only calculating where denominator is not zero
            # Using np.true_divide or manual check
            mask = denominator != 0
            ndvi[mask] = np.true_divide((G[mask] - R[mask]), denominator[mask])

            # Normalize NDVI from [-1, 1] to [0, 255] and convert to 8-bit unsigned integer
            # cv2.normalize will handle cases where ndvi is all zeros (min=max)
            normalized_ndvi = cv2.normalize(ndvi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'analisis_ndvi.jpg')
            cv2.imwrite(processed_image_path, normalized_ndvi)

            return "Imagen procesada con éxito. El Vigilante ha abierto los ojos."
        except Exception as e:
            print(f"Error processing image for NDVI: {e}") # Log specific error
            return "Error processing image for NDVI analysis", 500

    return "Unknown error during image analysis", 500

if __name__ == '__main__':
    app.run(debug=True)
