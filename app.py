import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
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

            # Process the image with OpenCV
            img = cv2.imread(original_image_path)
            if img is None:
                return "Could not read image", 500 # Or 400 if client error

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Define the path for the processed image
            base, ext = os.path.splitext(filename)
            processed_filename = f"{base}_analisis_resultado{ext}"
            if ext.lower() not in ['.jpg', '.jpeg', '.png']: # Ensure valid extension for saving
                 processed_filename = f"{base}_analisis_resultado.jpg" # Default to .jpg

            processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

            cv2.imwrite(processed_image_path, gray_img)

            return "Imagen procesada con éxito. El Vigilante ha abierto los ojos."
        except Exception as e:
            # Log the exception e for debugging
            print(f"Error processing image: {e}")
            return "Error processing image", 500

    return "Unknown error", 500

if __name__ == '__main__':
    app.run(debug=True)
