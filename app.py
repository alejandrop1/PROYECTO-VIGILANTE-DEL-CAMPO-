import os
from flask import Flask, render_template, request, send_from_directory, url_for # Ensure all are imported
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
            filename = secure_filename(file.filename) # This is the original filename, secured
            original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_image_path)

            img = cv2.imread(original_image_path)
            if img is None:
                # Optionally remove the saved file if it's unreadable
                # os.remove(original_image_path)
                return "Could not read image", 500

            img_float = img.astype(np.float32)
            B, G, R = cv2.split(img_float)

            denominator = G + R
            ndvi = np.zeros_like(G, dtype=np.float32)
            mask = denominator != 0
            ndvi[mask] = np.true_divide((G[mask] - R[mask]), denominator[mask])

            # Calculate NDVI Statistics
            total_pixels = ndvi.size
            if total_pixels == 0: # Should not happen if image is loaded
                porcentaje_saludable = 0
                porcentaje_estres = 0
                porcentaje_suelo_desnudo = 0
            else:
                saludable_pixels = np.sum(ndvi > 0.5)
                estres_pixels = np.sum((ndvi >= 0.2) & (ndvi <= 0.5)) # Note: inclusive of 0.2 and 0.5 for 'Estrés'
                suelo_desnudo_pixels = np.sum(ndvi < 0.2)

                porcentaje_saludable = (saludable_pixels / total_pixels) * 100
                porcentaje_estres = (estres_pixels / total_pixels) * 100
                porcentaje_suelo_desnudo = (suelo_desnudo_pixels / total_pixels) * 100

            # Normalize NDVI for display
            normalized_ndvi = cv2.normalize(ndvi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_ndvi = cv2.applyColorMap(normalized_ndvi, cv2.COLORMAP_JET)

            # Define a fixed name for the processed heatmap image for simplicity
            processed_heatmap_filename = 'analisis_heatmap.jpg'
            processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_heatmap_filename)
            cv2.imwrite(processed_image_path, heatmap_ndvi)

            original_url = url_for('uploaded_file', filename=filename)
            heatmap_url = url_for('uploaded_file', filename=processed_heatmap_filename)

            return render_template('resultado.html',
                                   original_image_url=original_url,
                                   processed_image_url=heatmap_url,
                                   porcentaje_saludable=round(porcentaje_saludable, 2),
                                   porcentaje_estres=round(porcentaje_estres, 2),
                                   porcentaje_suelo_desnudo=round(porcentaje_suelo_desnudo, 2))

        except Exception as e:
            print(f"Error processing image for NDVI: {e}")
            # It's good practice to clean up saved files if an error occurs mid-process
            # if os.path.exists(original_image_path):
            #     os.remove(original_image_path)
            # if os.path.exists(processed_image_path): # Define processed_image_path outside try if used here
            #     os.remove(processed_image_path)
            return "Error processing image for NDVI analysis: " + str(e), 500

    return "Unknown error during image analysis", 500

if __name__ == '__main__':
    app.run(debug=True)
