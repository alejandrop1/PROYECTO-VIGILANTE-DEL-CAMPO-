import os
from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)

# Define la ruta absoluta para la carpeta de la aplicación
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# Define la ruta absoluta para la carpeta de subidas
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads/')
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
            filename = secure_filename(file.filename)
            original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_image_path)

            img = cv2.imread(original_image_path)
            if img is None:
                return "Could not read image", 500

            # --- CÁLCULO DE NDVI ---
            img_float = img.astype(np.float32)
            B, G, R = cv2.split(img_float)
            denominator = G + R
            ndvi = np.zeros_like(G, dtype=np.float32)
            mask = denominator != 0
            ndvi[mask] = np.true_divide((G[mask] - R[mask]), denominator[mask])

            # --- CÁLCULO DE ESTADÍSTICAS ---
            # Umbrales para la clasificación
            saludable_threshold = 0.5
            estres_threshold = 0.2

            # Calcular el total de píxeles válidos (ignorar bordes negros)
            valid_pixels_mask = img.sum(axis=2) > 0
            total_valid_pixels = np.count_nonzero(valid_pixels_mask)

            # Contar píxeles en cada categoría, solo en las zonas válidas
            saludable_pixels = np.sum((ndvi > saludable_threshold) & valid_pixels_mask)
            estres_pixels = np.sum((ndvi >= estres_threshold) & (ndvi <= saludable_threshold) & valid_pixels_mask)
            suelo_pixels = np.sum((ndvi < estres_threshold) & valid_pixels_mask)

            # Calcular porcentajes, evitando división por cero
            if total_valid_pixels > 0:
                porc_saludable = round((saludable_pixels / total_valid_pixels) * 100, 2)
                porc_estres = round((estres_pixels / total_valid_pixels) * 100, 2)
                porc_suelo = round((suelo_pixels / total_valid_pixels) * 100, 2)
            else:
                porc_saludable, porc_estres, porc_suelo = 0, 0, 0

            # --- CREACIÓN DE IMAGEN DE RESULTADO ---
            normalized_ndvi = cv2.normalize(ndvi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_ndvi = cv2.applyColorMap(normalized_ndvi, cv2.COLORMAP_JET)

            heatmap_filename = 'analisis_heatmap.jpg'
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
            cv2.imwrite(heatmap_path, heatmap_ndvi)

            # Generar las URLs para mostrar las imágenes en la página
            original_url = url_for('uploaded_file', filename=filename)
            heatmap_url = url_for('uploaded_file', filename=heatmap_filename)

            # --- DEVOLVER RESULTADO A LA PÁGINA ---
            return render_template('resultado.html',
                                   original_image_url=original_url,
                                   processed_image_url=heatmap_url,
                                   porc_saludable=porc_saludable,
                                   porc_estres=porc_estres,
                                   porc_suelo=porc_suelo)

        except Exception as e:
            print(f"Error processing image: {e}")
            return "Error durante el análisis de la imagen: " + str(e), 500

    return "Error desconocido durante el análisis", 500

if __name__ == '__main__':
    app.run(debug=True)
