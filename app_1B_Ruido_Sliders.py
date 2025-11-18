from flask import Flask, render_template, Response, request
from io import BytesIO
import cv2
import numpy as np
import requests

app = Flask(__name__)

# --- CONFIGURACIÓN DE TU CÁMARA ---
_URL = 'http://192.168.88.74'
_PORT = '81'
_ST = '/stream'
SEP = ':'
stream_url = ''.join([_URL, SEP, _PORT, _ST])

# --- FUNCIONES DE RUIDO (Perfectas) ---
def add_gaussian_noise_color(image, mean=0, std_dev=1.0):
    try:
        std_dev = float(std_dev)
    except ValueError:
        std_dev = 0.0
    
    if std_dev == 0:
        return image
        
    image_float = image.astype(np.float32)
    gauss_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
    noisy_image_float = image_float + gauss_noise
    np.clip(noisy_image_float, 0, 255, out=noisy_image_float)
    noisy_image = noisy_image_float.astype(np.uint8)
    return noisy_image

def add_speckle_noise_color(image, variance=0.1):
    try:
        variance = float(variance)
    except ValueError:
        variance = 0.0
        
    if variance == 0:
        return image

    image_float = image.astype(np.float32)
    gauss = np.random.randn(*image.shape).astype(np.float32)
    noisy_image_float = image_float + image_float * (gauss * variance)
    np.clip(noisy_image_float, 0, 255, out=noisy_image_float)
    noisy_image = noisy_image_float.astype(np.uint8)
    return noisy_image

# --- GENERADOR DE FRAMES (Perfecto) ---
def generate_frames(gauss_std='0', speckle_var='0.0'):
    # Esta función se ejecuta con los valores que recibe de la ruta
    try:
        res = requests.get(stream_url, stream=True)
    except Exception:
        print("Error: No se puede conectar al stream de la cámara.")
        return

    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                
                if frame is None:
                    continue
                
                # Aplica ruido con los valores recibidos
                noisy_frame = add_gaussian_noise_color(frame, std_dev=gauss_std)
                noisy_frame = add_speckle_noise_color(noisy_frame, variance=speckle_var)
                
                # Concatena las imágenes
                height, width, _ = frame.shape
                total_image = np.zeros((height, width * 2, 3), dtype=np.uint8)
                total_image[:, :width] = frame
                total_image[:, width:] = noisy_frame
                
                (flag, encodedImage) = cv2.imencode(".jpg", total_image)
                if not flag:
                    continue

                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                      bytearray(encodedImage) + b'\r\n')
            
            except Exception:
                continue

# --- RUTAS DE FLASK (Perfectas) ---
@app.route("/")
def index():
    # Esta línea le dice a Flask que busque "index_ruido.html"
    # DENTRO de la carpeta "templates".
    return render_template("index_ruido.html")

@app.route("/video_stream")
def video_stream():
    # Lee los parámetros de la URL (ej: ?gauss=72&speckle=0.1)
    gauss_std = request.args.get('gauss', '0')
    speckle_var = request.args.get('speckle', '0.0')
    
    # Inicia un NUEVO generador CADA VEZ que se llama a esta ruta
    return Response(generate_frames(gauss_std=gauss_std, speckle_var=speckle_var),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)