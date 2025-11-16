# =================================================================
# ARCHIVO: app_parte1B_pytorch.py
# PRUEBA: Tarea 1-B.3 (Filtro Gaussiano con PyTorch)
# =================================================================

from flask import Flask, render_template, Response
from io import BytesIO
import cv2
import numpy as np
import requests
import time

# --- Importaciones de PyTorch ---
import torch
import torch.nn.functional as F
# --------------------------------

app = Flask(__name__)

# --- Configuración de la Cámara ---
_URL = 'http://192.168.88.74' # (Asegúrate de que esta sea la IP de tu cámara)
_PORT = '81'
_ST = '/stream'
SEP = ':'
stream_url = ''.join([_URL, SEP, _PORT, _ST])
# ----------------------------------

# --- Tarea 1-B.3: Crear el Kernel de PyTorch (Filtro Gaussiano 5x5) ---
# 1. Creamos un kernel gaussiano 5x5
g_kernel = np.array([
    [1, 4,  7,  4,  1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4,  7,  4,  1]
], dtype=np.float32)

# 2. Normalizamos el kernel (la suma debe ser 1)
g_kernel /= 273.0  # La suma de todos los elementos es 273

# 3. Convertimos el kernel a un Tensor de PyTorch
# El formato debe ser [out_channels, in_channels, height, width]
kernel_torch = torch.from_numpy(g_kernel).unsqueeze(0).unsqueeze(0)
# -------------------------------------------------------------------


def video_capture():
    """
    Esta función añade ruido Gaussiano y compara el filtro Gaussiano de OpenCV
    con una implementación manual de Filtro Gaussiano en PyTorch (F.conv2d).
    """
    try:
        res = requests.get(stream_url, stream=True)
    except Exception as e:
        print(f"Error al conectar con la cámara: {e}")
        return

    KERNEL_SIZE = 5

    for chunk in res.iter_content(chunk_size=100000):

        if len(chunk) > 100:
            try:
                # Decodificar la imagen
                img_data = BytesIO(chunk)
                frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                
                if frame is None:
                    continue
                    
                height, width, _ = frame.shape
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # --- Tarea 1-B.1: Generar Ruido Gaussiano ---
                mean = 0
                std_dev = 50
                gauss_noise = np.random.normal(mean, std_dev, gray.shape).astype('uint8')
                noisy_image = cv2.add(gray, gauss_noise)
                # ---------------------------------------------

                # --- Tarea 1-B.2: Filtro Gaussiano (OpenCV) ---
                opencv_filtered = cv2.GaussianBlur(noisy_image, (KERNEL_SIZE, KERNEL_SIZE), 0)
                # ---------------------------------------------

                # --- Tarea 1-B.3: Filtro Gaussiano (PyTorch) ---
                
                # 1. Convertir imagen ruidosa a Tensor float32
                # Formato [batch_size, in_channels, height, width]
                img_tensor = torch.from_numpy(noisy_image.astype(np.float32)).unsqueeze(0).unsqueeze(0)

                # 2. Aplicar la convolución 2D
                convolved_output = F.conv2d(
                    input=img_tensor,
                    weight=kernel_torch,
                    padding='same'  # 'same' se encarga del padding automáticamente
                )

                # 3. Convertir de vuelta a imagen numpy (uint8)
                pytorch_filtered = convolved_output.cpu().numpy().squeeze()
                pytorch_filtered = cv2.convertScaleAbs(pytorch_filtered)
                # ---------------------------------------------

                # --- Preparación para visualización (4 paneles) ---
                
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                noisy_bgr = cv2.cvtColor(noisy_image, cv2.COLOR_GRAY2BGR)
                opencv_bgr = cv2.cvtColor(opencv_filtered, cv2.COLOR_GRAY2BGR)
                pytorch_bgr = cv2.cvtColor(pytorch_filtered, cv2.COLOR_GRAY2BGR)

                # Apilar 4 imágenes horizontalmente:
                # [Original] | [Ruidosa] | [Filtro OpenCV] | [Filtro PyTorch]
                total_image = np.zeros((height, width * 4, 3), dtype=np.uint8)
                total_image[:, :width] = gray_bgr
                total_image[:, width:width*2] = noisy_bgr
                total_image[:, width*2:width*3] = opencv_bgr
                total_image[:, width*3:] = pytorch_bgr
                # --------------------------------------

                (flag, encodedImage) = cv2.imencode(".jpg", total_image)
                if not flag:
                    continue

                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                      bytearray(encodedImage) + b'\r\n')

            except Exception as e:
                print(f"Error durante el procesamiento: {e}")
                continue


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_stream")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)