# Author: vlarobbyk
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.
# Updated: 2025-08-11
#
# MODIFICADO POR JOSÉ ABAD (PARA PRÁCTICA 1-B.4: Bordes - CORREGIDO)

from flask import Flask, render_template, Response
from io import BytesIO
import cv2
import numpy as np
import requests
import time

app = Flask(__name__)

# --- Configuración de la Cámara ---
_URL = 'http://192.168.88.74'
_PORT = '81'
_ST = '/stream'
SEP = ':'
stream_url = ''.join([_URL, SEP, _PORT, _ST])
# ----------------------------------

def video_capture():
    """
    Esta función demuestra la importancia del suavizado (CORREGIDO)
    antes de la detección de bordes (Canny y Sobel) en una imagen con ruido.
    """
    try:
        res = requests.get(stream_url, stream=True)
    except Exception as e:
        print(f"Error al conectar con la cámara: {e}")
        return

    # Aumentamos el tamaño del kernel para un suavizado más fuerte
    KERNEL_SIZE = 9

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
                # (Reducimos el ruido a 20)
                mean = 0
                std_dev = 20
                gauss_noise = np.random.normal(mean, std_dev, gray.shape).astype('uint8')
                noisy_image = cv2.add(gray, gauss_noise)
                # ---------------------------------------------

                # --- Tarea 1-B.4: Detección de Bordes ---

                # 1. Bordes SIN suavizado (Resultado incorrecto)
                canny_noisy = cv2.Canny(noisy_image, 50, 150)
                
                # 2. Bordes Sobel SIN suavizado (Resultado incorrecto)
                sobel_noisy = cv2.Sobel(noisy_image, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE)
                sobel_noisy = cv2.convertScaleAbs(sobel_noisy)

                # 3. Bordes CON suavizado (Resultado correcto)
                # Aplicamos un filtro MÁS FUERTE (9x9)
                smoothed_image = cv2.GaussianBlur(noisy_image, (KERNEL_SIZE, KERNEL_SIZE), 0)
                canny_smoothed = cv2.Canny(smoothed_image, 50, 150)
                # ---------------------------------------------

                # --- Preparación para visualización (4 paneles) ---
                
                noisy_bgr = cv2.cvtColor(noisy_image, cv2.COLOR_GRAY2BGR)
                canny_noisy_bgr = cv2.cvtColor(canny_noisy, cv2.COLOR_GRAY2BGR)
                sobel_noisy_bgr = cv2.cvtColor(sobel_noisy, cv2.COLOR_GRAY2BGR)
                canny_smoothed_bgr = cv2.cvtColor(canny_smoothed, cv2.COLOR_GRAY2BGR)

                # Apilar 4 imágenes horizontalmente:
                # [Ruidosa] | [Canny S/Suavizar] | [Sobel S/Suavizar] | [Canny C/Suavizar]
                total_image = np.zeros((height, width * 4, 3), dtype=np.uint8)
                total_image[:, :width] = noisy_bgr
                total_image[:, width:width*2] = canny_noisy_bgr
                total_image[:, width*2:width*3] = sobel_noisy_bgr
                total_image[:, width*3:] = canny_smoothed_bgr
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