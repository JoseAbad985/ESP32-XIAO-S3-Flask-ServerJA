from flask import Flask, render_template, Response
from io import BytesIO
import cv2
import numpy as np
import requests

app = Flask(__name__)

_URL = 'http://192.168.88.74'
_PORT = '81'
_ST = '/stream'
SEP = ':'
stream_url = ''.join([_URL, SEP, _PORT, _ST])

def add_gaussian_noise(image, mean=0, std_dev=30):
    # --- ESTA ES LA FUNCIÓN ARREGLADA ---
    # Convertir a float32 para cálculos precisos
    image_float = image.astype(np.float32)
    
    # Generar ruido float32
    gauss_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
    
    # Sumar como floats
    noisy_image_float = image_float + gauss_noise
    
    # Recortar (clip) el resultado al rango válido de 8 bits [0, 255]
    np.clip(noisy_image_float, 0, 255, out=noisy_image_float)
    
    # Convertir de vuelta a uint8 para mostrar
    noisy_image = noisy_image_float.astype(np.uint8)
    return noisy_image
    # --- FIN DE LA CORRECCIÓN ---

def video_capture():
    try:
        res = requests.get(stream_url, stream=True)
    except Exception:
        return

    KERNEL_SIZE = 5

    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                
                if frame is None:
                    continue
                    
                height, width, _ = frame.shape

                # Ahora esta función SÍ aplicará el ruido correctamente
                noisy_image = add_gaussian_noise(frame, std_dev=40)

                median_filtered = cv2.medianBlur(noisy_image, KERNEL_SIZE)
                blur_filtered = cv2.blur(noisy_image, (KERNEL_SIZE, KERNEL_SIZE))
                gaussian_filtered = cv2.GaussianBlur(noisy_image, (KERNEL_SIZE, KERNEL_SIZE), 0)

                total_image = np.zeros((height, width * 4, 3), dtype=np.uint8)
                total_image[:, :width] = noisy_image
                total_image[:, width:width*2] = median_filtered
                total_image[:, width*2:width*3] = blur_filtered
                total_image[:, width*3:] = gaussian_filtered

                (flag, encodedImage) = cv2.imencode(".jpg", total_image)
                if not flag:
                    continue

                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                      bytearray(encodedImage) + b'\r\n')

            except Exception:
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