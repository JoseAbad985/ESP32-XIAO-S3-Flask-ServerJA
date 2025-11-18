from flask import Flask, render_template, Response
from io import BytesIO
import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F

app = Flask(__name__)

_URL = 'http://192.168.88.74'
_PORT = '81'
_ST = '/stream'
SEP = ':'
stream_url = ''.join([_URL, SEP, _PORT, _ST])

torch_kernel = torch.tensor([
    [1, 4,  7,  4,  1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4,  7,  4,  1]
], dtype=torch.float32) / 273.0
torch_kernel = torch_kernel.unsqueeze(0).unsqueeze(0)

def add_gaussian_noise_gray(image, mean=0, std_dev=30):
    # --- ESTA FUNCIÓN ESTÁ CORREGIDA ---
    # Esta función SÍ ACEPTA GRISES
    image_int16 = image.astype(np.int16)
    noise = np.zeros(image.shape, np.int16)
    cv2.randn(noise, mean, std_dev) 
    noisy_image_int16 = image_int16 + noise
    np.clip(noisy_image_int16, 0, 255, out=noisy_image_int16)
    noisy_image = noisy_image_int16.astype(np.uint8)
    return noisy_image

def video_capture():
    try:
        res = requests.get(stream_url, stream=True)
    except Exception:
        return

    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                
                if frame is None:
                    continue
                    
                height, width, _ = frame.shape
                
                # Convertir a gris ANTES de añadir ruido
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Aplicar la función de ruido que SÍ funciona en grises
                noisy_image = add_gaussian_noise_gray(gray, std_dev=50)

                opencv_filtered = cv2.GaussianBlur(noisy_image, (5, 5), 0)

                img_tensor = torch.from_numpy(noisy_image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                convolved_output = F.conv2d(
                    input=img_tensor,
                    weight=torch_kernel,
                    padding='same'
                )
                pytorch_filtered = convolved_output.cpu().numpy().squeeze()
                pytorch_filtered = cv2.convertScaleAbs(pytorch_filtered)
                
                noisy_bgr = cv2.cvtColor(noisy_image, cv2.COLOR_GRAY2BGR)
                opencv_bgr = cv2.cvtColor(opencv_filtered, cv2.COLOR_GRAY2BGR)
                pytorch_bgr = cv2.cvtColor(pytorch_filtered, cv2.COLOR_GRAY2BGR)

                total_image = np.zeros((height, width * 3, 3), dtype=np.uint8)
                total_image[:, :width] = noisy_bgr
                total_image[:, width:width*2] = opencv_bgr
                total_image[:, width*2:] = pytorch_bgr

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