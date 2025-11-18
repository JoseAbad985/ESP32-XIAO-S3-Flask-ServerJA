from flask import Flask, render_template, Response
from io import BytesIO
import cv2
import numpy as np
import requests
import time

app = Flask(__name__)

_URL = 'http://192.168.88.74'
_PORT = '81'
_ST = '/stream'
SEP = ':'
stream_url = ''.join([_URL, SEP, _PORT, _ST])

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def video_capture():
    try:
        res = requests.get(stream_url, stream=True)
    except Exception:
        return

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gamma_value = 1.8 

    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                
                if frame is None:
                    continue
                    
                height, width, _ = frame.shape
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                hist_global = cv2.equalizeHist(gray)
                hist_clahe = clahe.apply(gray)
                gamma_corrected = adjust_gamma(frame, gamma=gamma_value)

                hist_global_bgr = cv2.cvtColor(hist_global, cv2.COLOR_GRAY2BGR)
                
                # --- LÍNEA CORREGIDA ---
                # Antes decía: cv2.cvtColor(clahe, ...)
                # Ahora dice: cv2.cvtColor(hist_clahe, ...)
                clahe_bgr = cv2.cvtColor(hist_clahe, cv2.COLOR_GRAY2BGR)

                total_image = np.zeros((height, width * 4, 3), dtype=np.uint8)
                total_image[:, :width] = frame
                total_image[:, width:width*2] = hist_global_bgr
                total_image[:, width*2:width*3] = clahe_bgr
                total_image[:, width*3:] = gamma_corrected

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