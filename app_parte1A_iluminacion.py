
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


def video_capture():
    """
    Esta función aplica sustracción de fondo (MOG2) y
    compara métodos de ecualización de histograma (Global vs CLAHE).
    """
    try:
        res = requests.get(stream_url, stream=True)
    except Exception as e:
        print(f"Error al conectar con la cámara: {e}")
        return

    backSub = cv2.createBackgroundSubtractorMOG2()
    prev_time = 0
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


    for chunk in res.iter_content(chunk_size=100000):

        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                
                if frame is None:
                    continue
                    
                height, width, _ = frame.shape

                new_time = time.time()
                if (new_time - prev_time) > 0:
                    fps = 1 / (new_time - prev_time)
                else:
                    fps = 0
                prev_time = new_time
                fgMask = backSub.apply(frame)
                foreground = cv2.bitwise_and(frame, frame, mask=fgMask)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist_global = cv2.equalizeHist(gray)
                hist_clahe = clahe.apply(gray)
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                hist_global_bgr = cv2.cvtColor(hist_global, cv2.COLOR_GRAY2BGR)
                hist_clahe_bgr = cv2.cvtColor(hist_clahe, cv2.COLOR_GRAY2BGR)
                
                total_image = np.zeros((height, width * 4, 3), dtype=np.uint8)
                total_image[:, :width] = frame
                total_image[:, width:width*2] = foreground
                total_image[:, width*2:width*3] = hist_global_bgr
                total_image[:, width*3:] = hist_clahe_bgr
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