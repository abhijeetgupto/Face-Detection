import base64
import cv2
import io

import numpy as np
from PIL import Image
from engineio.payload import Payload
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import mediapipe as mp

Payload.max_decode_packets = 2048

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx + 7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)

    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)



@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)



@socketio.on('image')
def image(data_image):
    frame = (readb64(data_image))
    frame = face_detector(frame)

    imgencode = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)


def face_detector(img):
    mpFaceDetection = mp.solutions.face_detection
    faceDetection = mpFaceDetection.FaceDetection()

    img = cv2.flip(img, 1)
    results = faceDetection.process(img)
    if results.detections:
        for id, detection in enumerate(results.detections):
            if detection.score[0] > 0.50:
                bboxc = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), int(bboxc.width * iw), int(bboxc.height * ih)
                cv2.rectangle(img, bbox, (0, 255, 0), 2)
                cv2.putText(img, f"{int(detection.score[0] * 100)}%",
                            (int(bboxc.xmin * iw), int(bboxc.ymin * ih - 15)),
                            cv2.FONT_ITALIC, 1.5, (255, 255, 255), 2)

    cv2.waitKey(0)
    return img


if __name__ == '__main__':
    socketio.run(app, port=9990, debug=True)
