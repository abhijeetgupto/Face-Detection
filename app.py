import base64
import cv2
import io

import numpy as np
from PIL import Image
from engineio.payload import Payload
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from face_detector import face_detector

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



if __name__ == '__main__':
    socketio.run(app, port=9990, debug=True)
