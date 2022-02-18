from flask import Flask, render_template, Response, url_for
import cv2 as cv
from face_detector import face_detector

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    vid = cv.VideoCapture(0)
    success, dfs = vid.read()
    if success:
        return Response(face_detector(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
