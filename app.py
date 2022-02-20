from flask import Flask, render_template, Response, url_for, redirect
import cv2 as cv
from face_detector import face_detector
from flask_wtf import FlaskForm
from wtforms import SubmitField

app = Flask(__name__)
app.config['SECRET_KEY'] = 'password'


class InfoForm(FlaskForm):
    submit = SubmitField(label="Submit")

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InfoForm()

    if form.validate_on_submit():
        return redirect(url_for('home'))

    return render_template('home.html', form=form)


@app.route('/home',methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/video',methods=['GET', 'POST'])
def video():
    vid = cv.VideoCapture(0)
    success, dfs = vid.read()
    if success:
        return Response(face_detector(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
