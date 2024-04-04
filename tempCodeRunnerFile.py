from flask import Flask, redirect, url_for, render_template, Response
from threading import Thread
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)


model = load_model("H:/Deployment/model/my_model.h5")


class_labels = {
    0: "Class 0",
    1: "Class 1",

}


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def process_frames():
    label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'Is', 'J', 'K', 'L', 'M', 'N','Name', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','What', 'X', 'Y','Your', 'Z', 'blank']
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
        cropframe = frame[40:300, 0:300]
        cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
        cropframe = cv2.resize(cropframe, (48, 48))
        cropframe = extract_features(cropframe)
        pred = model.predict(cropframe)
        prediction_label = label[np.argmax(pred)]
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
        if prediction_label == 'blank':
            cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            accu = "{:.2f}".format(np.max(pred) * 100)
            cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 0, 255), 2, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/',endpoint='main_page')
def index():
    return render_template('index.html')

@app.route('/result.html')
def result():
    return render_template('result.html')

@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/redirect_to_index')
def redirect_to_index():
    return redirect('index')

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    Thread(target=process_frames).start()
    app.run(debug=True)
