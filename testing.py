# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 19:19:44 2021
@author: Sonu
"""

import time
import cv2
from flask import Flask, render_template, Response
import mediapipe as mp
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

rf_model = load('RFC_model.sav')

mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
    cap = cv2.VideoCapture(0)
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            curr_time = time.time()
            
            ret, frame = cap.read()

            # BGR 2 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip on horizontal
            image = cv2.flip(image, 1)

            # Set flag
            image.flags.writeable = False

            # Detections
            results = holistic.process(image)
            
            # get image shape
            image_height, image_width, _ = image.shape

            # Set flag to true
            image.flags.writeable = True

            # RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if results.right_hand_landmarks or results.left_hand_landmarks:
                # Mengambil Pose landmarks
                lh = list(np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3))
                rh = list(np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3))

                # Satukan baris
                row = lh+rh

                # Tambah class name 
                X = pd.DataFrame([row])
                hand_class = rf_model.predict(X)[0]
                
                cv2.rectangle(image, (0,0), (120, 40), (245, 117, 16), -1)
                cv2.putText(image, 'CLASS',(10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, hand_class.split(' ')[0], (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            
            #cv2.imshow('Testing Model', image)
            
            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            key = cv2.waitKey(20)
            if key == 27:
                break

@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__=="__main__":
    app.run(debug=True)