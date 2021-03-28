from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from joblib import dump, load
import tensorflow as tf
import keras
from keras.models import load_model
import pandas as pd
import numpy as np
import cv2
import urllib
from summarizer import Summarizer  # BERT model
import sklearn as sk
from sksurv.linear_model import CoxnetSurvivalAnalysis

survival_TR = load('coxnetTR.joblib')
survival_UT = load('coxnetUT.joblib')

summarizer = Summarizer()

dr_weights = load_model("dr_weights.h5")

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/records', methods=['GET'])
def view_records():
    return render_template('records.html')


@app.route('/records/<record>', methods=['GET'])
def records(record):
    return render_template('record.html', record=record)


@app.route('/add', methods=['GET', 'POST'])
def add():
    if(request.method == 'GET'):
        return render_template('add.html')
    else:
        json = request.json
        if(json["type"] == 1):
            if("image" not in json or json["image"] == ""):
                return '{"type":"error","response":"Image field must not be left blank."}'

            req = urllib.request.urlopen(json["image"])
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (224, 224))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            image_tensor = tf.expand_dims(image_tensor, 0)
            # image_tensor = tf.expand_dims(image_tensor, 2)
            result = dr_weights.predict(image_tensor)[0]
            highestVal = 0
            highestIndex = 0
            print(result)
            for index, value in enumerate(result):
                if(value > highestVal):
                    highestVal = value
                    highestIndex = index
            output = "No diabetic retinopathy."
            if highestIndex == 1:
                output = "Mild diabetic retinopathy."
            elif highestIndex == 2:
                output = "Moderate diabetic retinopathy."
            elif highestIndex == 3:
                output = "Severe diabetic retinopathy."
            elif highestIndex == 4:
                output = "Proliferative diabetic retinopathy."

            return '{"type":"success","response":"' + output + '"}'
        elif(json["type"] == 2):
            pass
        elif(json["type"] == 3):
            if("content" not in json or json["content"] == ""):
                return '{"type":"error","response":"Report field must not be left blank."}'
            output = summarizer(json["content"], num_sentences=3)
            return '{"type":"success","response":"' + output + '"}'
        else:
            return '{"type":"error","response":"Invalid request, please try again."}'
        return '{"type":"success","response":"result"}'


@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')


@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')
