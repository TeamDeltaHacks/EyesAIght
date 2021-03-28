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
            try:
                if("laser_type" not in json or json["laser_type"] == ""):
                    return '{"type":"error","response":"Laser type field must not be left blank."}'
                json["laser_type"] = int(json["laser_type"])
                if(json["laser_type"] != 1 and json["laser_type"] != 2):
                    return '{"type":"error","response":"Laser type field must not be left blank."}'

                if("eye" not in json or json["eye"] == ""):
                    return '{"type":"error","response":"Treated eye field must not be left blank."}'
                json["eye"] = int(json["eye"])
                if(json["eye"] != 1 and json["eye"] != 2):
                    return '{"type":"error","response":"Treated eye field must not be left blank."}'

                if("age" not in json or json["age"] == ""):
                    return '{"type":"error","response":"Age field must not be left blank."}'
                json["age"] = int(json["age"])
                if(json["age"] < 1):
                    json["age"] = 1
                if(json["age"] > 58):
                    json["age"] = 58

                if("diabetes_type" not in json or json["diabetes_type"] == ""):
                    return '{"type":"error","response":"Diabetes type field must not be left blank."}'
                json["diabetes_type"] = int(json["diabetes_type"])
                if(json["diabetes_type"] != 1 and json["diabetes_type"] != 2):
                    return '{"type":"error","response":"Diabetes type field must not be left blank."}'

                if("risk_untreated" not in json or json["risk_untreated"] == ""):
                    return '{"type":"error","response":"Untreated risk field must not be left blank."}'
                json["risk_untreated"] = int(json["risk_untreated"])
                if(json["risk_untreated"] < 6 or json["risk_untreated"] > 12):
                    return '{"type":"error","response":"Untreated risk field must not be left blank."}'

                if("risk_treated" not in json or json["risk_treated"] == ""):
                    return '{"type":"error","response":"Treated risk field must not be left blank."}'
                json["risk_treated"] = int(json["risk_treated"])
                if(json["risk_treated"] < 6 or json["risk_treated"] > 12):
                    return '{"type":"error","response":"Treated risk field must not be left blank."}'

                column_names = ['ID', 'Laser Type', 'Eye', 'Age', 'Type', 'Treated Group',
                                'Treated Status', 'Treated Time', 'Untreated Group', 'Untreated Status', 'Untreated Time']
                raw_ds = pd.read_csv('drdata.csv', na_values="NaN")
                raw_ds.columns = column_names
                dataset = raw_ds.copy()
                dataset2 = raw_ds.copy()
                dataset = dataset.drop(
                    columns=['ID', 'Treated Group', 'Treated Status', 'Treated Time', 'Laser Type'])
                dataset['Untreated Status'] = (
                    dataset['Untreated Status'] == 1).astype(bool)

                dataset2 = dataset2.drop(
                    columns=['ID', 'Untreated Group', 'Untreated Status', 'Untreated Time'])
                dataset2['Treated Status'] = (
                    dataset2['Treated Status'] == 1).astype(bool)

                X = dataset.iloc[:, :-2]
                X2 = dataset2.iloc[:, :-2]

                X_data = {'Age': [json["age"]], 'Eye': [json["eye"]], 'Type': [
                    json["diabetes_type"]], 'Untreated Group': [json["risk_untreated"]]}
                X2_data = {'Age': [json["age"]], 'Laser Type': [json["laser_type"]], 'Eye': [
                    json["eye"]], 'Type': [json["diabetes_type"]], 'Treated Group': [json["risk_treated"]]}
                X_dataf = pd.DataFrame(data=X_data)
                X2_dataf = pd.DataFrame(data=X2_data)
                X_dataf = X_dataf.append(X)
                X2_dataf = X2_dataf.append(X2)

                non_dummy_cols = ['Age']
                dummy_cols = list(set(X_dataf.columns) - set(non_dummy_cols))
                X_dataf = pd.get_dummies(X_dataf, columns=dummy_cols)

                dummy_cols2 = list(set(X2_dataf.columns) - set(non_dummy_cols))
                X2_dataf = pd.get_dummies(X2_dataf, columns=dummy_cols2)

                dataUT = X_dataf.iloc[:1]
                dataTR = X2_dataf.iloc[:1]
                print("2")
                print(dataUT)
                print(dataTR)
            except:
                return '{"type":"error","response":"Invalid request, please try again."}'
        elif(json["type"] == 3):
            if("content" not in json or json["content"] == ""):
                return '{"type":"error","response":"Report field must not be left blank."}'

            if("sentences" not in json or json["sentences"] == ""):
                return '{"type":"error","response":"Sentences field must not be left blank."}'
            json["sentences"] = int(json["sentences"])
            if(json["sentences"] < 2):
                json["sentences"] = 2

            output = summarizer(
                json["content"], num_sentences=json["sentences"])
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
