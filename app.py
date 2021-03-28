from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from joblib import dump, load
import tensorflow as tf
import keras
from keras.models import load_model
import pandas as pd
import cv2
from summarizer import Summarizer  # BERT model
import sklearn as sk
from sksurv.linear_model import CoxnetSurvivalAnalysis

survival_TR = load('coxnetTR.joblib')
survival_UT = load('coxnetUT.joblib')

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
        print(request.json)
        return "result"


@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')


@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')
