from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/records')
def view_records():
    return render_template('elements.html')


@app.route('/records/<record>')
def records(record):
    print(record)
    return render_template('elements.html')


@app.route('/add')
def add():
    return render_template('elements.html')


@app.route('/register')
def register():
    return render_template('elements.html')


@app.route('/login')
def login():
    return render_template('elements.html')
