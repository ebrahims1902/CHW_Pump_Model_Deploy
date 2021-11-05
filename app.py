from flask import Flask, render_template, request
import pickle
import numpy as np
from numpy.core.numeric import outer
import jsonify
import requests
import sklearn
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('CHW_Pump_1_Speed(%)_MODEL.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('home.html')


@app.route("/predict", methods = ["GET", "POST"])
def predict():
    float_features = [[np.float64(x) for x in request.form.values()]]
    final = np.array(float_features)
    prediction = model.predict(final)
    return render_template('predict.html', pred=prediction)

if __name__ == '__main__':
    app.run(debug=True)
