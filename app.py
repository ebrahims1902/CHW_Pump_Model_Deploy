from flask import Flask, render_template, request
import pickle
import numpy as np
from numpy.core.numeric import outer
import jsonify
import requests
import sklearn
import pandas as pd

# app = Flask(__name__)
app = Flask(__name__)

# list_of_model_pickles = ['CHW_Pump_1_Speed(%)_MODEL.pkl'] # add any model pickle file here

model = pickle.load(open('CHW_Pump_1_Speed(%)_MODEL.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('chw_Pump_1_Speed.html')


@app.route("/predict", methods = ["GET", "POST"])
def predict():
    # prediction_text_all_pickles = 'Water prediction should be:\n'
    # for model_file in list_of_model_pickles:
        # f_pickle = open(model_file, 'rb')

        # model = pickle.load(f_pickle)
    float_features = [np.float64(x) for x in request.form.values()]
    final = [np.array(float_features)]
    # final = [['1', '2', '3', '4', '5', '6', '7', '8', '9', '90']]
    prediction = model.predict(final)
        # prediction_text_all_pickles += f' {prediction} For {model_file}. \n'
        # f_pickle.close()
        
    # return prediction
    return render_template('chw_Pump_1_Speed.html', pred=prediction)

    # return render_template('chw_Pump_1_Speed.html', pred='Your CHW_Pump_1_Speed Percentage Prediction is {}'.format(prediction_CHW))
    # return render_template('chw_Pump_1_Speed.html', pred=prediction)
    # return render_template('chw_Pump_1_Speed.html',  pred=prediction_text_all_pickles)


if __name__ == '__main__':
    app.run(debug=True)
