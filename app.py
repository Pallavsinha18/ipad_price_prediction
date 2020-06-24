# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:06:56 2020

@author: Pallav Kumar Sinha
"""


import pandas as pd
import  numpy as np
import pickle as pk
from flask import Flask, render_template, jsonify,request

# Initialising flask application


app = Flask(__name__)
model= pk.load(open('model.pkl','rb'))


## Routing the application to root folder
@app.route('/')
def home():
    return render_template('index.html')

# routing to prediction outcome
@app.route('/predict',methods=['POST'])    

def predict():
    int_feature= [int(x) for x in request.form.values()] 
    final_data = [np.array(int_feature)]
    prediction= model.predict(final_data)
    output= prediction[0] 
        
    return render_template('index.html', prediction_text= 'prediction of IPAD is USD {}'.format(round(output),2))

if __name__== '__main__':
    app.run(debug=True)              