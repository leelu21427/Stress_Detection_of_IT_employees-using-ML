# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:43:40 2024

@author: DELL
"""
from flask import Flask,request, jsonify, render_template
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))
@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction=model.predict(features)
    if(prediction==0):
        a="rest"
    elif(prediction==1):
        a="low"
    elif(prediction==2):
        a="medium"
    elif(prediction==3):
        a="High"
    else:
        a="Extreme"
        
        
    #print("Prediction made:", prediction)
    return render_template("index.html",prediction_text="Stress is:{}".format(a) )
    
    
if __name__=="__main__":
    app.run(debug=False,host="0.0.0.0")
    
    