
from flask import Flask,request, jsonify, render_template
import pickle
import numpy as np
 # For older versions of scikit-learn

# Load the pickled model using the correct scikit-learn version

app=Flask(__name__)
"""with open('model.pkl', 'rb') as file:
    model = joblib.load(file)"""
model=pickle.load(open("model.pkl","rb"))
@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction=model.predict(features)
    print(prediction)
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
        
        
    print("Prediction made:", prediction)
    return render_template("index.html",prediction_text="Stress is:{}".format(a) )
    
    
if __name__=="__main__":
    app.run(debug=True)
    
    