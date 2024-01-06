# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:10:41 2024

@author: DELL
"""


"""from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import numpy as np
import seaborn as sb

df=pd.read_csv("Dataset.csv")
print(df.head())

X = df[["snoring_rate", 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen', 'eye_movement', 'sleeping_hours', 'heart_rate',"Working_hrs","Shifts"]]
y = df['stress_level']
X_train, X_test, y_train, y_test=train_test_split(df.iloc[:, :10], df['stress_level'], 
                                                  test_size=0.2, random_state=8)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
#sc=StandardScaler()
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
random_forest_classifier.fit(X_train, y_train)
y_pred = random_forest_classifier.predict(X_test)

#make pickle file
#y_predict = random_forest_classifier.predict([[60.0,20.0,96.0,10.0,95.0,85.0,7.0,60.0 ]])

#print(y_predict)

if (y_pred[0]==0):

    a="rest"

elif (y_pred[0]==1):

    a="Low stress"

elif(y_pred[0]==2):

    a="Medium stress"

elif (y_pred[0]==3):

    a="High stress"

else:

    a="Extreme stress"
pickle.dump(random_forest_classifier,open("model.pkl","wb"))"""
# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = None

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        model = pickle.load(open("model.pkl", "rb"))

    # Extracting features from the form submission
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    
    # Making prediction using the loaded model
    prediction = model.predict(features)
    stress_levels = ['Rest', 'Low stress', 'Medium stress', 'High stress', 'Extreme stress']
    predicted_stress = stress_levels[int(prediction[0])]
    
    return render_template("index.html", prediction_text="Stress level is: {}".format(predicted_stress))

if __name__ == "__main__":
    # Loading the dataset and training the model
    df = pd.read_csv("Dataset.csv")
    X = df[["snoring_rate", 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen', 'eye_movement', 'sleeping_hours', 'heart_rate',"Working_hrs","Shifts"]]
    y = df['stress_level']
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :10], df['stress_level'], test_size=0.2, random_state=8)

    # Create a Random Forest classifier
    random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    random_forest_classifier.fit(X_train, y_train)
    
    # Saving the trained model to a file
    pickle.dump(random_forest_classifier, open("model.pkl", "wb"))

    # Starting the Flask app
    app.run(debug=True)


