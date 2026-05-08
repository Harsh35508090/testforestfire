import pickle
from flask import Flask, render_template, request, jsonify  ##jsonify = apdu op json file ma malse em
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app =application

## import ridge regressor and standerdscaler pickle
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
standerd_scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

 
        new_scaled_data=standerd_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_scaled_data)

        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')

@app.route("/")
def index():
    return render_template("index.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8090)


    ### Badhu UDEMY ma 170 no. no lacture jovo khali github ma upload karvanu baki chhe baki thai gayu chhe 
    ## badhu lacture ma joine samji jaav 
    
    ##   """ LAGSE VAR PAN AVSE MAJA """