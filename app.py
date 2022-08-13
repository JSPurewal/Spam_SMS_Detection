# -*- coding: utf-8 -*-
"""
@author: Jaskaran S. Purewal
"""
import numpy as np
from flask import Flask, render_template, request
import pickle

app=Flask(__name__)
model=pickle.load(open('models/modelSMS.pkl','rb'))
trans=pickle.load(open('models/vectorizer.pkl','rb'))



@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    message=request.form['message']
    messageData=[message]
    vect=trans.transform(messageData).toarray()
    ans=model.predict(vect)
    return render_template('result.html',prediction=ans)
    

if __name__=="__main__":
    app.run()