import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
import pandas as pd

app=Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
def addfeatures(data):
    if data.Name.str.contains('((?:Mr\.|Mrs\.|Miss\.|Ms\.))').any():
        data['Title']=data.Name.str.extract('((?:Mr\.|Mrs\.|Miss\.|Ms\.))')[0].str.lower()
    else:
        data['Title']=['mr.']
    data['cabin_class']=data.Cabin.str.extract('(^[a-zA-Z])')[0].str.lower()
    data['ticket_class']=data.Ticket.str.extract('^([\w\-]+)')[0].str.extract('([A-Za-z])')
    data['Title']=data['Title'].fillna(data['Title'].mode()[0])
    data['ticket_class']=data['ticket_class'].fillna(data['ticket_class'].mode()[0])
    data=data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
    return data

def ValuePredictor(to_predict_list):
    val=pd.DataFrame.from_dict(to_predict_list,orient='index').T
    val['PassengerId']=0
    val['Ticket']='CA. 2343'
    val=addfeatures(val)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(val)
    return result

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        prediction=ValuePredictor(to_predict_list)[0]
        if prediction==0:
            res='Not Survived'
        else:
            res='Survived'
        return render_template("result.html",prediction=res)

if __name__ == '__main__':
    app.run()
