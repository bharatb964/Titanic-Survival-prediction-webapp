import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
import pandas as pd


# def create_app(test_config=None):
#     # create and configure the app
#     app = Flask(__name__, instance_relative_config=True)
#     app.config.from_mapping(
#         SECRET_KEY='dev',
#         DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
#     )

#     if test_config is None:
#         # load the instance config, if it exists, when not testing
#         app.config.from_pyfile('config.py', silent=True)
#     else:
#         # load the test config if passed in
#         app.config.from_mapping(test_config)

#     # ensure the instance folder exists
#     try:
#         os.makedirs(app.instance_path)
#     except OSError:
#         pass

#     # a simple page that says hello
#     @app.route('/')
#     @app.route('/index')
#     def index():
#         return flask.render_template('index.html')
    
    
#     def ValuePredictor(to_predict_list):
#         to_predict = np.array(to_predict_list).reshape(1,12)
#         loaded_model = pickle.load(open("model.pkl","rb"))
#         result = loaded_model.predict(to_predict)
#         return result[0]
    
#     @app.route('/result',methods = ['POST', 'GET'])
#     def result():
#         if request.method == 'POST':
#             to_predict_list = request.form.to_dict()
#             to_predict_list=list(to_predict_list.values())
#             to_predict_list = list(map(int, to_predict_list))
#             result = ValuePredictor(to_predict_list)
#             if int(result)==1:
#                 prediction='Income more than 50K'
#             else:
#                 prediction='Income less that 50K'
#             return render_template("result.html",prediction=prediction)

#     return app

app=Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
def addfeatures(data):
    data['Title']=data.Name.str.extract('((?:Mr\.|Mrs\.|Miss\.|Ms\.))')[0].str.lower()
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