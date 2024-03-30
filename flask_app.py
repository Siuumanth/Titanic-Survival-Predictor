import pickle
import pandas as pd
from flask import Flask,render_template,redirect,url_for,request
from flask_cors import CORS
from Kaggle_Contest_main import Titanic


with open('saved_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

app=Flask(__name__)
CORS(app)

@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/submit',methods=['POST','GET'])
def submitted():
    if request.method=='POST':

        pclass=float(request.form['pclass'])
        sex =request.form['sex']
        age =float(request.form['age']) 
        sibsp =float(request.form['sibsp'])
        parch =float(request.form['parch'])
        fare =float(request.form['fare'])
    if sex=='M' or sex.lower()=='male':
        sex=int(1)
    else:
        sex=int(0)

    input_passenger={
        'Pclass':pclass,
        'Sex':sex,
        'Age':age,
        'SibSp':sibsp,
        'Parch':parch,
        'Fare':fare
        }
    
    input_df=pd.DataFrame(input_passenger,index=[0])

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X_input = input_df[features]

    prediction = loaded_model.predict(X_input)
    
    final=int(prediction[0])
    return redirect(url_for("result",res=final))


@app.route('/result/<int:res>')
def result(res):
    sur=""
    if(res==1):
        sur="Good News!! The passenger has SURVIVED!!"
    else:
        sur="Unfortunately, the passenger did not survive :("
    ac=int(Titanic.calculate_accuracy()*100)
    ac=str(ac)+'%'
    return render_template('result.html',survival=sur,acc=ac)

    


if __name__=='__main__':
    app.run(debug=True)
