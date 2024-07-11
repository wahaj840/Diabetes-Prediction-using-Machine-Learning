from flask import Flask, request, render_template
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score

import pickle

model = pickle.load(open('logistic_reg_model.pkl', 'rb'))
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    glucose = float(request.form.get('glucose'))
    bmi = float(request.form.get('bmi'))
    dpf = float(request.form.get('dpf'))
    age = float(request.form.get('age'))
    
    data = np.array([[glucose, bmi, dpf, age]])
    data = data.astype(float)

    print("================DATA===================", data)

    pred = model.predict(data)
    results = round(pred[0], 2)
    
    # print("================Prediction===================", pred)

    # Calculating accuracy using the loaded model and testing data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = "{:.2f}".format(accuracy)
    accuracy = float(accuracy) * 100
    
    # print("================Accuracy===================", accuracy)
        
    # Mean Accuracy
    mean_accuracy = np.mean(cross_val_score(model, X_test, y_test, cv=5))
    mean_accuracy = "{:.2f}".format(mean_accuracy)
    mean_accuracy = float(mean_accuracy) * 100

    # Recall score
    # recall = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='recall'))
    recall = recall_score(y_test, y_pred)
    recall = "{:.2f}".format(recall)
    recall = float(recall) * 100

    return render_template('index.html', results=results, accuracy=accuracy, mean_accuracy=mean_accuracy, recall=recall)

if __name__ == '__main__':
    app.run(port=3000, debug=True)