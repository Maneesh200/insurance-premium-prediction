from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('artifacts/ridge_regression_model.pkl')

@app.route('/', methods=['GET','POST'])
def home():
    prediction = None

    if request.method == 'POST':
        input_data = {
            "age" : int(request.form['age']),
            "gender" : request.form['gender'],
            "bmi" : float(request.form['bmi']),
            "children" : int(request.form['children']),
            "smoker" : request.form['smoker'],
            "region" : request.form['region'],
            "medical_history" : request.form['medical_history'],
            "family_medical_history" : request.form['family_medical_history'],
            "exercise_frequency" : request.form['exercise_frequency'],
            "occupation" : request.form['occupation'],
            "coverage_level" : request.form['coverage_level']
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)