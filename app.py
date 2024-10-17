from flask import Flask, request, jsonify, render_template
from disease_model import (
    getSeverityDict, getDescription, getprecautionDict, check_pattern,
    print_disease, sec_predict, calc_condition, tree_to_code, text_to_speech
)

import pyttsx3
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load Datasets for training and testing
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')

# Data preparation
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Encode the target labels
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Decision Tree Model Implementation
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Optional: Creating Support Vector Machine Model
model = SVC()
model.fit(x_train, y_train)
print("Accuracy score for SVM:", model.score(x_test, y_test))

# Initialize dictionaries for severity, description, and precaution
severityDictionary = getSeverityDict()
description_list = getDescription()
precautionDictionary = getprecautionDict()

# Initialize symptoms dictionary
symptoms_dict = {symptom: index for index, symptom in enumerate(x.columns)}

@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html has a form for symptom input

@app.route('/predict', methods=['POST'])
def predict():
    # Get the symptom input from the form
    data = request.form.get('symptom', '').strip()  # Safely get the input and strip any extra spaces
    print(f"User symptom input: {data}")
    
    if not data:
        return jsonify({"error": "No symptom provided"}), 400

    try:
        # Call tree_to_code for prediction
        prediction = tree_to_code(clf, data)  # Assuming tree_to_code can handle a single symptom
        # Using text-to-speech to announce the result (optional)
        engine.say(f"The predicted condition is {prediction}")
        engine.runAndWait()
        
        return jsonify(prediction=prediction)
    except KeyError as e:
        print(f"Symptom not recognized: {str(e)}")
        return jsonify(error="Symptom not recognized"), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify(error="An unexpected error occurred"), 500

if __name__ == '__main__':
    app.run(debug=True)
