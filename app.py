from flask import Flask, request, jsonify, render_template
import pyttsx3
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import re
import csv

app = Flask(__name__)

# Load essential functions and data from the notebook (either copy code or save as a separate module)
# For instance, import text_to_speech, calc_condition, sec_predict, etc.

@app.route('/')
def home():
    return render_template('index.html')  # A simple homepage where users can enter symptoms.

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms').split(',')  # Get symptoms from the form
    prediction = sec_predict(symptoms)
    disease = print_disease(prediction)
    response = {'disease': disease}
    return jsonify(response)
